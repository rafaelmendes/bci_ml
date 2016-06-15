import numpy as np # numpy - used for array and matrices operations
import math as math # used for basic mathematical operations

import scipy.signal as sp
import scipy.linalg as lg

class CSP:
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).
    This object can be used as a supervised decomposition to estimate
    spatial filters for feature extraction in a 2 class decoding problem.
    CSP in the context of EEG was first described in [1]; a comprehensive
    tutorial on CSP can be found in [2].
    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals.
        This number should be set by cross-validation.
    reg : float | str | None (default None)
        if not None, allow regularization for covariance estimation
        if float, shrinkage covariance is used (0 <= shrinkage <= 1).
        if str, optimal shrinkage using Ledoit-Wolf Shrinkage ('ledoit_wolf')
        or Oracle Approximating Shrinkage ('oas').
    log : bool (default True)
        If true, apply log to standardize the features.
        If false, features are just z-scored.
    cov_est : str (default 'concat')
        If 'concat', covariance matrices are estimated on concatenated epochs
        for each class.
        If 'epoch', covariance matrices are estimated on each epoch separately
        and then averaged over each class.
    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_channels,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_channels,)
        If fit, the std squared power for each component.
    """

    def __init__(self, n_components=4, reg=None, log=True, cov_est="concat"):
        """Init of CSP."""
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_est = cov_est
        self.filters_ = None
        self.patterns_ = None
        self.mean_ = None
        self.std_ = None

    def get_params(self, deep=True):
        """Return all parameters (mimics sklearn API).
        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        """
        params = {"n_components": self.n_components,
                  "reg": self.reg,
                  "log": self.log}
        return params

    def fit(self, epochs_data, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        epochs_data : ndarray, shape (n_epochs, n_channels, n_times)
            The data to estimate the CSP on.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """

        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        epochs_data = np.atleast_3d(epochs_data)
        e, c, t = epochs_data.shape
        # check number of epochs
        if e != len(y):
            raise ValueError("n_epochs must be the same for epochs_data and y")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("More than two different classes in the data.")
        if not (self.cov_est == "concat" or self.cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")

        if self.cov_est == "concat":  # concatenate epochs
            class_1 = np.transpose(epochs_data[y == classes[0]],
                                   [1, 0, 2]).reshape(c, -1)
            class_2 = np.transpose(epochs_data[y == classes[1]],
                                   [1, 0, 2]).reshape(c, -1)
            cov_1 = _regularized_covariance(class_1, reg=self.reg)
            cov_2 = _regularized_covariance(class_2, reg=self.reg)
        elif self.cov_est == "epoch":
            class_1 = epochs_data[y == classes[0]]
            class_2 = epochs_data[y == classes[1]]
            cov_1 = np.zeros((c, c))
            for t in class_1:
                cov_1 += _regularized_covariance(t, reg=self.reg)
            cov_1 /= class_1.shape[0]
            cov_2 = np.zeros((c, c))
            for t in class_2:
                cov_2 += _regularized_covariance(t, reg=self.reg)
            cov_2 /= class_2.shape[0]

        # normalize by trace
        cov_1 /= np.trace(cov_1)
        cov_2 /= np.trace(cov_2)

        e, w = lg.eigh(cov_1, cov_1 + cov_2)
        n_vals = len(e)
        # Rearrange vectors
        ind = np.empty(n_vals, dtype=int)
        ind[::2] = np.arange(n_vals - 1, n_vals // 2 - 1, -1)
        ind[1::2] = np.arange(0, n_vals // 2)
        w = w[:, ind]  # first, last, second, second last, third, ...
        self.filters_ = w.T
        self.patterns_ = lg.pinv(w)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, epochs_data, y=None):
        """Estimate epochs sources given the CSP filters.
        Parameters
        ----------
        epochs_data : array, shape (n_epochs, n_channels, n_times)
            The data.
        y : None
            Not used.
        Returns
        -------
        X : ndarray of shape (n_epochs, n_sources)
            The CSP features averaged over time.
        """
        if not isinstance(epochs_data, np.ndarray):
            raise ValueError("epochs_data should be of type ndarray (got %s)."
                             % type(epochs_data))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in epochs_data])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)
        if self.log:
            X = np.log(X)
        else:
            X -= self.mean_
            X /= self.std_
        return X

def _regularized_covariance(data, reg=None):

    if reg is None:
        cov = np.cov(data)
    
    return cov