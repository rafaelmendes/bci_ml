# Import a lot of functions to keep it simple to use
from pylab import * 
import matplotlib as plt

from scipy.signal import remez 
from scipy.signal import freqz # Plots filter frequency response
from scipy.signal import lfilter # Applies designed filter to data
from scipy.signal import firwin 

def mfreqz(b,a=1):
    w,h = freqz(b,a)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w/max(w),h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

sample_rate = 250
# 1 second of data samples at spacing of 1/1000 seconds
t = arange(0, 1, 1.0/250)

noise_amp = 5.0

A = 100
s = A*sin(2*pi*20*t)+A*sin(2*pi*200*t)+noise_amp * randn(len(t))
# s = sin(2*pi*100*t)+sin(2*pi*200*t)

# Note: you may need to use fft.fft is you are using ipython
ft = fft(s)/len(s)
subplot(411)
plot(s)

# subplot(412)
# plot(20*log10(abs(ft)))
# show()

# lpf = remez(21, [0, 0.2, 0.3, 0.5], [1.0, 0.0])
# w, h = freqz(lpf)
# subplot(413)
# plot(w/(2*pi), 20*log10(abs(h)))
# show()
# sout = lfilter(lpf, 1, s)
# subplot(414)
# plot(20*log10(abs(fft(s))))
# plot(20*log10(abs(fft(sout))))
# show()


filter_order = 255
f1, f2 = 8.0, 30.0
nyq_rate = sample_rate / 2

w1, w2 = f1 / nyq_rate, f2 / nyq_rate
filter_coef = firwin(filter_order, [w1, w2] , pass_zero=False)
# plot(w, 20*log10(abs(h)))
# show()

w, h = freqz(filter_coef)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
show()