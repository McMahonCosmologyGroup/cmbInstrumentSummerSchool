import numpy


class smooth_hann:
    def __init__(self, x, window="hanning"):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        self.x = x
        # self.window_len = window_len
        self.window = window
        return

    def smooth(self, window_len=11):
        if self.x.ndim != 1:
            print("smooth only accepts 1 dimension arrays.")
        #         raise ValueError "smooth only accepts 1 dimension arrays."

        if self.x.size < window_len:
            print("Input vector needs to be bigger than window size.")
        #         raise ValueError "Input vector needs to be bigger than window size."

        if window_len < 3:
            return self.x

        if not self.window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            print(
                "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
            )
        #         raise ValueError "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = numpy.r_[
            self.x[window_len - 1 : 0 : -1], self.x, self.x[-2 : -window_len - 1 : -1]
        ]
        # print(len(s))
        if self.window == "flat":  # moving average
            w = numpy.ones(window_len, "d")
        else:
            w = eval("numpy." + self.window + "(window_len)")

        y = numpy.convolve(w / w.sum(), s, mode="valid")
        return y
