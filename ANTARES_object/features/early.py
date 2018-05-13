import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class EarlyMixin(object):
    """
    Methods to derive early classification features for LAobjects
    """

    def _fit_early_lightcurve(self, time, flux, fluxerr):
        """
        Return tsquarize fit to early light curve
        """
        def fit_func(t, a, c):
            t0 = 0
            return np.heaviside((t - t0), 1) * (a * (t - t0) ** 2) + c

        flux = flux - flux[0]
        try:
            parameter, covariance = curve_fit(fit_func, time, flux, sigma=fluxerr, p0=[max(flux), min(flux)])
        except:
            import pdb
            pdb.set_trace()

        return fit_func, parameter

    def get_early_rise_rate(self, recompute=False):
        """
        Compute the early rise rate (slope) of the light curve.
        """

        earlyriserate = getattr(self, 'earlyriserate', None)
        if earlyriserate is not None:
            if not recompute:
                return earlyriserate

        earlyriserate = {}
        outlc = self.get_lc(recompute=recompute)

        col = ['b', 'g', 'r', 'm', 'k', 'y']
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1:
                earlyriserate[pb] = 0
                continue

            fit_func, parameter = self._fit_early_lightcurve(ttime, tFlux, tFluxErr)
            fit_flux = fit_func(ttime, *parameter)

            # print(fit_func, parameter)
            # print(fit_func(5, *parameter))

            # plt.errorbar(ttime, tFlux-tFlux[0], yerr=tFluxErr, fmt='.', color=col[i], label=pb)
            # plt.plot(ttime, fit_flux, color=col[i], label=pb)

            t0 = 1
            t1 = 10
            f0 = fit_func(t0, *parameter)
            f1 = fit_func(t1, *parameter)
            tearlyriserate = (f1/f0) / ((t1 - t0)/(1 + self.z))

            earlyriserate[pb] = tearlyriserate
        # plt.title(self.objectId)
        # plt.legend()
        # plt.show()

        self.earlyriserate = earlyriserate
        return earlyriserate

    def get_color_at_n_days(self, n, recompute=True):
        """
        Compute the colors at n days and the linear slope of the color
        """
        color = getattr(self, 'color', None)
        if color is not None:
            if not recompute:
                return color

        color = {}
        color_slope = {}
        outlc = self.get_lc(recompute=recompute)

        tflux_ndays = {}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1:
                tflux_ndays[pb] = 0
                continue

            fit_func, parameter = self._fit_early_lightcurve(ttime, tFlux, tFluxErr)
            tflux_ndays[pb] = fit_func(n, *parameter)

        passbands = ('u', 'g', 'r', 'i', 'z', 'Y')  # order of filters matters as it must be 'u-g' rather than 'g-u'
        for i, pb1 in enumerate(passbands):
            for j, pb2 in enumerate(passbands):
                if i < j:
                    c = pb1 + '-' + pb2
                    if pb1 not in tflux_ndays.keys() or pb2 not in tflux_ndays.keys() or tflux_ndays[pb2] == 0:
                        color[c] = 0.
                        color_slope[c] = 0.
                        continue
                    color[c] = tflux_ndays[pb1] / tflux_ndays[pb2]
                    color_slope[c] = color[c] / n

        self.color = color
        self.color_slope = color_slope
        return color, color_slope
