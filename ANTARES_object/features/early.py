import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import emcee
from scipy.stats import chisquare
import pylab
from collections import OrderedDict





class EarlyMixin(object):
    """
    Methods to derive early classification features for LAobjects
    """

    def _fit_early_lightcurve(self, outlc):
        """
        Return tsquarize fit to early light curve
        """

        def fit_all_pb_light_curves(params, times, fluxes, fluxerrs):
            print(params)
            t0 = params[0]
            pars = params[1:]

            chi2 = 0
            for i, pb in enumerate(times):
                a, c = pars[i*2:i*2+2]

                model = np.heaviside((times[pb] - t0), 1) * (a * (times[pb] - t0) ** 2) + c
                chi2 += sum((fluxes[pb] - model) ** 2 / fluxerrs[pb] ** 2)

            print(chi2)
            return chi2

        def fit_func(t, a, c, t0):
            return np.heaviside((t - t0), 1) * (a * (t - t0) ** 2) + c

        times = OrderedDict()
        fluxes = OrderedDict()
        fluxerrs = OrderedDict()
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1:
                continue
            tFlux = tFlux - np.median(tFlux[ttime < 0])
            # mask = ttime > -30
            # tFlux = tFlux[mask]
            # ttime = ttime[mask]
            # tFluxErr = tFluxErr[mask]

            times[pb] = ttime
            fluxes[pb] = tFlux
            fluxerrs[pb] = tFluxErr

        x0 = [-7]
        bounds = [(-15, 0)]
        for pb in fluxes:
            x0 += [np.mean(fluxes[pb]), np.median(fluxes[pb])]
            bounds += [(0, max(fluxes[pb])), (min(fluxes[pb]), max(fluxes[pb]))]

        # optimise_result = minimize(fit_all_pb_light_curves, x0=x0, args=(times, fluxes, fluxerrs), bounds=bounds)
        # t0 = optimise_result.x[0]
        # bestpars = optimise_result.x[1:]
        # best = {pb: np.append(bestpars[i*2:i*2+2], t0) for i, pb in enumerate(times)}

        ndim = len(x0)
        best = emcee_fit_all_pb_lightcurves(times, fluxes, fluxerrs, ndim, np.array(x0), bounds)

        # best, covariance = curve_fit(fit_func, time, flux, sigma=fluxerr, p0=[max(flux), min(flux)])

        # best = fit_all_pb_lightcurves(time, flux, fluxerr)
        print('best', best)
        print('times', times, 'fluxes', fluxes)

        return fit_func, best

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

        fit_func, parameters = self._fit_early_lightcurve(outlc)

        col = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'm', 'z': 'k', 'Y': 'y'}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tFlux = tFlux - np.median(tFlux[ttime < 0])
            # mask = ttime > -30
            # tFlux = tFlux[mask]
            # ttime = ttime[mask]
            # tFluxErr = tFluxErr[mask]

            if len(ttime) <= 1:
                earlyriserate[pb] = 0
                plt.errorbar(ttime, tFlux, yerr=tFluxErr, fmt='.', color=col[pb], label=pb)
                continue

            fit_flux = fit_func(np.arange(min(ttime), max(ttime), 0.2), *parameters[pb])

            # print(fit_func, parameter)
            # print(fit_func(5, *parameter))

            plt.errorbar(ttime, tFlux, yerr=tFluxErr, fmt='.', color=col[pb], label=pb)
            plt.plot(np.arange(min(ttime), max(ttime), 0.2), fit_flux, color=col[pb])

            t0 = 1
            t1 = 10
            f0 = fit_func(t0, *parameters[pb])
            f1 = fit_func(t1, *parameters[pb])
            tearlyriserate = (f1/f0) / ((t1 - t0)/(1 + self.z))

            earlyriserate[pb] = tearlyriserate
        plt.title(self.objectId)
        plt.xlim(-40, 15)
        plt.legend()
        plt.show()

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

        fit_func, parameters = self._fit_early_lightcurve(outlc)

        tflux_ndays = {}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1:
                tflux_ndays[pb] = 0
                continue

            tFlux = tFlux - np.median(tFlux[ttime < 0])

            tflux_ndays[pb] = fit_func(n, *parameters[pb])

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
        #             plt.plot(ttime, color[c], label=pb)
        #
        # plt.show()
        self.color = color
        self.color_slope = color_slope
        return color, color_slope



def lnlike(params, times, fluxes, fluxerrs):
        print(params)
        t0 = params[0]
        pars = params[1:]

        chi2 = 0
        for i, pb in enumerate(times):
            a, c = pars[i * 2:i * 2 + 2]
            print('a', pb, a, c)

            model = np.heaviside((times[pb] - t0), 1) * (a * (times[pb] - t0) ** 2) + c
            if pb == 'r':
                print('model', model, fluxes[pb], a, c, t0)
            chi2 += sum((fluxes[pb] - model) ** 2 / fluxerrs[pb] ** 2)

        print('chi2', chi2)
        return -chi2


def lnprior(params):
    # print('params', params)
    t0 = params[0]
    pars = params[1:]

    if -15 < t0 < 0:
        return 0.0
    return -np.inf


def lnprob(params, t, flux, fluxerr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, t, flux, fluxerr)


def emcee_fit_all_pb_lightcurves(times, fluxes, fluxerrs, ndim, x0=None, bounds=None):

    nwalkers = 50
    pos = np.array([x0 + 3*np.random.randn(ndim) for i in range(nwalkers)])
    print(pos[:,0])


    # Ensure intial params within parameter bounds
    params = OrderedDict()
    params['t0'] = {'bounds': bounds[0], 'value': x0[0], 'scale': 1}
    i = 0
    for pb in times.keys():
        for name in ['a', 'c']:
            params[name + '_' + pb] = {'bounds': bounds[i], 'value': x0[i], 'scale': 3}
            i += 1

    for i, name in enumerate(params.keys()):
        # import pdb; pdb.set_trace()
        lb, ub = params[name]['bounds']
        p0     = params[name]['value']
        std    = params[name]['scale']
        # take a 5 sigma range
        lr, ur = (p0-5.*std, p0+5.*std)
        ll = max(lb, lr, 0.)
        ul = min(ub, ur)
        ind = np.where((pos[:,i] <= ll) | (pos[:,i] >= ul))
        nreplace = len(pos[:,i][ind])
        pos[:,i][ind] = np.random.rand(nreplace)*(ul - ll) + ll
    print('new', np.array(pos)[:,0])
    print(pos)


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(times, fluxes, fluxerrs))
    sampler.run_mcmc(pos, 300)
    samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    # for i in range(ndim):
    #     pylab.figure()
    #     pylab.plot(samples[:, i])
    # plt.show()

    samples[:, 2] = np.exp(samples[:, 2])
    bestpars = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=0))))
    print(bestpars)
    t0 = bestpars[0]
    bestpars = bestpars[1:]
    best = {pb: np.append(bestpars[i * 2:i * 2 + 2], t0) for i, pb in enumerate(times)}

    import corner
    fig = corner.corner(samples, labels=list(params.keys()))

    return best
