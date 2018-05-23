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

            if len(ttime) <= 1 or not np.any(ttime < 0):
                continue
            tFluxUnred = tFluxUnred - np.median(tFluxUnred[ttime < 0])
            # mask = ttime > -30
            # tFluxUnred = tFluxUnred[mask]
            # ttime = ttime[mask]
            # tFluxErrUnred = tFluxErrUnred[mask]

            times[pb] = ttime
            fluxes[pb] = tFluxUnred
            fluxerrs[pb] = tFluxErrUnred

        x0 = [-7]
        bounds = [(-20*(1+self.z), 0)]
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

        return fit_func, best

    def get_early_rise_rate(self, recompute=False):
        """
        Compute the early rise rate (slope) of the light curve.
        """

        earlyriserate = getattr(self, 'earlyriserate', None)
        a_fit = getattr(self, 'a_fit', None)
        c_fit = getattr(self, 'c_fit', None)
        if earlyriserate is not None:
            if not recompute:
                return earlyriserate, a_fit, c_fit

        earlyriserate = {}
        a_fit = {}
        c_fit = {}
        return_vals = {}
        outlc = self.get_lc(recompute=recompute)

        fit_func = getattr(self, 'early_fit_func', None)
        parameters = getattr(self, 'early_parameters', None)
        if fit_func is None or parameters is None:
            fit_func, parameters = self._fit_early_lightcurve(outlc)
            self.early_fit_func, self.early_parameters = fit_func, parameters

        col = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'm', 'z': 'k', 'Y': 'y'}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tFluxUnred = tFluxUnred - np.median(tFluxUnred[ttime < 0])
            # mask = ttime > -30
            # tFluxUnred = tFluxUnred[mask]
            # ttime = ttime[mask]
            # tFluxErrUnred = tFluxErrUnred[mask]

            if len(ttime) <= 1 or not np.any(ttime < 0):
                return_vals[pb] = (-99, -99, -99)
                plt.errorbar(ttime, tFluxUnred, yerr=tFluxErrUnred, fmt='.', color=col[pb], label=pb)
                continue

            fit_flux = fit_func(np.arange(min(ttime), max(ttime), 0.2), *parameters[pb])

            plt.errorbar(ttime, tFluxUnred, yerr=tFluxErrUnred, fmt='.', color=col[pb], label=pb)
            plt.plot(np.arange(min(ttime), max(ttime), 0.2), fit_flux, color=col[pb])

            a, c, t0 = parameters[pb]

            t1 = 1 + t0
            t2 = 10 + t0
            f1 = fit_func(t1, *parameters[pb])
            f2 = fit_func(t2, *parameters[pb])
            tearlyriserate = (f2/f1) / ((t2 - t1)/(1 + self.z))

            earlyriserate[pb] = tearlyriserate
            a_fit[pb] = parameters[pb][0]
            c_fit[pb] = parameters[pb][1]
            return_vals[pb] = (tearlyriserate, parameters[pb][0], parameters[pb][1])

        plt.title(self.objectId)
        plt.xlabel("Days since trigger")
        plt.ylabel("Flux")
        # plt.xlim(-40, 15)
        plt.legend()
        try:
            plt.show()
        except AttributeError:
            import pdb; pdb.set_trace()

        self.earlyriserate = earlyriserate
        self.a_fit = a_fit
        self.c_fit = c_fit
        return return_vals

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

        fit_func = getattr(self, 'early_fit_func', None)
        parameters = getattr(self, 'early_parameters', None)
        if fit_func is None or parameters is None:
            fit_func, parameters = self._fit_early_lightcurve(outlc)
            self.early_fit_func, self.early_parameters = fit_func, parameters

        ignorepb = []
        tflux_ndays = {}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(ttime) <= 1 or not np.any(ttime < 0):
                tflux_ndays[pb] = 0
                continue

            # Check if there is data after t0 trigger for this passband
            a, c, t0 = parameters[pb]
            if max(ttime) < t0:
                print('No data for', pb, ttime)
                ignorepb.append(pb)
            else:
                print("time", pb, ttime)

            n = t0 + n

            tflux_ndays[pb] = fit_func(n, *parameters[pb])

        plt.figure()

        passbands = ('u', 'g', 'r', 'i', 'z', 'Y')  # order of filters matters as it must be 'u-g' rather than 'g-u'
        for i, pb1 in enumerate(passbands):
            for j, pb2 in enumerate(passbands):
                if i < j:
                    c = pb1 + '-' + pb2
                    if pb1 not in tflux_ndays.keys() or pb2 not in tflux_ndays.keys() or tflux_ndays[pb2] == 0 or (pb1 in ignorepb or pb2 in ignorepb):
                        color[c] = -99.
                        color_slope[c] = -99.
                        print("Not plotting", c)
                        continue
                    color[c] = -2.5*np.log10(tflux_ndays[pb1] / tflux_ndays[pb2])
                    color_slope[c] = color[c] / n
                    fit_t = np.arange(-30, 15, 0.2)
                    plt.plot(fit_t, -2.5*np.log10(fit_func(fit_t, *parameters[pb1])/fit_func(fit_t, *parameters[pb2])), label=c)

        plt.title(self.objectId)
        plt.xlabel("Days since trigger")
        plt.ylabel("Color")
        # plt.xlim(-40, 15)
        plt.legend()
        plt.show()
        self.color = color
        self.color_slope = color_slope
        return color, color_slope


def lnlike(params, times, fluxes, fluxerrs):
        # print(params)
        t0 = params[0]
        pars = params[1:]

        chi2 = 0
        for i, pb in enumerate(times):
            a, c = pars[i * 2:i * 2 + 2]

            model = np.heaviside((times[pb] - t0), 1) * (a * (times[pb] - t0) ** 2) + c
            chi2 += sum((fluxes[pb] - model) ** 2 / fluxerrs[pb] ** 2)

            # print(pb, a, c)

        # print('chi2', chi2, params)
        return -chi2


def lnprior(params):
    # print('params', params)
    t0 = params[0]
    pars = params[1:]

    if -35 < t0 < 0:
        return 0.0
    return -np.inf


def lnprob(params, t, flux, fluxerr):
    lp = lnprior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(params, t, flux, fluxerr)


def emcee_fit_all_pb_lightcurves(times, fluxes, fluxerrs, ndim, x0=None, bounds=None):

    nwalkers = 100
    nsteps = 500
    burn = 50
    pos = np.array([x0 + 3*np.random.randn(ndim) for i in range(nwalkers)])
    # print(pos[:,0])

    print("running mcmc...")

    # Ensure intial params within parameter bounds
    params = OrderedDict()
    params['t0'] = {'bounds': bounds[0], 'value': x0[0], 'scale': 1}
    i = 0
    for pb in times.keys():
        for name in ['a', 'c']:
            params[pb + ': ' + name] = {'bounds': bounds[i], 'value': x0[i], 'scale': 3}
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
    # print('new', np.array(pos)[:,0])
    # print(pos)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(times, fluxes, fluxerrs))

    # FInd parameters of lowest chi2
    pos, prob, state = sampler.run_mcmc(pos, 1)
    opos, oprob, orstate = [], [], []
    for pos, prob, rstate in sampler.sample(pos, prob, state, iterations=nsteps):
        opos.append(pos.copy())
        oprob.append(prob.copy())
    pos = np.array(opos)
    prob = np.array(oprob)
    nstep, nwalk = np.unravel_index(prob.argmax(), prob.shape)
    bestpars = pos[nstep, nwalk]
    posterior=prob

    print("best", bestpars)
    # import pylab
    # for j in range(nwalkers):
    #     pylab.plot(posterior[:, j])
    # for i in range(len(params)):
    #     pylab.figure()
    #     pylab.title(list(params.keys())[i])
    #     for j in range(nwalkers):
    #         pylab.plot(pos[:, j, i])

    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    # samples[:, 2] = np.exp(samples[:, 2])
    # bestpars2 = list(map(lambda v: (v[0]), zip(*np.percentile(samples, [50], axis=0))))
    # print('b2', bestpars2)

    print(bestpars)
    t0 = bestpars[0]
    bestpars = bestpars[1:]
    best = {pb: np.append(bestpars[i * 2:i * 2 + 2], t0) for i, pb in enumerate(times)}

    # to_delete = []
    # for i, name in enumerate(params):
    #     if not np.any(samples[:, i]):  # all zeros for parameter then delete column
    #         to_delete.append((i, name))
    # for i, name in to_delete:
    #     samples = np.delete(samples, i, axis=1)
    #     del params[name]

    # from chainconsumer import ChainConsumer
    # c = ChainConsumer()
    # c.add_chain(samples, parameters=list(params.keys()))
    # c.configure(summary=True, cloud=False)
    # c.plotter.plot()
    # c.plotter.plot_walks(convolve=100)

    return best
