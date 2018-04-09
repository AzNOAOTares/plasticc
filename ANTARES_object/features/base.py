# -*- coding: UTF-8 -*-
"""
Functions for the derivation of base features with LAobjects
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
import math
from six.moves import zip
import scipy.stats
import scipy.interpolate as scinterp
from astropy.stats import median_absolute_deviation
from . import stats_computation


class BaseMixin(object):
    """
    Methods to derive baseline features for LAobjects
    """

    def get_lc(self, phase_offset=None, per=False, recompute=False, usephotflag=False):
        """
        Return a lightcurve suitable for computing features
        """
        outlc = getattr(self, '_outlc', None)
        if outlc is not None:
            if not recompute:
                return outlc

        phase = self.get_phase(phase_offset=phase_offset, per=per)
        filters = self.filters
        outlc = {}

        # Return the observations in the filter
        for i, pb in enumerate(self.filters):
            mask = (self.passband == pb)
            m2 = phase[mask].argsort()

            thispbphase       = phase[mask][m2]
            thisFlux          = self.flux[mask][m2]
            thisFluxErr       = self.fluxErr[mask][m2]
            thisFluxUnred     = self.fluxUnred[mask][m2]
            thisFluxErrUnred  = self.fluxErrUnred[mask][m2]
            thisFluxRenorm    = self.fluxRenorm[mask][m2]
            thisFluxErrRenorm = self.fluxErrRenorm[mask][m2]

            if len(thisFlux) > 0:
                # kinda hackish way to get photflag as well
                if usephotflag and self.photflag is not None:
                    photflag = self.photflag[mask][m2]
                else:
                    photflag = None

                outlc[pb] = (thispbphase, thisFlux, thisFluxErr,\
                                            thisFluxUnred, thisFluxErrUnred,\
                                            thisFluxRenorm, thisFluxErrRenorm,\
                                            photflag)
        self._outlc = outlc
        return outlc

    def get_amplitude(self, per=None, phase_offset=None, recompute=False, usephotflag=False):
        """
        Return the amplitude
        """

        outamp = getattr(self, 'amplitude', None)
        if outamp is not None:
            if not recompute:
                return outamp

        outamp = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset, usephotflag=True)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
                
            # this definition should give identical results for fluxUnred or fluxRenorm 
            if usephotflag and photflag is not None:
                # we have a way to filter observations
                photmask = photflag >= 4096
                print(len(thisFluxRenorm[photmask]))
                newflux = thisFluxRenorm[photmask]
            else:
                newflux = thisFluxRenorm 

            if len(newflux) <= 1:  # if this Flux is empty or there's only measure
                amp = 0.
            else:
                f_99 = np.percentile(newflux, 99)
                f_01 = np.percentile(newflux,  1)
                f_50 = np.percentile(newflux, 50)
                amp = np.abs((f_99 - f_01)/(f_50 - f_01))
                if not np.isfinite(amp):
                    amp = 0. 
            outamp[pb] = amp

        self.setattr_from_dict_default('amplitude', outamp, 0.)
        self.amplitude = outamp
        return outamp

    def get_stats(self, per=None, phase_offset=None, recompute=False, usephotflag=False):
        """
        Basic statistics for LAobject
        # min, max, mean, std, kurtosis, skewness
        """

        outstats = getattr(self, 'stats', None)
        if outstats is not None:
            if not recompute:
                return outstats

        outstats = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset, usephotflag=usephotflag)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            if usephotflag and photflag is not None:
                photmask = photflag >= 4096
                thisFluxRenorm = thisFluxRenorm[photmask]
                if len(thisFluxRenorm) == 0:
                    thisstat = scipy.stats.describe([0., 0., 0.])
                else:
                    thisstat = scipy.stats.describe(thisFluxRenorm)
            else:
                thisstat = scipy.stats.describe(thisFluxRenorm)

            outstats[pb] = thisstat

        self.stats = outstats
        return outstats

    def get_skew(self, per=None, phase_offset=None, recompute=False):
        """
        Different definition of skewness
        """
        outskew = getattr(self, 'skew', None)
        if outskew is not None:
            if not recompute:
                return outskew

        outskew = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset)
        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
            npb = len(thisFluxRenorm)

            thisstats = outstats.get(pb)
            if thisstats is None:
                continue

            thismean = np.mean(thisFluxRenorm)
            thisstd  = np.std(thisFluxRenorm)

            thisskew = (1. / npb) * math.fsum(((thisFluxRenorm - thismean) ** 3.) / (thisstd ** 3.))
            outskew[pb] = thisskew

        self.skew = outskew
        return outskew

    def get_StdOverMean(self, per=False, phase_offset=None, recompute=False):
        """
        Return the Standard Deviation over the Mean
        """
        outstats = self.get_stats(recompute=recompute, per=per,  phase_offset=phase_offset, usephotflag=False)
        outSOMean = {pb: (x[3] ** 0.5 / x[2]) for pb, x in outstats.items()}
        return outSOMean

    def get_ShapiroWilk(self, per=False,  phase_offset=None, recompute=False):
        """
        Get the Shapriro-Wilk W statistic
        """

        sw = getattr(self, 'ShapiroWilk', None)
        if sw is not None:
            if not recompute:
                return sw
        sw = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset)
        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
            if len(thisFlux) <= 3:
                continue
            thissw, _ = scipy.stats.shapiro(thisFluxRenorm)
            sw[pb] = thissw
        self.ShapiroWilk = sw
        return sw

    def get_Q31(self, per=False, phase_offset=None, recompute=False):
        """
        Get the Q31 of the lightcurve
        """
        q31 = getattr(self, 'Q31', None)
        if q31 is not None:
            if not recompute:
                return q31

        q31 = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
            thisq31 = np.percentile(thisFluxRenorm, 75) - np.percentile(thisFluxRenorm, 25)
            q31[pb] = thisq31
        self.Q31 = q31
        return q31

    def get_RMS(self, per=False, phase_offset=None, recompute=False):
        """
        Get the RMS of the lightcurve
        """
        rms = getattr(self, 'RMS', None)
        if rms is not None:
            if not recompute:
                return rms

        rms = {}
        outlc = self.get_lc(recompute=recompute, per=per,   phase_offset=phase_offset, usephotflag=True)
        outstats = self.get_stats(recompute=recompute, per=per,   phase_offset=phase_offset, usephotflag=False)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            if photflag is not None:
                photmask = photflag >= 4096
                thisFluxRenorm    = thisFluxRenorm[photmask]
                thisFluxErrRenorm = thisFluxErrRenorm[photmask] 

            if len(thisFlux) == 0:  # if this Flux is empty
                rms[pb] = 0.
                continue
    
            thismean = np.mean(thisFluxRenorm)
            thisrms = math.fsum(((thisFluxRenorm - thismean) / thisFluxErrRenorm) ** 2.)
            thisrms /= math.fsum(1. / thisFluxErrRenorm ** 2.)
            thisrms = thisrms ** 0.5
            rms[pb] = thisrms
        self.RMS = rms
        return rms

    def get_ShannonEntropy(self, per=False, phase_offset=None, recompute=False):
        """
        Compute the Shannon Entropy of the lightcurve distribution
        """

        entropy = getattr(self, 'entropy', None)
        if entropy is not None:
            if not recompute:
                return entropy
        entropy = {}
        outlc = self.get_lc(recompute=recompute, per=per,   phase_offset=phase_offset)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
            thisEntropy = stats_computation.shannon_entropy(thisFluxRenorm, thisFluxErrRenorm)
            entropy[pb] = thisEntropy
        self.entropy = entropy
        return entropy

    def get_MAD(self, per=False, phase_offset=None, recompute=False):
        """
        Compute the median absolute deviation of the lightcurve
        """

        mad = getattr(self, 'MAD', None)
        if mad is not None:
            if not recompute:
                return mad

        mad = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset, usephotflag=True)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            if photflag is not None:
                photmask = photflag >= 4096
                thisFluxRenorm = thisFluxRenorm[photmask]
            if len(thisFluxRenorm) == 0:  # if this Flux is empty
                thismad = 0
            else:
                thismad = median_absolute_deviation(thisFluxRenorm)
            mad[pb] = thismad
        self.MAD = mad
        return mad

    def get_vonNeumannRatio(self, per=False, phase_offset=None, recompute=False):
        """
        Compute the Von-Neumann Ratio of the lightcurve
        This is sometimes just called Eta in the context of variables

        The von Neumann ratio η was defined in 1941 by John von Neumann and serves as
        the mean square successive difference divided by the sample variance. When this
        ratio is small, it is an indication of a strong positive correlation between
        the successive photometric data points.  See: (J. Von Neumann, The Annals of
        Mathematical Statistics 12, 367 (1941))

        This seems like something that'd be much useful to compute from a phase
        curve...
        """

        vnr = getattr(self, 'VNR', None)
        if vnr is not None:
            if not recompute:
                return vnr

        vnr = {}
        outlc = self.get_lc(recompute=recompute, per=per,   phase_offset=phase_offset)
        outstats = self.get_stats(recompute=recompute, per=per,   phase_offset=phase_offset)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc
            delta = math.fsum(((thisFluxRenorm[1:] - thisFluxRenorm[:-1]) ** 2.) / (len(thisFluxRenorm) - 1))
            thisstats = outstats.get(pb)
            if thisstats is None:
                continue
            thisvar = thisstats[3]
            thisvnr = delta / (thisvar)
            vnr[pb] = thisvnr
        self.VNR = vnr
        return vnr

    def get_StetsonJ(self, per=False, phase_offset=None, recompute=False):
        """
        Compute the Stetson J statistic of the lightcurve
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.257.466&rep=rep1&type=pdf
        """

        stetsonJ = getattr(self, 'stetsonJ', None)
        if stetsonJ is not None:
            if not recompute:
                return stetsonJ

        stetsonJ = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset, usephotflag=True)
        outstats = self.get_stats(recompute=recompute, per=per,   phase_offset=phase_offset, usephotflag=False)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            thisstats = outstats.get(pb)
            if thisstats is None:
                continue

            if photflag is not None:
                photmask = photflag >= 4096
                thisFluxRenorm    = thisFluxRenorm[photmask]
                thisFluxErrRenorm = thisFluxErrRenorm[photmask]

            thismean = thisstats[2]
            npb = len(thisFluxRenorm)

            if npb < 2:
                stetsonJ[pb] = 0.
                continue

            delta = (npb / (npb - 1)) * ((thisFluxRenorm - thismean) / thisFluxErrRenorm)
            val = np.nan_to_num(delta[0:-1] * delta[1:])
            sign = np.sign(val)
            thisJ = math.fsum(sign * (np.abs(val) ** 0.5))
            stetsonJ[pb] = thisJ
        self.stetsonJ = stetsonJ
        return stetsonJ

    def get_StetsonK(self, per=False,  phase_offset=None, recompute=False):
        """
        Compute the Stetson K statistic of the lightcurve
        """

        stetsonK = getattr(self, 'stetsonK', None)
        if stetsonK is not None:
            if not recompute:
                return stetsonK

        stetsonK = {}
        outlc = self.get_lc(recompute=recompute, per=per,   phase_offset=phase_offset)
        outstats = self.get_stats(recompute=recompute, per=per,   phase_offset=phase_offset)

        for i, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            thisstats = outstats.get(pb)
            if thisstats is None:
                stetsonK[pb] = 0.
                continue

            thismean = thisstats[2]
            npb = len(thisFluxRenorm)

            if npb < 2:
                stetsonK[pb] = 0.
                continue

            delta = (npb / (npb - 1)) * ((thisFluxRenorm - thismean) / thisFluxErrRenorm)
            thisK = (np.sum(np.fabs(delta))/npb) / np.sqrt(np.sum(delta*delta) /npb)
            thisK = np.nan_to_num(thisK)
            stetsonK[pb] = thisK
        self.stetsonK = stetsonK
        return stetsonK

    def get_StetsonL(self, per=False, phase_offset=None, recompute=False):
        """
        Get the Stetson L variability index
        """
        stetsonL = getattr(self, 'stetsonL', None)
        if stetsonL is not None:
            if not recompute:
                return stetsonL

        stetsonJ = self. get_StetsonJ(per=per, phase_offset=phase_offset, recompute=recompute)
        stetsonK = self. get_StetsonK(per=per, phase_offset=phase_offset, recompute=recompute)

        stetsonL = {}
        for pb in stetsonJ:
            thisJ = stetsonJ.get(pb, 0.)
            thisK = stetsonK.get(pb, 0.)
            thisL = thisJ*thisK/0.798
            thisL = np.nan_to_num(thisL)
            stetsonL[pb] = thisL
        self.stetsonL = stetsonL 
        return stetsonL 

    def get_AcorrIntegral(self, per=False, phase_offset=None, recompute=False, usephotflag=False):
        """
        Compute the Autocorrelation get_AcorrIntegral
        """
        AcorrInt = getattr(self, 'AcorrInt', None)
        if AcorrInt is not None:
            if not recompute:
                return AcorrInt

        AcorrInt = {}
        outlc = self.get_lc(recompute=recompute, per=per,   phase_offset=phase_offset)
        outstats = self.get_stats(recompute=recompute, per=per,   phase_offset=phase_offset)
        outrms = self.get_RMS(recompute=recompute, per=per,   phase_offset=phase_offset)

        for j, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            thisstats = outstats.get(pb)
            if thisstats is None:
                continue
            thismean = thisstats[2]

            thisrms = outrms[pb]
            if thisrms is None:
                continue

            npb = len(thisFluxRenorm)
            t = np.arange(1, npb)
            sum_list = []
            val_list = []
            for i in t:
                sum_list.append(math.fsum((thisFluxRenorm[0:npb - i] - thismean) * (thisFluxRenorm[i:npb] - thismean)))
                val_list.append(1. / ((npb - i) * thisrms ** 2.))
            thisAcorr = np.abs(math.fsum([x * y for x, y in zip(sum_list, val_list)]))
            AcorrInt[pb] = thisAcorr
        self.AcorrInt = AcorrInt
        return AcorrInt

    def get_hlratio(self, per=False, phase_offset=None, recompute=False):
        """
        Compute the ratio of amplitude of observations higher than the average
        to those lower than the average, taking into account observed
        uncertainties. This ratio should be higher for eclipsing binaries than
        pulsating variables.
        """

        hlratio = getattr(self, 'hlratio', None)
        if hlratio is not None:
            if not recompute:
                return hlratio

        hlratio = {}
        outlc = self.get_lc(recompute=recompute, per=per, phase_offset=phase_offset)
        outstats = self.get_stats(recompute=recompute, per=per, phase_offset=phase_offset)

        for j, pb in enumerate(outlc):
            thislc = outlc.get(pb)
            thispbphase, thisFlux, thisFluxErr, thisFluxUnred, thisFluxErrUnred, thisFluxRenorm, thisFluxErrRenorm, photflag = thislc

            thisWeight = 1. / thisFluxErrRenorm 

            thisstats = outstats.get(pb)
            if thisstats is None:
                continue
            thismean = thisstats[2]
            il = thisFluxRenorm > thismean
            wl = thisWeight[il]
            wlsum = np.sum(wl)
            fl = thisFluxRenorm[il]
            wl_weighted_std = np.sum(wl * (fl - thismean) ** 2) / wlsum

            ih = thisFluxRenorm <= thismean
            wh = thisWeight[ih]
            whsum = np.sum(wh)
            fh = thisFluxRenorm[ih]
            wh_weighted_std = np.sum(wh * (fh - thismean) ** 2) / whsum

            hlratio[pb] = np.nan_to_num(np.sqrt(wl_weighted_std / wh_weighted_std))
        self.hlratio = hlratio
        return hlratio
