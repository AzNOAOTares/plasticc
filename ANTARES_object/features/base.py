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
from .. import constants 
from . import stats_computation


class BaseMixin(object):
    """
    Methods to derive baseline features for LAobjects
    """

    def get_lc(self, recompute=False):
        """
        Most often, we want a lightcurve broken up passband by passband to
        compute features. This stores the input light curve in that format.
        Return a lightcurve suitable for computing features
        """
        outlc = getattr(self, '_outlc', None)
        if outlc is None or recompute:        
            filters = self.filters
            outlc = {}

            # store the indices of each filter 
            for i, pb in enumerate(self.filters):
                mask = np.where(self.passband == pb)[0]
                m2 = self.time[mask].argsort()
                ind = mask[m2]
                if len(ind) > 0:
                    outlc[pb] = ind
            self._outlc = outlc
       
        # for each of the default_cols, get the elements for t passband
        out = {}
        for pb, ind in outlc.items():
            out[pb] = [getattr(self, column)[ind] for column in self._default_cols]

        return out

    def get_amplitude(self, recompute=False):
        """
        Return the amplitude
        """
        outamp = getattr(self, 'amplitude', None)
        if outamp is not None:
            if not recompute:
                return outamp

        outamp = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
                
            newflux = tFluxRenorm
            if len(newflux) <= 1:  # if t Flux is empty or there's only measure
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

    def get_filtered_amplitude(self, recompute=False):
        """
        Return the amplitude
        """
        outamp = getattr(self, 'filtered_amplitude', None)
        if outamp is not None:
            if not recompute:
                return outamp

        outamp = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
                
            photmask = tphotflag >= constants.GOOD_PHOTFLAG
            newflux = tFluxRenorm[photmask]

            if len(newflux) <= 1:  # if t Flux is empty or there's only measure
                amp = 0.
            else:
                f_99 = np.percentile(newflux, 99)
                f_01 = np.percentile(newflux,  1)
                f_50 = np.percentile(newflux, 50)
                amp = np.abs((f_99 - f_01)/(f_50 - f_01))
                if not np.isfinite(amp):
                    amp = 0. 
            outamp[pb] = amp

        self.setattr_from_dict_default('filtered_amplitude', outamp, 0.)
        self.filtered_amplitude = outamp
        return outamp


    def get_stats(self, recompute=False):
        """
        Basic statistics for LAobject
        # min, max, mean, std, kurtosis, skewness
        """

        outstats = getattr(self, 'stats', None)
        if outstats is not None:
            if not recompute:
                return outstats

        outstats = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            if len(tFluxRenorm) == 0:
                tstat = scipy.stats.describe([0., 0., 0.])
            else:
                tstat = scipy.stats.describe(tFluxRenorm)

            outstats[pb] = tstat

        self.stats = outstats
        return outstats


    def get_filtered_stats(self, recompute=False):
        """
        Basic statistics for LAobject from filtered observations
        # min, max, mean, std, kurtosis, skewness
        """

        outstats = getattr(self, 'filtered_stats', None)
        if outstats is not None:
            if not recompute:
                return outstats

        outstats = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            mask = tphotflag >= constants.GOOD_PHOTFLAG
            tFluxRenorm = tFluxRenorm[mask]

            if len(tFluxRenorm) == 0:
                tstat = scipy.stats.describe([0., 0., 0.])
            else:
                tstat = scipy.stats.describe(tFluxRenorm)

            outstats[pb] = tstat

        self.filtered_stats = outstats
        return outstats


    def get_skew(self, recompute=False):
        """
        Different definition of skewness
        """
        outskew = getattr(self, 'skew', None)
        if outskew is not None:
            if not recompute:
                return outskew

        outskew = {}
        outlc = self.get_lc(recompute=recompute)
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            npb = len(tFluxRenorm)
            tmean = np.mean(tFluxRenorm)
            tstd  = np.std(tFluxRenorm)

            tskew = (1. / npb) * math.fsum(((tFluxRenorm - tmean) ** 3.) / (tstd ** 3.))
            outskew[pb] = tskew

        self.skew = outskew
        return outskew


    def get_StdOverMean(self, recompute=False):
        """
        Return the Standard Deviation over the Mean
        """
        outstats = self.get_stats(recompute=recompute)
        outSOMean = {pb: (x[3] ** 0.5 / x[2]) for pb, x in outstats.items()}
        return outSOMean


    def get_ShapiroWilk(self, recompute=False):
        """
        Get the Shapriro-Wilk W statistic and the p-value for the test 
        """

        sw = getattr(self, 'ShapiroWilk', None)
        if sw is not None:
            if not recompute:
                return sw
        sw = {}
        outlc = self.get_lc(recompute=recompute)
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            if len(tFlux) <= 3:
                continue
            tsw, tswp = scipy.stats.shapiro(tFluxRenorm)
            sw[pb] = (tsw, tswp)
        self.ShapiroWilk = sw
        return sw


    def get_Q31(self, recompute=False):
        """
        Get the Q31 of the lightcurve
        """
        q31 = getattr(self, 'Q31', None)
        if q31 is not None:
            if not recompute:
                return q31

        q31 = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            tq31 = np.percentile(tFluxRenorm, 75) - np.percentile(tFluxRenorm, 25)
            q31[pb] = tq31
        self.Q31 = q31
        return q31


    def get_RMS(self, recompute=False):
        """
        Get the RMS of the lightcurve
        """
        rms = getattr(self, 'RMS', None)
        if rms is not None:
            if not recompute:
                return rms

        rms = {}
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            photmask = tphotflag >= constants.GOOD_PHOTFLAG
            tFluxRenorm    = tFluxRenorm[photmask]
            tFluxErrRenorm = tFluxErrRenorm[photmask] 

            if len(tFluxRenorm) <= 1:  # if t Flux is empty or has only one element
                rms[pb] = 0.
                continue
    
            tmean = np.mean(tFluxRenorm)
            trms = math.fsum(((tFluxRenorm - tmean) / tFluxErrRenorm) ** 2.)
            trms /= math.fsum(1./ tFluxErrRenorm** 2.)
            trms = trms ** 0.5
            rms[pb] = trms
        self.RMS = rms
        return rms


    def get_ShannonEntropy(self, recompute=False):
        """
        Compute the Shannon Entropy of the lightcurve distribution
        """

        entropy = getattr(self, 'entropy', None)
        if entropy is not None:
            if not recompute:
                return entropy
        entropy = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            tEntropy = stats_computation.shannon_entropy(tFluxRenorm, tFluxErrRenorm)
            entropy[pb] = tEntropy
        self.entropy = entropy
        return entropy


    def get_MAD(self, recompute=False):
        """
        Compute the median absolute deviation of the lightcurve
        """

        mad = getattr(self, 'MAD', None)
        if mad is not None:
            if not recompute:
                return mad

        mad = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            photmask = tphotflag >= constants.GOOD_PHOTFLAG
            tFluxRenorm = tFluxRenorm[photmask]
            if len(tFluxRenorm) == 0:  # if t Flux is empty
                tmad = 0.
            else:
                tmad = median_absolute_deviation(tFluxRenorm)
            mad[pb] = tmad
        self.MAD = mad
        return mad


    def get_vonNeumannRatio(self, recompute=False):
        """
        Compute the Von-Neumann Ratio of the lightcurve
        This is sometimes just called Eta in the context of variables

        The von Neumann ratio η was defined in 1941 by John von Neumann and serves as
        the mean square successive difference divided by the sample variance. When t
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
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            delta = math.fsum(((tFluxRenorm[1:] - tFluxRenorm[:-1]) ** 2.) / (len(tFluxRenorm) - 1))
            tstats = outstats.get(pb)
            if tstats is None:
                continue
            tvar = tstats[3]
            tvnr = delta / (tvar)
            vnr[pb] = tvnr
        self.VNR = vnr
        return vnr


    def get_StetsonJ(self, recompute=False):
        """
        Compute the Stetson J statistic of the lightcurve
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.257.466&rep=rep1&type=pdf
        """

        stetsonJ = getattr(self, 'stetsonJ', None)
        if stetsonJ is not None:
            if not recompute:
                return stetsonJ

        stetsonJ = {}
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tstats = outstats.get(pb)
            if tstats is None:
                continue

            photmask = tphotflag >= constants.GOOD_PHOTFLAG
            tFluxRenorm    = tFluxRenorm[photmask]
            tFluxErrRenorm = tFluxErrRenorm[photmask]

            tmean = tstats[2]
            npb = len(tFluxRenorm)

            if npb < 2:
                stetsonJ[pb] = 0.
                continue

            delta = (npb / (npb - 1)) * ((tFluxRenorm - tmean) / tFluxErrRenorm)
            val = np.nan_to_num(delta[0:-1] * delta[1:])
            sign = np.sign(val)
            tJ = math.fsum(sign * (np.abs(val) ** 0.5))
            stetsonJ[pb] = tJ
        self.stetsonJ = stetsonJ
        return stetsonJ


    def get_StetsonK(self, recompute=False):
        """
        Compute the Stetson K statistic of the lightcurve
        """

        stetsonK = getattr(self, 'stetsonK', None)
        if stetsonK is not None:
            if not recompute:
                return stetsonK

        stetsonK = {}
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tstats = outstats.get(pb)
            if tstats is None:
                stetsonK[pb] = 0.
                continue

            tmean = tstats[2]
            npb = len(tFluxRenorm)

            if npb < 2:
                stetsonK[pb] = 0.
                continue

            delta = (npb / (npb - 1)) * ((tFluxRenorm - tmean) / tFluxErrRenorm)
            tK = (np.sum(np.fabs(delta))/npb) / np.sqrt(np.sum(delta*delta) /npb)
            tK = np.nan_to_num(tK)
            stetsonK[pb] = tK
        self.stetsonK = stetsonK
        return stetsonK


    def get_StetsonL(self, recompute=False):
        """
        Get the Stetson L variability index
        """
        stetsonL = getattr(self, 'stetsonL', None)
        if stetsonL is not None:
            if not recompute:
                return stetsonL

        stetsonJ = self. get_StetsonJ(recompute=recompute)
        stetsonK = self. get_StetsonK(recompute=recompute)

        stetsonL = {}
        for pb in stetsonJ:
            tJ = stetsonJ.get(pb, 0.)
            tK = stetsonK.get(pb, 0.)
            tL = tJ*tK/0.798
            tL = np.nan_to_num(tL)
            stetsonL[pb] = tL
        self.stetsonL = stetsonL 
        return stetsonL 


    def get_AcorrIntegral(self, recompute=False):
        """
        Compute the Autocorrelation get_AcorrIntegral
        """
        AcorrInt = getattr(self, 'AcorrInt', None)
        if AcorrInt is not None:
            if not recompute:
                return AcorrInt

        AcorrInt = {}
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)
        outrms = self.get_RMS(recompute=recompute)

        for j, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tstats = outstats.get(pb)
            if tstats is None:
                continue
            tmean = tstats[2]

            trms = outrms[pb]
            if trms is None:
                continue
            if trms == 0.:
                AcorrInt[pb] = 0.
                continue

            npb = len(tFluxRenorm)
            t = np.arange(1, npb)
            sum_list = []
            val_list = []
            for i in t:
                sum_list.append(math.fsum((tFluxRenorm[0:npb - i] - tmean) * (tFluxRenorm[i:npb] - tmean)))
                val_list.append(1. / ((npb - i) * trms ** 2.))
            tAcorr = np.abs(math.fsum([x * y for x, y in zip(sum_list, val_list)]))
            AcorrInt[pb] = tAcorr
        self.AcorrInt = AcorrInt
        return AcorrInt


    def get_hlratio(self, recompute=False):
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
        outlc = self.get_lc(recompute=recompute)
        outstats = self.get_stats(recompute=recompute)

        for j, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            tWeight = 1. / tFluxErrRenorm 

            tstats = outstats.get(pb)
            if tstats is None:
                continue
            tmean = tstats[2]
            il = tFluxRenorm > tmean
            wl = tWeight[il]
            wlsum = np.sum(wl)
            fl = tFluxRenorm[il]
            wl_weighted_std = np.sum(wl * (fl - tmean) ** 2) / wlsum

            ih = tFluxRenorm <= tmean
            wh = tWeight[ih]
            whsum = np.sum(wh)
            fh = tFluxRenorm[ih]
            wh_weighted_std = np.sum(wh * (fh - tmean) ** 2) / whsum

            hlratio[pb] = np.nan_to_num(np.sqrt(wl_weighted_std / wh_weighted_std))
        self.hlratio = hlratio
        return hlratio


    def get_color_amplitudes(self, recompute=False):
        """
        Get the amplitude difference between passbands
        """
        colorAmp = getattr(self, 'colorAmp', None)
        if colorAmp is not None:
            if not recompute:
                return colorAmp

        colorAmp = {}
        amp = self.get_filtered_amplitude(recompute=recompute)

        passbands = ('u', 'g', 'r', 'i', 'z', 'Y') # order of filters matters as it must be 'u-g' rather than 'g-u'
        for i, pb1 in enumerate(passbands):
            for j, pb2 in enumerate(passbands):
                if i < j:
                    color = pb1 + '-' + pb2
                    if pb1 not in amp.keys():
                        amp[pb1] = 0.
                    if pb2 not in amp.keys():
                        amp[pb2] = 0.
                    colorAmp[color] = amp[pb1] - amp[pb2]

        self.colorAmp = colorAmp
        return colorAmp


    def get_color_mean(self, recompute=False):
        colorMean = getattr(self, 'colorMean', None)
        if colorMean is not None:
            if not recompute:
                return colorMean

        colorMean = {}
        outlc = self.get_lc(recompute=recompute)

        tmean = {}
        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc
            tmean[pb] = np.mean(tFluxRenorm)

        passbands = ('u', 'g', 'r', 'i', 'z', 'Y') # order of filters matters as it must be 'u-g' rather than 'g-u'
        for i, pb1 in enumerate(passbands):
            for j, pb2 in enumerate(passbands):
                if i < j:
                    color = pb1 + '-' + pb2
                    if pb1 not in tmean.keys():
                        tmean[pb1] = 0.
                    if pb2 not in tmean.keys():
                        tmean[pb2] = 0.
                    colorMean[color] = tmean[pb1] / tmean[pb2]

        self.colorMean = colorMean
        return colorMean


