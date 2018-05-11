# -*- coding: UTF-8 -*-
"""
ANTARES Object class specification
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
import numpy as np
from . import constants
from .features.periodic import PeriodicMixin
from .features.gp import GPMixin
from .features.spline import SplineMixin
from .features.base import BaseMixin
from .features.early import EarlyMixin
from .features.plasticc import PlasticcMixin
from astropy.stats import sigma_clip
import extinction

__all__ = ['LAobject']

class LAobject(PlasticcMixin, PeriodicMixin, GPMixin, SplineMixin, BaseMixin, EarlyMixin):
    """
    ANTARES object - locus aggregated alert lightcurve and feature encapsulator

    :py:func:`antares.model.helper.MakeTouchstoneObject` creates a
    :py:class:`ANTARES_object` instance from an event alert.

    TODO: rename ``MakeTouchstoneObject`` something more useful

    Parameters
    ----------
    locusId : str or int
        the ANTARES locus aggegated alert ID
    objectId : str or int
        The unique identifier for each Object - it's name from the source survey e.g. diaObjectId
    time : array-like
        The mid-point of the observation - e.g. midPointTai from DIASource
    flux : array-like
        The calibrated flux - e.g. totFlux of each DIASource
    fluxErr : array-like
        The calibrated flux uncertainty - e.g. totFluxErr of each DIASource
    obsId : array-like
        The list of IDS  corresponding to the observations - e.g. diaSourceId
    photflag : array-like
        Photometry flag corresponding to each DIASource 
        The flag above which the photometry is asserted to be good is defined
        in GOOD_PHOTFLAG is defined in `constants.py`
    passband : array-like
        The passband of each observation
    zeropoint : float or array-like
        The zeropoint of each observation - used to convert to Pogson magnitudes if possible
        The default zeropoint DEFAULT_ZEROPOINT is defined in `constants.py`
    per : bool
        Boolean flag indicating if this object is considered a periodic variable or not
    best_period : float or None
        If `per`, then the best-estimate of the period of this object
    header : None or dict
        A header entry as a dictionary of values that should be preserved during processing
    mag : bool
        Is this input light curve provided in magnitudes? Converts to flux
        using zeropoint if True
    preprocess : bool
        Clean up invalid values from the light curve
    clean : bool
        Attempt to remove outliers from the light curve. Implies preprocess. Use with caution.
    ebv : float
        Milky-Way extinction on line of sight to this object 
        Assumed to be zero if not provided 
        Applied to construct the unreddened flux - fluxUnred
    renorm : bool
        NOT IMPLMENTED, DEPRECATED 
    mu : float 
        distance modulus of this object - if renorm is True, the fluxes are
        corrected from this distance modulus to 10pc - note that there is no
        k-correction being applied.
        NOT IMPLMENTED, DEPRECATED 

    Notes
    -----
        Follows the DPDD as of 20170810:
        https://docushare.lsstcorp.org/docushare/dsweb/Get/LSE-163/

        ``time``, ``flux``, ``fluxErr``, ``obsID``, ``passband`` must be 1-D
        arrays with the same shape, and have at least one element
    """

    def __init__(self, locusId, objectId, time, flux, fluxErr, obsId, photflag, passband, zeropoint=constants.DEFAULT_ZEROPOINT, 
                 per=False, best_period=None, header=None, mag=False, preprocess=True, 
                 clean=False, renorm=False, ebv=0., mu=None, **kwargs):

        # object name or ID - an alert ID generated by ANTARES or pre-supplied
        # note that if ANTARES is generating its own IDs, then we need someway
        # to store a DIAObject ID

        # scalars
        self.objectId = objectId
        self.locusId  = locusId
        self.ebv      = float(ebv)

        # vectors
        self.time     = np.array(time).astype('f')
        zeropoint     = np.atleast_1d(zeropoint)
        # if there's a single number for zeropoint, make sure that it is an array
        if len(zeropoint) == 1:
            zeropoint = np.repeat(zeropoint, _tshape[0])
        self.zeropoint = zeropoint

        # convert input magnitudes to fluxes 
        if mag:
            mag        = np.array(flux).astype('f')
            magErr     = np.array(fluxErr).astype('f')
            self.flux  = 10.**(-0.4*(mag - zeropoint))
            self.fluxErr = np.abs(self.flux*magErr*(np.log(10.)/2.5))
        else:
            self.flux     = np.array(flux).astype('f')
            self.fluxErr  = np.array(fluxErr).astype('f')
        self.obsId    = np.array(obsId)
        self.passband = passband
        self.photflag = photflag

        # names of extra vectors
        self._extra_cols = []

        # check that the arrays have the same shape
        assert self.time.ndim == 1
        _tshape = self.time.shape
        assert _tshape == self.flux.shape
        assert _tshape == self.fluxErr.shape
        assert _tshape == self.obsId.shape
        assert _tshape == self.passband.shape
        assert _tshape == self.photflag.shape
        assert _tshape == self.zeropoint.shape

        for key, val in kwargs.items():
            if np.isscalar(val):
                setattr(self, key, val)
            else:
                valarr = np.array(val)
                setattr(self, key, valarr)
                if _tshape == valarr.shape:
                    self._extra_cols.append(key)

        # keep track of what passbands we have
        self.filters = set(self.passband)

        # save the other inputs
        self.per = per
        if self.per:
            self.best_period = best_period
            self.lcPeriodic = np.nan
            self.lcNonPeriodic = None
        else:
            self.best_period = None
            self.lcPeriodic = None
            self.lcNonPeriodic = np.nan

        if header is None:
            header = {}
        self.header = header

        # pre-processing = remove invalid values
        # cleaning = remove outliers
        # cleaning the lightcurve implies pre-processing
        if clean and not preprocess:
            preprocess = True

        if preprocess:
            # exclude bad values
            mask = np.logical_and((self.fluxErr > 1E-8), np.isfinite(self.fluxErr))
            mask = np.logical_and(mask, np.isfinite(self.flux))
            mask = np.logical_and(mask, np.isfinite(self.zeropoint))
            # mask = np.logical_and(mask, self.photflag == constants.BAD_PHOTFLAG)

            # these cuts are only applied if the light curve is provided in magnitudes
            # they handle annoying dummy values and very low S/N points
            if mag:
                mask = np.logical_and(mask, self.flux > 0.)
                mask = np.logical_and(mask, self.flux <= 99.)
                mask = np.logical_and(mask, self.fluxErr <= 9.)

            # if any of the filtered points are flagged as good detections with
            # PHOTFLAG > 0, save them irrespective
            saveind = (self.photflag[mask] >= constants.GOOD_PHOTFLAG)
            mask[saveind] = True

            # apply the mask
            self.time      = self.time[mask]
            self.flux      = self.flux[mask]
            self.fluxErr   = self.fluxErr[mask]
            self.obsId     = self.obsId[mask]
            self.photflag  = self.photflag[mask]
            self.passband  = self.passband[mask]
            self.zeropoint = self.zeropoint[mask]

            for key in self._extra_cols:
                val = getattr(self, key)
                setattr(self, key, val[mask])

            if not clean:
                return self.finalize()

            # we should still update filters in case all the data in a filter was invalid
            self.filters = set(self.passband)

            # begin cleaning the lightcurve - hic sunt dracones
            # remove points with S/N < 1
            mask = self.fluxErr < np.abs(self.flux)

            # apply the mask
            self.time      = self.time[mask]
            self.flux      = self.flux[mask]
            self.fluxErr   = self.fluxErr[mask]
            self.obsId     = self.obsId[mask]
            self.photflag  = self.photflag[mask]
            self.passband  = self.passband[mask]
            self.zeropoint = self.zeropoint[mask]
            for key in self._extra_cols:
                val = getattr(self, key)
                setattr(self, key, val[mask])

            t   = None
            f   = None
            df  = None
            zpt = None
            pbs = None
            oid = None
            pf  = None
            filtered_extra_cols = {}

            # do some sigmaclipping to reject outliers
            for pb in self.filters:
                indf = (self.passband == pb)
                filtered_err = sigma_clip(self.fluxErr[indf], sigma=3., iters=5, copy=True)
                bad1 = filtered_err.mask

                saveind = np.where(self.photflag[indf][bad1] >= constants.GOOD_PHOTFLAG)
                bad1[saveind] = False

                useind = ~bad1

                if t is None:
                    t = self.time[indf][useind]
                else:
                    t = np.concatenate((t, self.time[indf][useind]))

                if f is None:
                    f = self.flux[indf][useind]
                else:
                    f = np.concatenate((f, self.flux[indf][useind]))

                if df is None:
                    df = self.fluxErr[indf][useind]
                else:
                    df = np.concatenate((df, self.fluxErr[indf][useind]))

                if zpt is None:
                    zpt = self.zeropoint[indf][useind]
                else:
                    zpt = np.concatenate((zpt, self.zeropoint[indf][useind]))

                if pbs is None:
                    pbs = self.passband[indf][useind]
                else:
                    pbs = np.concatenate((pbs, self.passband[indf][useind]))

                if oid is None:
                    oid = self.obsId[indf][useind]
                else:
                    oid = np.concatenate((oid, self.obsId[indf][useind]))

                if pf is None:
                    pf = self.photflag[indf][useind]
                else:
                    pf = np.concatenate((pf, self.photflag[indf][useind]))


                for key in self._extra_cols:
                    orig = getattr(self, key)
                    val = filtered_extra_cols.get(key, None)
                    if val is None:
                        val = orig[indf][useind]
                        filtered_extra_cols[key] = val
                    else:
                        val = np.concatenate((val, orig[indf][useind]))
                        filtered_extra_cols[key] = val

            self.time      = t
            self.flux      = f
            self.fluxErr   = df
            self.passband  = pbs
            self.zeropoint = zpt
            self.obsId     = oid
            self.photflag  = photflag

            for key in self._extra_cols:
                val = filtered_extra_cols.get(key)
                setattr(self, key, val)

            # finalize the light curve after cleaning    
            return self.finalize()


    def _remove_flux_extinction(self):
        """
        Remove extinction for light curve assuming Fitzpatrick '99 reddening
        law, given some value of E(B-V)
        """
        self.fluxUnred     = self.flux.copy()
        self.fluxErrUnred  = self.fluxErr.copy()
        self.fluxRenorm    = self.flux.copy()
        self.fluxErrRenorm = self.fluxErr.copy()

        # Using negative a_v so that extinction.apply works in reverse and removes the extinction
        extinctions = extinction.fitzpatrick99(wave=self._good_filter_wave,\
                a_v=-3.1 * self.ebv, r_v=3.1, unit='aa')

        for i, pb in enumerate(self._good_filters):
            mask = (self.passband == pb)

            flux_pb    = self.flux[mask]
            fluxerr_pb = self.fluxErr[mask]
            npbobs     = len(flux_pb)

            if npbobs > 1:
                # there's at least enough observations to find minimum and maximum
                flux_out = extinction.apply(extinctions[i], flux_pb, inplace=False)
                fluxerr_out = extinction.apply(extinctions[i], fluxerr_pb, inplace=False)
                self.fluxUnred[mask] = flux_out
                self.fluxErrUnred[mask] = fluxerr_out
                
                minfluxpb = flux_out.min()
                maxfluxpb = flux_out.max()
                norm = maxfluxpb - minfluxpb 

                self.fluxRenorm[mask] = flux_out - minfluxpb 
                self.fluxErrRenorm[mask] = fluxerr_out

                self.fluxRenorm[mask] /= norm 
                self.fluxErrRenorm[mask] /= norm
            elif npbobs == 1:
                # deal with the case with one observation in this passband by setting renorm = 0.5 
                flux_out = extinction.apply(extinctions[i], flux_pb, inplace=False)
                fluxerr_out = extinction.apply(extinctions[i], fluxerr_pb, inplace=False)
                self.fluxUnred[mask] = flux_out
                self.fluxErrUnred[mask] = fluxerr_out

                norm = self.fluxUnred[mask]/0.5
                self.fluxRenorm[mask] /= norm 
                self.fluxErrRenorm[mask] /= norm
            else:
                pass

        self._default_cols = ['time', 'flux', 'fluxErr', 'fluxUnred', 'fluxErrUnred',\
                                'fluxRenorm', 'fluxErrRenorm', 'photflag', 'zeropoint', 'obsId']
        return


    def finalize(self):
        """
        Finalizes the LAobject by setting the list of available
        filters and the light curve length after filtering.

        HACK : the filters are limited to the PLAsTiCC ugrizY set since the
        PropertyTable SQL in ANTARES must have pre-defined keys
        """
        # this forces only some filters will be used for feature computation
        # this is not ideal, but a necessary stop-gap while we revise
        # the PropertyTable SQL
        self._good_filters = ['u', 'g', 'r', 'i', 'z', 'Y']
        self._good_filter_wave = np.array([3569.5, 4766.5, 6214.5, 7544.5, 8707.5, 10039.5])

        use_filters = set(self._good_filters) & self.filters
        if not self.filters.issubset(use_filters):
            message = 'Number of useful filters ({}) does not equal number available filters ({}) - some filters will not be used'.format(
                ''.join(use_filters), ''.join(self.filters))
            warnings.warn(message, RuntimeWarning)
        self.filters = set(use_filters)
        mask = np.array([True if x in self.filters else False for x in self.passband])

        if mask.size:  # Not empty arrays
            self.time      = self.time[mask]
            self.flux      = self.flux[mask]
            self.fluxErr   = self.fluxErr[mask]
            self.obsId     = self.obsId[mask]
            self.passband  = self.passband[mask]
            self.zeropoint = self.zeropoint[mask]
            for key in self._extra_cols:
                val = getattr(self, key)
                setattr(self, key, val[mask])

        self.nobs = len(self.time)
        if self.nobs == 0:
            message = 'Object {} with locus ID {} has no good observations.'.format(self.objectId, self.locusId)
            raise ValueError(message)

        return self._remove_flux_extinction()


    def setattr_from_dict_default(self, rootname, values_dict, default_value):
        """
        Set attributes for the LAobject from a values dictionary indexed by passband
        The set attribute names are rootname_passband

        Attributes are only set for passbands listed in _good_filters i.e. if
        the light curve has passbands that were non-standard, then no features
        are set.

        If a passband listed in _good_filters is not present in the
        values_dict, the default_value is set
        """
        for pb in self._good_filters:
            setattr(self, '{}_{}'.format(rootname, pb), values_dict.get(pb, default_value))
