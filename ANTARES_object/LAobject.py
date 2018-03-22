# -*- coding: UTF-8 -*-
"""
ANTARES Object class specification

"""

from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
import numpy as np
from .features.periodic import PeriodicMixin
from .features.gp import GPMixin
from .features.spline import SplineMixin
from .features.base import BaseMixin
from .features.plasticc import PlasticcMixin
from astropy.stats import sigma_clip
import extinction

__all__ = ['LAobject']


class LAobject(PlasticcMixin, PeriodicMixin, GPMixin, SplineMixin, BaseMixin):
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
    passband : array-like
        The passband of each observation
    zeropoint : float or array-like
        The zeropoint of each observation - used to convert to Pogson magnitudes if possible
    per : bool
        Boolean flag indicating if this object is considered a periodic variable or not
    best_period : float or None
        If `per`, then the best-estimate of the period of this object
    header : None or dict
        A header entry as a dictionary of values that should be preserved during processing
    preprocess : bool
        Clean up invalid values from the light curve
    clean : bool
        Attempt to remove outliers from the light curve. Implies preprocess. Use with caution.

    Notes
    -----
        Follows the DPDD as of 20170810:
        https://docushare.lsstcorp.org/docushare/dsweb/Get/LSE-163/

        ``time``, ``flux``, ``fluxErr``, ``obsID``, ``passband`` must be 1-D
        arrays with the same shape, and have at least one element
    """

    def __init__(self, locusId, objectId, time, flux, fluxErr, obsId, passband, zeropoint, per=False, best_period=None,
                 header=None, mag=True, preprocess=True, clean=False,
                 renorm=False, remove_extinction=False, mwebv=None, mu=None, **kwargs):

        # object name or ID - an alert ID generated by ANTARES or pre-supplied
        # note that if ANTARES is generating its own IDs, then we need someway
        # to store a DIAObject ID

        self.objectId = objectId
        self.locusId = locusId

        self.time = np.array(time).astype('f')
        self.flux = np.array(flux).astype('f')
        self.fluxErr = np.array(fluxErr).astype('f')
        self.obsId = np.array(obsId)
        self.passband = passband
        zeropoint = np.atleast_1d(zeropoint)

        self._extra_cols = []

        # check that the arrays have the same shape
        assert self.time.ndim == 1
        _tshape = self.time.shape
        assert _tshape == self.flux.shape
        assert _tshape == self.fluxErr.shape
        assert _tshape == self.obsId.shape
        assert _tshape == self.passband.shape

        # if there's a single number for zeropoint, make sure that it is an array
        if len(zeropoint) == 1:
            zeropoint = np.repeat(zeropoint, _tshape[0])
        self.zeropoint = zeropoint

        for key, val in kwargs.items():
            if np.isscalar(val):
                setattr(self, key, val)
            else:
                valarr = np.array(val)
                setattr(self, key, valarr)
                if _tshape == valarr.shape:
                    self._extra_cols.append(key)

        # keep track of what passbands we have
        self.filters = list(set(self.passband))

        # save the other inputs
        self.mag = mag
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

        # remove extinction
        if remove_extinction:
            if mwebv is None:
                raise TypeError("Need arg 'mwebv' to remove extinction")
            else:
                self.flux, self.fluxErr = remove_flux_extinction(self.flux, self.fluxErr, self.passband, mwebv)
        
        # renorm flux
        if renorm:
            if mu is None:
                raise TypeError("Need arg 'mu' to renormalise flux")
            else:
                self.flux, self.fluxErr = renorm_flux_lightcurve(self.flux, self.fluxErr, mu)
                

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

            # these cuts are only applied if the light curve is provided in magnitudes
            # they handle annoying dummy values and very low S/N points
            if mag:
                mask = np.logical_and(mask, self.flux > 0.)
                mask = np.logical_and(mask, self.flux <= 99.)
                mask = np.logical_and(mask, self.fluxErr <= 9.)

            if 'photflag' in self._extra_cols:
                saveind = (self.photflag[mask] > 0)
                mask[saveind] = True

            self.time = self.time[mask]
            self.flux = self.flux[mask]
            self.fluxErr = self.fluxErr[mask]
            self.obsId = self.obsId[mask]
            self.passband = self.passband[mask]
            self.zeropoint = self.zeropoint[mask]
            for key in self._extra_cols:
                val = getattr(self, key)
                setattr(self, key, val[mask])

            if not clean:
                return self.finalize()

            # we should still update filters in case all the data in a filter was invalid
            self.filters = list(set(self.passband))

            # begin cleaning the lightcurve - hic sunt dracones
            if mag:
                # remove points with large mag error
                mask = self.fluxErr < 0.5
            else:
                # or remove points with S/N < 1
                mask = self.fluxErr < np.abs(self.flux)
            self.time = self.time[mask]
            self.flux = self.flux[mask]
            self.fluxErr = self.fluxErr[mask]
            self.obsId = self.obsId[mask]
            self.passband = self.passband[mask]
            self.zeropoint = self.zeropoint[mask]
            for key in self._extra_cols:
                val = getattr(self, key)

            t = None
            f = None
            df = None
            zpt = None
            pbs = None
            oid = None
            filtered_extra_cols = {}

            # do some sigmaclipping to reject outliers
            for pb in self.filters:
                indf = (self.passband == pb)
                filtered_err = sigma_clip(self.fluxErr[indf], sigma=3., iters=5, copy=True)
                bad1 = filtered_err.mask

                # filtered_flux = sigma_clip(self.flux[indf], sigma=7., iters=5, copy=True)
                # bad2 = filtered_flux.mask
                # useind = ~np.logical_or(bad1, bad2)

                if 'photflag' in self._extra_cols:
                    saveind = (self.photflag[indf][bad1] > 0)
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

                for key in self._extra_cols:
                    orig = getattr(self, key)
                    val = filtered_extra_cols.get(key, None)
                    if val is None:
                        val = orig[indf][useind]
                        filtered_extra_cols[key] = val
                    else:
                        val = np.concatenate((val, orig[indf][useind]))
                        filtered_extra_cols[key] = val

            self.time = t
            self.flux = f
            self.fluxErr = df
            self.passband = pbs
            self.zeropoint = zpt
            self.obsId = oid
            for key in self._extra_cols:
                val = filtered_extra_cols.get(key)
                setattr(self, key, val)

            return self.finalize()

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
        self._good_filters = set(['u', 'g', 'r', 'i', 'z', 'Y'])
        avail_filters = set(self.passband)
        use_filters = self._good_filters & avail_filters
        if not avail_filters.issubset(use_filters):
            message = 'Number of useful filters ({}) does not equal number available filters ({}) - some filters will not be used'.format(
                ''.join(use_filters), ''.join(avail_filters))
            warnings.warn(message, RuntimeWarning)
        self.filters = list(use_filters)

        self.nobs = len(self.time)
        if self.nobs == 0:
            message = 'Object {} with locus ID {} has no good observations.'.format(self.objectId, self.locusId)
            # raise ValueError(message)

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


def renorm_flux_lightcurve(flux, fluxerr, mu):
    """ Normalise flux light curves with distance modulus."""
    d = 10 ** (mu / 5 + 1)
    dsquared = d ** 2

    norm = 1e19

    fluxout = flux * dsquared / norm
    fluxerrout = fluxerr * dsquared / norm

    return fluxout, fluxerrout


def remove_flux_extinction(flux, fluxerr, passband, mwebv):
    passbands = ['u', 'g', 'r', 'i', 'z', 'Y']
    PB_WAVE = np.array([3569.5, 4766.5, 6214.5, 7544.5, 8707.5, 10039.5])

    # Using negative a_v so that extinction.apply works in reverse and removes the extinction
    extinctions = extinction.fitzpatrick99(wave=PB_WAVE, a_v=-3.1 * mwebv, r_v=3.1, unit='aa')

    for i, pb in enumerate(passbands):
        flux_pb = flux[passband == pb]
        fluxerr_pb = fluxerr[passband == pb]

        extinction.apply(extinctions[i], flux_pb)
        flux_out = extinction.apply(extinctions[i], flux_pb, inplace=False)
        fluxerr_out = extinction.apply(extinctions[i], fluxerr_pb, inplace=False)

    return flux_out, fluxerr_out

