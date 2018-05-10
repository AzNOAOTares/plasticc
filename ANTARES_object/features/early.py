import numpy as np

class EarlyMixin(object):
    """
    Methods to derive early classification features for LAobjects
    """

    def get_color_at_n_days(self, n, flux_pb1, flux_pb2, time):
        color = flux_pb1 / flux_pb2
        nday_color = np.interp(n, time, color)
        color_slope = nday_color / n

        return nday_color, color_slope

    def get_rise_time(self, recompute=False):
        """
        Compute the earlyrisetime of the light curve.
        """

        risetime = getattr(self, 'risetime', None)
        if risetime is not None:
            if not recompute:
                return risetime

        risetime = {}
        outlc = self.get_lc(recompute=recompute)

        for i, pb in enumerate(outlc):
            tlc = outlc.get(pb)
            ttime, tFlux, tFluxErr, tFluxUnred, tFluxErrUnred, tFluxRenorm, tFluxErrRenorm, tphotflag, tzeropoint, tobsId = tlc

            photmask = tphotflag >= constants.GOOD_PHOTFLAG
            ttime = ttime[photmask]
            tFluxRenorm = tFluxRenorm[photmask]

            # Assuming that ttime = 0 is the peak_mjd, then:

            if len(tFluxRenorm) <= 1:  # if t Flux is nearly empty
                trisetime = -1.
            else:
                trisetime = 0 - ttime[0]
            risetime[pb] = trisetime

        self.risetime = risetime
        return risetime


    def get_early_rise_rate(self, recompute=False):
        """
        Compute the early rise rate (slope) of the light curve.
        """

        riserate = getattr(self, 'riserate', None)
        if riserate is not None:
            if not recompute:
                return riserate

        risetime = self.get_rise_time(recompute=recompute)
        amplitude = self.get_filtered_amplitude(recompute=recompute)

        riserate = {}
        for pb in risetime:
            trisetime = amplitude.get(pb, -1.)
            tamplitude = amplitude.get(pb, 0.)
            if trisetime == -1.:
                triserate = 0
            else:
                triserate = tamplitude / trisetime
            riserate[pb] = triserate
        self.riserate = riserate
        return riserate