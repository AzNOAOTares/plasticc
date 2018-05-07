import h5py
import numpy as np

import helpers


def get_feature_names(passbands, ignore=()):
    feature_names = ['objid', 'redshift']
    feature_names = sum([['variance_%s' % p, 'kurtosis_%s' % p, 'filt-variance_%s' % p, 'filt-kurtosis_%s' % p,
                           'shapiro_%s' % p, 'p-value_%s' % p, 'skew_%s' % p, 'q31_%s' % p,
                           'stetsonk_%s' % p, 'acorr_%s' % p, 'von-neumann_%s' % p, 'hlratio_%s' % p,
                           'amplitude_%s' % p, 'filt-amplitude_%s' % p,  'somean_%s' % p, 'rms_%s' % p, 'mad_%s' % p,
                           'stetsonj_%s' % p, 'stetsonl_%s' % p, 'entropy_%s' % p, 'nobs4096_%s' % p,
                           'risetime_%s' % p, 'riserate_%s' % p,
                           'cesium_flux_percentile_ratio_mid20_%s' % p, 'cesium_flux_percentile_ratio_mid50_%s' % p, 'cesium_flux_percentile_ratio_mid65_%s' % p, 'cesium_flux_percentile_ratio_mid80_%s' % p, 'cesium_max_slope_%s' % p, 'cesium_maximum_%s' % p, 'cesium_median_%s' % p, 'cesium_median_absolute_deviation_%s' % p, 'cesium_percent_amplitude_%s' % p, 'cesium_period_fast_%s' % p, 'cesium_qso_log_chi2_qsonu_%s' % p, 'cesium_qso_log_chi2nuNULL_chi2nu_%s' % p, 'cesium_fold2P_slope_90percentile_%s' % p, 'cesium_freq1_amplitude1_%s' % p, 'cesium_freq1_amplitude2_%s' % p, 'cesium_freq1_amplitude3_%s' % p, 'cesium_freq1_amplitude4_%s' % p, 'cesium_freq1_freq_%s' % p, 'cesium_freq1_lambda_%s' % p, 'cesium_freq1_rel_phase2_%s' % p, 'cesium_freq1_rel_phase3_%s' % p, 'cesium_freq1_rel_phase4_%s' % p, 'cesium_freq1_signif_%s' % p, 'cesium_freq2_amplitude1_%s' % p, 'cesium_freq2_amplitude2_%s' % p, 'cesium_freq2_amplitude3_%s' % p, 'cesium_freq2_amplitude4_%s' % p, 'cesium_freq2_rel_phase2_%s' % p, 'cesium_freq2_rel_phase3_%s' % p, 'cesium_freq2_rel_phase4_%s' % p, 'cesium_freq3_amplitude1_%s' % p, 'cesium_freq3_amplitude2_%s' % p, 'cesium_freq3_amplitude3_%s' % p, 'cesium_freq3_amplitude4_%s' % p, 'cesium_freq3_rel_phase2_%s' % p, 'cesium_freq3_rel_phase3_%s' % p, 'cesium_freq3_rel_phase4_%s' % p, 'cesium_freq_amplitude_ratio_21_%s' % p, 'cesium_freq_amplitude_ratio_31_%s' % p, 'cesium_freq_n_alias_%s' % p, 'cesium_freq_signif_ratio_21_%s' % p, 'cesium_freq_signif_ratio_31_%s' % p, 'cesium_freq_varrat_%s' % p, 'cesium_freq_y_offset_%s' % p, 'cesium_medperc90_2p_p_%s' % p, 'cesium_p2p_scatter_pfold_over_mad_%s' % p, 'cesium_p2p_ssqr_diff_over_var_%s' % p, 'cesium_scatter_res_raw_%s' % p] for p in passbands], [])

    color_fields = []
    for i, pb1 in enumerate(passbands):
        for j, pb2 in enumerate(passbands):
            if i < j:
                color = pb1 + '-' + pb2
                color_fields += ['amp %s' % color]
                color_fields += ['mean %s' % color]
    feature_names += color_fields
    period_fields = ['period1', 'period_score1', 'period2', 'period_score2', 'period3', 'period_score3', 'period4',
                     'period_score4', 'period5', 'period_score5']
    feature_names += period_fields

    for name in ignore:
        feature_names.remove(name)

    feature_names = np.array(feature_names)

    return feature_names


def get_features(fpath, data_release, field_in='%', model_in='%', aggregate_classes=False):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    features = np.array(hdffile[data_release])
    hdffile.close()
    if aggregate_classes:
        agg_map = helpers.aggregate_sntypes(reverse=True)

    indexes = []
    modelList = []
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        modelList.append(model)

        if aggregate_classes is True:
            submodels = agg_map[int(model_in)]
            for m in submodels:
                if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(m)):
                    indexes.append(i)
        else:
            if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(model_in)):
                indexes.append(i)

    features = features[indexes]

    return features