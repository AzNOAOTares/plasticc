#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')
sys.path.append(WORK_DIR)
import numpy as np

import ANTARES_object

import plasticc
import plasticc.get_data

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
from scipy.stats import gaussian_kde, describe
from astropy.coordinates import SkyCoord, ICRS
import healpy as hp

cmap2 = cm.inferno
cmap2.set_bad('grey')
cmap2.set_under('white')

def map_from_arrays(l, b, obj, model_name, pix_2_coord_dict=None,\
                 interp=True, interp_min=0., verbose=False, overwrite=False, NSIDE=32, fig_dir=None):
    """
    Build a Healpix map from some arrays, l, b, objname
    
    This is a fairly involved function - loads map and returns if it exists, else will generate from scratch
    Optionally interpolates missing pixels that are surrounded by at least 5 (hardcoded) good neighbors
    Returns a map, a dictionary of mapping from coordinate to Healpix pixel, and the NSIDE of the map (32 by default)
    The dictionary mapping can be reused on subsequent calls to generate more maps from the same CSV
    """
   
    DEFAULT_NSIDE = 32
    if NSIDE < 8:
        NSIDE = DEFAULT_NSIDE
    # this is a tradeoff effectively set by native map resolution vs LSST Simlib pointing resolution
    # the higher we raise this, the higher the resolution of the Healpix map
    # but the more NaN pixels that can't be interpolated over
    # we just need to make it large enough that each HEALpix Pixel covers at least one Simlib pointing

    filename = '{}.fits'.format(model_name)
    if fig_dir is None:
        fig_dir = os.getcwd()
    filename = os.path.join(fig_dir, filename)
    if os.path.exists(filename) and not overwrite:
        try:
            mapdata = hp.fitsfunc.read_map(filename, h=False, verbose=True)       
            return mapdata, {}, NSIDE
        except Exception as e:
            pass
    print("Rendering {} map".format(model_name))                
    npix = hp.nside2npix(NSIDE)
    newmap = np.zeros(npix, dtype=np.float32)
    array_inds = np.arange(npix)
    newmap[:] = np.nan
        
    # if we know the mapping between 1-D pixel and position on the sky, then we can just restore that
    # Since Rahul has a fixed set of Simlib positions, this should rapidly become populated
    if pix_2_coord_dict is None:
        theta = l
        phi   = b
        pix_2_coord_dict = {}
        pix_coord = hp.pixelfunc.ang2pix(NSIDE, theta, phi, nest=False, lonlat=True)
        nmax = len(pix_coord)
    else:
        pix_coord = pix_2_coord_dict.keys()
        nmax = len(pix_coord)
        if nmax == 0:
            theta = l
            phi   = b
            pix_coord = hp.pixelfunc.ang2pix(NSIDE, theta, phi, nest=False, lonlat=True)
        
        
    # this keeps a track of if we considered this pixel for the map
    # since multiple positions can map to the same coordinate (and do, particularly near the pole)
    used_coord = {}
    
    nmax = len(pix_coord)
    for i, coord in enumerate(pix_coord):
        # skip pixels we've already set
        if coord in used_coord:
            continue
        
        # if we have the mapping from pixel to coordinate
        # then just restore the indices of coordinates that contribute to this pixel
        ind = pix_2_coord_dict.get(coord, None)
        if ind is None:
            ind = np.where(pix_coord == coord)[0]
        
        # we're just making a hit map so just need the length
        newmap[coord] = len(obj[ind])
        pix_2_coord_dict[coord] = ind
        used_coord[coord] = 1
    
    # what was the extent of the data we set
    used_coord = np.array(used_coord.keys())
    min_pix = used_coord.min()
    max_pix = used_coord.max()
    
    bad_ind = newmap < interp_min
    newmap[bad_ind] = np.nan
    
    # if we aren't interpolating, just save and return
    if not interp:
        # write this map to a file so we can restore in future without recomputing
        try:
            hp.fitsfunc.write_map(filename, newmap, nest=False, coord='G',\
                          partial=False, fits_IDL=False,\
                          column_names=['num',], overwrite=True)
        except Exception as e:
            print("{}".format(e))
            pass
        return newmap, pix_2_coord_dict, NSIDE

    # attempt to interpolate bad pixels using any neighboring good pixels
    nan_coords = ~np.isfinite(newmap)
    newmap[nan_coords] = hp.pixelfunc.UNSEEN

    print('INTERPOLATING')
    # what are the bad coordinates *within the range of the coordinates we considered*
    bad_coords = np.where((nan_coords) & (array_inds > min_pix) & (array_inds < max_pix))[0]
    ctr = 0
    use_pix = []
    use_val = []
    for coord in bad_coords:
        theta, phi = hp.pixelfunc.pix2ang(NSIDE, coord, nest=False, lonlat=True)
        interp_pix = hp.pixelfunc.get_all_neighbours(NSIDE, theta, phi, nest=False, lonlat=True)
        ind = interp_pix != -1
        interp_pix = interp_pix[ind]
        pix_val    = newmap[interp_pix]
        good_pix   = pix_val != hp.pixelfunc.UNSEEN
        ngood = len(pix_val[good_pix])
        # there's 8 neighbors, total, but some may have no neighbor (-1) or be set to UNSEEN themselves.
        # require at least 5 for interpolation. If not give up.
        if ngood >= 5:      
            interp_val = np.mean(pix_val[good_pix])
        else:
            interp_val = hp.pixelfunc.UNSEEN
        if verbose and ctr < 10: 
            print(coord, interp_val)
            ctr+=1       
        if interp_val != hp.pixelfunc.UNSEEN:
            if interp_val < interp_min:
                interp_val = interp_min
        # save the interpolated value and coordinate
        # do not set it inside the loop
        # we don't want to use interpolated values themselves for further interpolation
        use_pix.append(coord)
        use_val.append(interp_val)
        
    use_pix = np.array(use_pix)
    use_val = np.array(use_val)
    newmap[use_pix] = use_val
    
    stillbad_ind = np.where(newmap == hp.pixelfunc.UNSEEN)
    newmap[stillbad_ind] = np.nan
    
    # write this map to a file so we can restore in future without recomputing
    try:
        hp.fitsfunc.write_map(filename, newmap, nest=False, coord='G', partial=False, fits_IDL=False,\
                          column_names=['num',], overwrite=True)
    except Exception as e:
        print("{}".format(e))
        pass
    
    return newmap, pix_2_coord_dict, NSIDE
 


def main():


    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    fig_dir = os.path.join(WORK_DIR, 'Figures', data_release)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    _ = kwargs.pop('model')
    _ = kwargs.get('field')
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl']
    
    out_field = 'WFD'
    kwargs['field'] = out_field 

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)

    fig2 = plt.figure(figsize=(15,10))
    ax2 = fig2.add_subplot(111)

    cmap = plt.cm.tab20
    nlines = len(sntypes.keys())
    color = iter(cmap(np.linspace(0,1,nlines)))
    legend = []
    labels = []

    b_range = np.arange(-90, 90.1, 0.1)

    with PdfPages(f'{fig_dir}/rate_analysis/rate_vs_gallat_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):
            
            kwargs['model'] = model 
            kwargs['big'] = True
            head = getter.get_lcs_headers(**kwargs)

            model_name = sntypes[model]
            print(model_name)
            
            head = list(head)
            nobs = len(head) 
            if nobs == 0:
                continue

            objid, _, _, ra, dec = zip(*head) 

            ra = np.array(ra)
            dec = np.array(dec)
            objid = np.array(objid)
                
            c = SkyCoord(ra, dec, "icrs", unit='deg')
            gal = c.galactic
            l = gal.l.value 
            b = gal.b.value 

            fig1 = plt.figure(figsize=(15,10))
            fig_num = plt.gcf().number

            pix_2_coord_dict = {}
            map_dir = os.path.join(fig_dir, 'rate_analysis', 'maps', out_field)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            full_model_name = f'{model_name}_{model}_{out_field}_{data_release}'
            long_model_name = f'{model_name}_{model}'
            model_map, pix_2_coord_dict, NSIDE = map_from_arrays(l, b, objid, full_model_name,\
                          pix_2_coord_dict = pix_2_coord_dict, interp=False, overwrite=False, NSIDE=16, fig_dir=map_dir)

            hp.mollview(model_map, coord=['G',], norm='hist', title=model_name, cmap=cmap2, fig=fig_num)
            hp.graticule(coord=['C'])
            pdf.savefig(fig1)

            if len(b) > 10:
                c = next(color)
                density = gaussian_kde(b, bw_method='scott')
                ax2.plot(b_range, density(b_range), color=c)

                patch = mpatches.Patch(color=c, label=long_model_name)
                legend.append(patch)
                labels.append(long_model_name)

            if nobs > 10:
                alpha = 1./int(np.log10(nobs))
            if alpha <= 0.2:
                alpha/=10
        #end loop over models
        fig2.legend(legend, labels, ncol=5, fontsize='small', loc='upper center')
        fig2.tight_layout(rect=[0,0,1,0.93])
        pdf.savefig(fig2)
    #close pdf fig



if __name__=='__main__':
    sys.exit(main())
