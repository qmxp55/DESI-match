from __future__ import division, print_function
import sys
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.ticker import NullFormatter
from astropy import units as u
import fitsio
import pandas as pd
from astropy.io import ascii
import glob
import sys, os, time, argparse
import matplotlib.patches as patches
import warnings; warnings.simplefilter('ignore')

from desitarget.targetmask import desi_mask, bgs_mask
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
#from pyvenn import venn
from matplotlib_chord_diagram import matplotlib_chord

from main_def import cut_sweeps, cut, load_catA, get_mag_decals


if __name__ == "__main__":

    #search and load the corresponding sweep files for the selected area
    sweep_dir = os.path.join('/global/project/projectdirs/cosmo/data/legacysurvey/','dr7', 'sweep', '7.1')
    catS = cut_sweeps(200, 230, -2,5, sweep_dir)

    #load and cut desitarget to catalogue footprint
    print('loading desitarget OLD')
    columns0 = ['RA', 'DEC', 'DESI_TARGET']
    catT = fitsio.read('targets-dr7.1-0.23.0.fits', columns=columns0, ext=1)
    catT = cut(200, 230, -2,5, catT)
    print('# DESITARGET OLD', len(catT))

    #load desitarget with new BGS target selection
    print('loading desitarget NEW')
    catT2 = fitsio.read('/global/homes/q/qmxp55/DESI/desitarget_data/targetdir/targets-dr7.1-0.26.0.fits', columns=columns0, ext=1)
    catT2 = cut(200, 230, -2,5, catT2)
    print('# DESITARGET NEW', len(catT2))

    print('Applying Star-Galaxy separation cuts')
    cat, catr, cat0, matrix_in, matrix_out, mask_list_MAG, dropped = load_cat(catS, catT, FILE=False, desitarget=True)

    Nbgs_target_old_faint = len(catT[((np.uint64(catT['DESI_TARGET']) & np.uint64(bgs_mask['BGS_FAINT']))!=0)])
    Nbgs_target_old_bright = len(catT[((np.uint64(catT['DESI_TARGET']) & np.uint64(bgs_mask['BGS_BRIGHT']))!=0)])
    Nbgs_target_old = len(catT[((np.uint64(catT['DESI_TARGET']) & np.uint64(desi_mask['BGS_ANY']))!=0)])
    Nbgs_target_new_faint = len(catT2[((np.uint64(catT2['DESI_TARGET']) & np.uint64(bgs_mask['BGS_FAINT']))!=0)])
    Nbgs_target_new_bright = len(catT2[((np.uint64(catT2['DESI_TARGET']) & np.uint64(bgs_mask['BGS_BRIGHT']))!=0)])
    Nbgs_target_new_wise = len(catT2[((np.uint64(catT2['DESI_TARGET']) & np.uint64(bgs_mask['BGS_KNOWN_ANY']))!=0)])
    Nbgs_target_new = len(catT2[((np.uint64(catT2['DESI_TARGET']) & np.uint64(desi_mask['BGS_ANY']))!=0)])


    rflux = catS['FLUX_R'] / catS['MW_TRANSMISSION_R']
    bgs = rflux > 10**((22.5-20.0)/2.5)
    bgs &= catS['TYPE'] != b'PSF '
    bgs_faint = rflux > 10**((22.5-20.0)/2.5)
    bgs_faint &= rflux <= 10**((22.5-19.5)/2.5)
    bgs_faint &= catS['TYPE'] != b'PSF '
    bgs_bright = rflux <= 10**((22.5-19.5)/2.5)
    bgs_bright &= catS['TYPE'] != b'PSF '

    gmag, rmag, zmag, w1mag, Gmag, rr = get_mag_decals(cat)
    Grr = Gmag - rr  #rr is without extinction correction
    bgs_new = Grr > 0.6
    bgs_new |= Gmag == 0
    cat_new = cat[bgs_new]
    g, r, z, w1, G, rr = get_mag_decals(cat_new)
    bgs_wise = Grr < 0.4
    bgs_wise &= Grr > -1.0
    bgs_wise &= zmag - w1mag - (gmag - rmag) > -0.5
    bgs_wise &= cat['FLUX_W1']*np.sqrt(cat['FLUX_IVAR_W1']) > 5
    
    print(np.sum(catS['BRIGHTSTARINBLOB']))
    print('------------------------')
    
    print('BGS_FAINT (DESITARGET OLD, ME):', Nbgs_target_old_faint, len(catS[bgs_faint]))
    print('BGS_BRIGHT (DESITARGET OLD, ME):', Nbgs_target_old_bright, len(catS[bgs_bright]))
    print('BGS_TOTAL (DESITARGET OLD, ME):', Nbgs_target_old, len(catS[bgs]))
    print('===================================')
    print('BGS_FAINT (DESITARGET NEW, ME):', Nbgs_target_new_faint, len(cat_new[np.logical_and(r < 20, r >= 19.5)]))
    print('BGS_BRIGHT (DESITARGET NEW, ME):', Nbgs_target_new_bright, len(cat_new[r < 19.5]))
    print('BGS_WISE (DESITARGET NEW, ME):', Nbgs_target_new_wise, len(cat[bgs_wise]))
    print('BGS_TOTAL (DESITARGET NEW, ME):', Nbgs_target_new, len(cat[bgs_wise]) + len(cat_new[np.logical_and(r < 20, r >= 19.5)]) + len(cat_new[r < 19.5]))


