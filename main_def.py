import numpy as np
import sys, os, time, argparse, glob
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.ticker import NullFormatter
from astropy import units as u
import pandas as pd
from astropy.io import ascii
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles

from desitarget.targetmask import desi_mask, bgs_mask

def chord(matrix, nodes):
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes([0,0,1,1])

#nodePos = chordDiagram(flux, ax, colors=[hex2rgb(x) for x in ['#666666', '#66ff66', '#ff6666', '#6666ff']])
    nodePos = matplotlib_chord.chordDiagram(matrix, ax)
    ax.axis('off')
    prop = dict(fontsize=16*0.8, ha='center', va='center')
    nodes = nodes#['non-crystal', 'FCC', 'HCP', 'BCC', 'other']
    for i in range(len(matrix)):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
        
def stacked_bar(data, series_labels, category_labels=None, 
                show_values=False, value_format="{}", y_label=None, 
                grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
            
class Point:

    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord

class Rectangle:
    def __init__(self, bottom_left, top_right, colour):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.colour = colour

    def intersects(self, other):
        return not (self.top_right.x <= other.bottom_left.x or self.bottom_left.x >= other.top_right.x or self.top_right.y <= other.bottom_left.y or self.bottom_left.y >= other.top_right.y)
    
    def plot(self, other):
        fig, ax = plt.subplots(figsize=(15,8))
        rect = patches.Rectangle((self.bottom_left.x,self.bottom_left.y), abs(self.top_right.x - self.bottom_left.x), abs(self.top_right.y - self.bottom_left.y),linewidth=1.5, alpha=0.5, color='r')
        rect2 = patches.Rectangle((other.bottom_left.x,other.bottom_left.y), abs(other.top_right.x - other.bottom_left.x), abs(other.top_right.y - other.bottom_left.y),linewidth=1.5, alpha=0.5, color='blue')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        xlims = np.array([self.bottom_left.x, self.top_right.x, other.bottom_left.x, other.top_right.x])
        ylims = np.array([self.bottom_left.y, self.top_right.y, other.bottom_left.y, other.top_right.y])
        ax.set_xlim(xlims.min()-1, xlims.max()+1)
        ax.set_ylim(ylims.min()-1, ylims.max()+1)
        #plt.show()
        
def cut(ramin, ramax, decmin, decmax, catalog):
    
    mask = np.logical_and(catalog['RA'] >= ramin, catalog['RA'] <= ramax)
    mask &= np.logical_and(catalog['DEC'] >= decmin, catalog['DEC'] <= decmax)
    cat = catalog[mask]
    #print('CUT #:',len(cat))
    
    return cat

def cut_sweeps(ramin, ramax, decmin, decmax, sweep_dir):
    
    cat1_paths = sorted(glob.glob(os.path.join(sweep_dir, '*.fits')))
    #cat = np.empty()
    j = 0
    
    for fileindex in range(len(cat1_paths)):
    #for fileindex in range(5):

        cat1_path = cat1_paths[fileindex]
        filename = cat1_path[-26:-5]
        brick = cat1_path[-20:-5]
        ra1min = float(brick[0:3])
        ra1max = float(brick[8:11])
        dec1min = float(brick[4:7])
        if brick[3]=='m':
            dec1min = -dec1min
        dec1max = float(brick[-3:])
        if brick[-4]=='m':
            dec1max = -dec1max
        
        r1=Rectangle(Point(ramin,decmin), Point(ramax, decmax), 'red')
        r2=Rectangle(Point(ra1min, dec1min), Point(ra1max, dec1max), 'blue')
        
        if not r1.intersects(r2):
            continue
        
        if j == 0:
            cat = fitsio.read(cat1_path)
            cat = cut(ramin, ramax, decmin, decmax, cat)
            print(filename, len(cat))
            j += 1
            continue
        
        name = fitsio.read(cat1_path, ext=1)
        name = cut(ramin, ramax, decmin, decmax, name)
        print(filename, len(name))
        
        cat = np.concatenate((cat, name))
        j += 1
        
    print('Bricks that matched: %i' %(j))
    print('Sample region # objects: %i' %(len(cat)))
    
    return cat

def add_desitarget(catS, catT):
    
    print('catS:', len(catS))
    print('catT:', len(catT))
    
    raS = np.array(catS['RA'])
    decS = np.array(catS['DEC'])
    
    raT = np.array(catT['RA'])
    decT = np.array(catT['DEC'])

    # Matching catalogs
    print("Matching...")
    skycatS = SkyCoord(raS*u.degree,decS*u.degree, frame='icrs')
    skycatT = SkyCoord(raT*u.degree,decT*u.degree, frame='icrs')
    idx, d2d, _ = skycatT.match_to_catalog_sky(skycatS)
    print('len idx:',len(idx))
    print('d2d != 0',len(d2d[d2d != 0]))
        # For each object in catT, a closest match to catS is found. Thus not all catS objects are included. 
        # idx is the catS index for catT -- idx[0] is the catS index that the first catT object matched.
        # Similarly d2d is the distance between the matches. 
    
    desitarget = np.zeros(len(catS))
    bgstarget = np.zeros(len(catS))
    catT_ra = np.zeros(len(catS))
    catT_dec = np.zeros(len(catS))
    #mask = d2d != 0
    #idx = idx[mask]
    desitarget[idx] = catT['DESI_TARGET']
    bgstarget[idx] = catT['BGS_TARGET']
    catT_ra[idx] = catT['RA']
    catT_dec[idx] = catT['DEC']
    
    for i in idx:
        diff_dec = catS['DEC'][i] - catT_dec[i]
        diff_ra = catS['RA'][i] - catT_ra[i]
        if (diff_dec != 0) or (diff_ra != 0):
            print('ERROR! PROBLEM WITH INDEX')
            break

    return desitarget, bgstarget, idx, d2d

def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def get_mag_decals(df):
    
    #df = df[(df['FLUX_R'] > 0) & (df['FLUX_G'] > 0) & (df['FLUX_Z'] > 0) & (df['FLUX_W1'] > 0)]
    rmag =  flux_to_mag(df['FLUX_R']/df['MW_TRANSMISSION_R'])
    gmag = flux_to_mag(df['FLUX_G']/df['MW_TRANSMISSION_G'])
    zmag = flux_to_mag(df['FLUX_Z']/df['MW_TRANSMISSION_Z'])
    w1mag = flux_to_mag(df['FLUX_W1']/df['MW_TRANSMISSION_W1'])
    Gmag = df['GAIA_PHOT_G_MEAN_MAG']
    rr = flux_to_mag(df['FLUX_R'])
    if len(df) != len(rmag):
        print('ERROR! lenghts do not match')

    return gmag, rmag, zmag, w1mag, Gmag, rr

def target_con(df, NAME):
    
    target_desi = desi_mask.names() #("BGS_ANY", "QSO", "LRG", "ELG", "BAD_SKY", "NO_TARGET", "BRIGHT_OBJECT", "MWS_ANY", "STD", "STD_BRIGHT", "STD_WD", "IN_BRIGHT_OBJECT", "NEAR_BRIGHT_OBJECT")
    target_bgs = bgs_mask.names()
    print('Total #%s: %g --- desitarget matches: %g --- bgstarget matches: %g'%(NAME,len(df),len(df[df['desitarget'] != 0]) ,len(df[df['bgstarget'] != 0])))
    print('Of which...')
    target_l = []
    for i in target_desi:
        
        target_per = len(df[((np.uint64(df['desitarget']) & np.uint64(desi_mask[i]))!=0)])
        if target_per == 0:
            continue
        print('%s: %2.4g %%' %(i, 100*target_per/len(df)))
        target_l.append([i, 100*target_per/len(df)])
        
    for i in target_bgs:
        
        target_per = len(df[((np.uint64(df['bgstarget']) & np.uint64(bgs_mask[i]))!=0)])
        if target_per == 0:
            continue
        print('%s: %2.4g %%' %(i, 100*target_per/len(df)))
        target_l.append([i, 100*target_per/len(df)])
    
    print('%s: %2.4g %%' %('G==0', 100*len(df[df['GAIA_PHOT_G_MEAN_MAG'] == 0])/len(df)))
    target_l.append(['G==0', 100*len(df[df['GAIA_PHOT_G_MEAN_MAG'] == 0])/len(df)])
    print('%s: %2.4g %%' %('PSF', 100*len(df[df['TYPE'] == b'PSF '])/len(df)))
    target_l.append(['PSF', 100*len(df[df['TYPE'] == b'PSF '])/len(df)])
    print('%s: %2.4g %%' %('no-PSF', 100*len(df[df['TYPE'] != b'PSF '])/len(df)))
    target_l.append(['no-PSF', 100*len(df[df['TYPE'] != b'PSF '])/len(df)])
    print('%s: %2.4g %%' %('desitarget non matches', 100*len(df[df['desitarget'] == 0])/len(df)))
    target_l.append(['desitarget non matches', 100*len(df[df['desitarget'] == 0])/len(df)])
    print('%s: %2.4g %%' %('bgstarget non matches', 100*len(df[df['bgstarget'] == 0])/len(df)))
    target_l.append(['bgstarget non matches', 100*len(df[df['bgstarget'] == 0])/len(df)])
    
    return target_l

def matrix_plot(matrix, matrix_names_rows, matrix_names_columns):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax)
    
    matrix_names_rows = np.append(['TEST'], np.array(matrix_names_rows), axis=0)
    matrix_names_columns = np.append(['TEST'], np.array(matrix_names_columns), axis=0)
    ax.set_xticklabels(matrix_names_columns)
    ax.set_yticklabels(matrix_names_rows)
    
    
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.show()
    
def bar_plot(matrix):
    
    # set width of bar
    barWidth = 0.10
    plt.figure(figsize=(12, 6))
 
    #set height of bar
    bars = matrix.T
 
    # Set position of bar on X axis
    r1 = np.arange(len(bars.iloc[0]))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    r5 = [x + barWidth for x in r4]
    r6 = [x + barWidth for x in r5]
    r7 = [x + barWidth for x in r6]

    rpos = (r1, r2, r3, r4, r5, r6, r7)
    color = ('red', 'green', 'blue', 'grey', 'orange', 'cyan', 'pink')
 
    # Make the plot
    for i in range(len(matrix.T.iloc[0])):
        plt.bar(rpos[i], bars.iloc[i], color=color[i], width=barWidth, edgecolor='white', label= matrix.columns[i])
 
    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars.iloc[0]))], list(np.array(matrix.index)))
 
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    
def outa(lista):
    return np.array([not i for i in lista])
    
def venn_plot(catr, cat, mask_list_MAG, i, j, k, labels, pos='out', title='OUT'):
    
    plt.figure(figsize=(6, 6))
    if pos == 'out':
        venn3([set(catr['RA'][[not i for i in mask_list_MAG[i]]]), set(catr['RA'][[not i for i in mask_list_MAG[j]]]), set(catr['RA'][[not i for i in mask_list_MAG[k]]])], set_labels = labels)
    if pos == 'in':
        venn3([set(catr['RA'][mask_list_MAG[i]]), set(catr['RA'][mask_list_MAG[j]]), set(catr['RA'][mask_list_MAG[k]])], set_labels = labels)
    if pos == 'mid':
        venn3([set(catr['RA'][[not i for i in mask_list_MAG[i]]]), set(catr['RA'][mask_list_MAG[j]]), set(catr['RA'][mask_list_MAG[k]])], set_labels = labels)
    
    venn_outs = Table()
    venn_outs[labels[0]] = (outa(mask_list_MAG[i]) & (mask_list_MAG[j]) & (mask_list_MAG[k]))
    venn_outs[labels[1]] = (outa(mask_list_MAG[j]) & (mask_list_MAG[i]) & (mask_list_MAG[k]))
    venn_outs[labels[2]] = (outa(mask_list_MAG[k]) & (mask_list_MAG[i]) & (mask_list_MAG[j]))
    
    venn_outs[labels[0]+'-'+labels[1]] = (outa(mask_list_MAG[i]) & (outa(mask_list_MAG[j])))
    venn_outs[labels[0]+'-'+labels[2]] = (outa(mask_list_MAG[i]) & (outa(mask_list_MAG[k])))
    venn_outs[labels[1]+'-'+labels[2]] = (outa(mask_list_MAG[j]) & (outa(mask_list_MAG[k])))
    
    venn_outs[labels[0]+'-'+labels[1]+'-'+labels[2]] = (outa(mask_list_MAG[i]) & (outa(mask_list_MAG[j])) & (outa(mask_list_MAG[k])))
    
    out_tot = np.sum(venn_outs[labels[0]]) + np.sum(venn_outs[labels[1]]) + np.sum(venn_outs[labels[2]]) + np.sum(venn_outs[labels[0]+'-'+labels[1]]) + np.sum(venn_outs[labels[0]+'-'+labels[2]]) + np.sum(venn_outs[labels[1]+'-'+labels[2]]) - 3 * np.sum(venn_outs[labels[0]+'-'+labels[1]+'-'+labels[2]]) 
    print('Objects out alone...')
    for i in range(3):
        print('%s: %d' %(labels[i], np.sum(venn_outs[labels[i]])))
        
    print('Objects out together...')
    print('%s: %d' %(labels[0]+'-'+labels[1], np.sum(venn_outs[labels[0]+'-'+labels[1]])))
    print('%s: %d' %(labels[0]+'-'+labels[2], np.sum(venn_outs[labels[0]+'-'+labels[2]])))
    print('%s: %d' %(labels[1]+'-'+labels[2], np.sum(venn_outs[labels[1]+'-'+labels[2]])))
    print('%s: %d' %(labels[0]+'-'+labels[1]+'-'+labels[2], np.sum(venn_outs[labels[0]+'-'+labels[1]+'-'+labels[2]])))
    print('---------------------')
    print('%d objects out of a total %d = %2.2f %%' %(out_tot, len(catr) - len(cat), 100*out_tot/(len(catr) - len(cat))))
    
    plt.title('%2.2f %% %s' %(100*out_tot/(len(catr) - len(cat)), title))
    
    return venn_outs

def save_from_venn(catr, vennData, mask, N, outFile):
    #catr     -- array: catalogue before cuality cuts
    #vennData -- array: output of venn_plot def. This are the rejected objects by those quality cuts
    #mask     -- boolen-array: mask (or quality cut) to extract the data alone, without sharing objects with the others
    #N        -- float: sample size
    #outFile  -- string: path to store the data
    #-----------------
    #return:
    #RA, DEC, r-MAGNITUDE
    g, r, z, w1, G, rr = get_mag_decals(catr)
    sampleCat = catr['RA', 'DEC'][vennData[mask]]
    sampleCat['rmag'] = r[vennData[mask]]
    ranSelect = random.sample(range(len(sampleCat)), N)
    ascii.write(sampleCat[ranSelect], outFile, overwrite=True)
    
    return sampleCat[ranSelect]
    
def load_catOLD(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))

    gmag, rmag, zmag, w1mag, Gmag, rr = get_mag_decals(cat0S)
    
    #Cut Fluxes <= 0 to avoid inf and nan in 'flux_to_mag' 
    maskFLUX = (cat0S['FLUX_R'] > 0) & (cat0S['FLUX_G'] > 0) & (cat0S['FLUX_Z'] > 0) & (cat0S['FLUX_W1'] > 0)
    # Require 2+ exposures in grz
    maskNOBS = (cat0S['NOBS_G']>=1) & (cat0S['NOBS_R']>=1) & (cat0S['NOBS_Z']>=1)
    #maskMAG = np.logical_and(rmag>=15, rmag<=20)
    maskMAG = rmag < 20
    maskFRACMASKED = (cat0S['FRACMASKED_G']<0.4) & (cat0S['FRACMASKED_R']<0.4) & (cat0S['FRACMASKED_Z']<0.4)
    maskFRACFLUX = (cat0S['FRACFLUX_G']<5) & (cat0S['FRACFLUX_R']<5) & (cat0S['FRACFLUX_Z']<5)
    maskFRACIN = (cat0S['FRACIN_G']>0.3) & (cat0S['FRACIN_R']>0.3) & (cat0S['FRACIN_Z']>0.3)
    maskFLUX_IVAR = (cat0S['FLUX_IVAR_G']>0) & (cat0S['FLUX_IVAR_R']>0) & (cat0S['FLUX_IVAR_Z']>0)
    mask_gr = np.logical_and(gmag - rmag > -1, gmag - rmag < 4)
    mask_rz = np.logical_and(rmag - zmag > -1, rmag - zmag < 4)
    maskBRIGHTSTAR = np.array([not i for i in cat0S['BRIGHTSTARINBLOB']])
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    print('Lenght DR7 sample within 15 <= r <= 20:', len(cat0S[maskMAG]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskFLUX, maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ('maskFLUX', 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[maskMAG], mask_list[i][maskMAG])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[maskMAG] & mask_rz[maskMAG]
    mask_list_MAG = np.array([maskNOBS[maskMAG], maskFRACMASKED[maskMAG], maskFRACFLUX[maskMAG], maskFRACIN[maskMAG], maskFLUX_IVAR[maskMAG], maskBRIGHTSTAR[maskMAG], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[maskMAG]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & maskMAG & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    if desitarget:
        print('adding desitarget column to catalogue...')
        desitarget,idx, d2d = add_desitarget(cat0S, catT)
        cat0S = Table(cat0S)
        cat0S['desitarget'] = desitarget

    return cat0S[mask], cat0S[maskMAG], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped


def unextinct_fluxes(objects):
    """Calculate unextincted DECam and WISE fluxes

    Args:
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z, 
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2

    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX

    Output type is Table if input is Table, otherwise numpy structured array
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4')]

    result = np.zeros(len(objects), dtype=dtype)

    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']

    return result

def MASK(objects, TYPE):

    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])
    
    primary = np.ones_like(rflux, dtype='?')
    bgs = primary.copy()
    bgs_gaia = primary.copy()
    bgs_nogaia = primary.copy()
    bgs &= (gnobs>=1) & (rnobs>=1) & (znobs>=1)
    bgs &= (gfracmasked<0.4) & (rfracmasked<0.4) & (zfracmasked<0.4)
    bgs &= (gfracflux<5.0) & (rfracflux<5.0) & (zfracflux<5.0)
    bgs &= (gfracin>0.3) & (rfracin>0.3) & (zfracin>0.3)
    bgs &= (gfluxivar>0) & (rfluxivar>0) & (zfluxivar>0)
    bgs &= rflux > gflux * 10**(-1.0/2.5)
    bgs &= rflux < gflux * 10**(4.0/2.5)
    bgs &= zflux > rflux * 10**(-1.0/2.5)
    bgs &= zflux < rflux * 10**(4.0/2.5)
    bgs &= np.array([not i for i in BRIGHTSTARINBLOB])
    
    if TYPE == 'BGS_PREV':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        return bgs
    
    if TYPE == 'BGS_FAINT':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= rflux <= 10**((22.5-19.5)/2.5)
        #print(len(bgs))
        bgs_gaia = bgs & (Grr > 0.6)
        bgs_nogaia = bgs & (gaiagmag == 0)
        bgs = bgs_gaia | bgs_nogaia
        #print(len(bgs))
        
        #print('bgs_gaia:', np.sum(bgs_gaia))
        #print('bgs_nogaia:', np.sum(bgs_nogaia))
        return bgs
    
    if TYPE == 'BGS_BRIGHT':
        bgs &= rflux > 10**((22.5-19.5)/2.5)
        bgs_gaia = bgs & (Grr > 0.6)
        bgs_nogaia = bgs & (gaiagmag == 0)
        bgs = bgs_gaia | bgs_nogaia
        
        #print('bgs_gaia:', np.sum(bgs_gaia))
        #print('bgs_nogaia:', np.sum(bgs_nogaia))
        return bgs
    
    if TYPE == 'BGS_WISE':
        bgs &= rflux > 10**((22.5-20.0)/2.5)
        bgs &= Grr < 0.4
        bgs &= Grr > -1
        bgs &= w1flux*gflux > (zflux*rflux)*10**(-0.2)
        bgs &= w1snr > 5
        return bgs
    

def load_cat_caseA(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = (gnobs>=1) & (rnobs>=1) & (znobs>=1)
    maskMAG = rflux > 10**((22.5-20.0)/2.5)
    maskFRACMASKED = (gfracmasked<0.4) & (rfracmasked<0.4) & (zfracmasked<0.4)
    maskFRACFLUX = (gfracflux<5.0) & (rfracflux<5.0) & (zfracflux<5.0)
    maskFRACIN = (gfracin>0.3) & (rfracin>0.3) & (zfracin>0.3)
    maskFLUX_IVAR = (gfluxivar>0) & (rfluxivar>0) & (zfluxivar>0)
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    print('Lenght DR7 sample within r <= 20:', len(cat0S[maskMAG]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[maskMAG], mask_list[i][maskMAG])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[maskMAG] & mask_rz[maskMAG]
    mask_list_MAG = np.array([maskNOBS[maskMAG], maskFRACMASKED[maskMAG], maskFRACFLUX[maskMAG], maskFRACIN[maskMAG], maskFLUX_IVAR[maskMAG], maskBRIGHTSTAR[maskMAG], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[maskMAG]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & maskMAG & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    cat0S = Table(cat0S)
    cat0S['Grr'] = Grr
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')

    return cat0S[mask], cat0S[maskMAG], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped


def load_cat_caseB(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = rnobs>=1
    maskMAG = rflux > 10**((22.5-20.0)/2.5)
    maskFRACMASKED = rfracmasked<0.4
    maskFRACFLUX = rfracflux<5.0
    maskFRACIN = rfracin>0.3
    maskFLUX_IVAR = rfluxivar>0
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    print('Lenght DR7 sample within r <= 20:', len(cat0S[maskMAG]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[maskMAG], mask_list[i][maskMAG])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[maskMAG] & mask_rz[maskMAG]
    mask_list_MAG = np.array([maskNOBS[maskMAG], maskFRACMASKED[maskMAG], maskFRACFLUX[maskMAG], maskFRACIN[maskMAG], maskFLUX_IVAR[maskMAG], maskBRIGHTSTAR[maskMAG], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[maskMAG]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & maskMAG & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        cat0S = Table(cat0S)
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')
    cat0S['Grr'] = Grr

    return cat0S[mask], cat0S[maskMAG], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped

def load_cat_caseC(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = (gnobs>=1) & (rnobs>=1) & (znobs>=1)
    maskMAG = rflux > 10**((22.5-20.0)/2.5)
    maskFRACMASKED = (gfracmasked<0.4) & (rfracmasked<0.4) & (zfracmasked<0.4)
    maskFRACFLUX = (gfracflux<5.0) & (rfracflux<5.0) & (zfracflux<5.0)
    maskFRACIN = (gfracin>0.3) & (rfracin>0.3) & (zfracin>0.3)
    maskFLUX_IVAR = (gfluxivar>0) & (rfluxivar>0) & (zfluxivar>0)
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    maskGrr = Grr > 0.6
    maskNoGaia = gaiagmag == 0
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    #define the initial mask
    mask_ini1 = maskMAG & maskGrr
    mask_ini2 = maskMAG & maskNoGaia
    mask_ini = mask_ini1 | mask_ini2
    print('Lenght DR7 sample within (r < 20) & (Grr > 0.6) | (gaiagmag == 0):', len(cat0S[mask_ini]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[mask_ini], mask_list[i][mask_ini])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[mask_ini] & mask_rz[mask_ini]
    mask_list_MAG = np.array([maskNOBS[mask_ini], maskFRACMASKED[mask_ini], maskFRACFLUX[mask_ini], maskFRACIN[mask_ini], maskFLUX_IVAR[mask_ini], maskBRIGHTSTAR[mask_ini], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[mask_ini]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & mask_ini & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR 
    #mask1 = mask & maskGrr
    #mask2 = mask & maskNoGaia
    #mask = mask1 | mask2
    
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    cat0S = Table(cat0S)
    cat0S['Grr'] = Grr
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')

    return cat0S[mask], cat0S[mask_ini], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped

def load_cat_caseD(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = rnobs>=1
    maskMAG = rflux > 10**((22.5-20.0)/2.5)
    maskFRACMASKED = rfracmasked<0.4
    maskFRACFLUX = rfracflux<5.0
    maskFRACIN = rfracin>0.3
    maskFLUX_IVAR = rfluxivar>0
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    maskGrr = Grr > 0.6
    maskNoGaia = gaiagmag == 0
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    #define the initial mask
    mask_ini1 = maskMAG & maskGrr
    mask_ini2 = maskMAG & maskNoGaia
    mask_ini = mask_ini1 | mask_ini2
    print('Lenght DR7 sample within (r < 20) & (Grr > 0.6) | (gaiagmag == 0):', len(cat0S[mask_ini]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[mask_ini], mask_list[i][mask_ini])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[mask_ini] & mask_rz[mask_ini]
    mask_list_MAG = np.array([maskNOBS[mask_ini], maskFRACMASKED[mask_ini], maskFRACFLUX[mask_ini], maskFRACIN[mask_ini], maskFLUX_IVAR[mask_ini], maskBRIGHTSTAR[mask_ini], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[mask_ini]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & mask_ini & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR 
    #mask1 = mask & maskGrr
    #mask2 = mask & maskNoGaia
    #mask = mask1 | mask2
    
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    cat0S = Table(cat0S)
    cat0S['Grr'] = Grr
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')

    return cat0S[mask], cat0S[mask_ini], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped

#==================================================================================================



def load_cat_caseCplus(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = (gnobs>=1) & (rnobs>=1) & (znobs>=1)
    maskMAG = rflux > 10**((22.5-20.0)/2.5)
    maskFRACMASKED = (gfracmasked<0.4) & (rfracmasked<0.4) & (zfracmasked<0.4)
    maskFRACFLUX = (gfracflux<5.0) & (rfracflux<5.0) & (zfracflux<5.0)
    maskFRACIN = (gfracin>0.3) & (rfracin>0.3) & (zfracin>0.3)
    maskFLUX_IVAR = (gfluxivar>0) & (rfluxivar>0) & (zfluxivar>0)
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    maskGrr = Grr > 0.6
    maskNoGaia = gaiagmag == 0
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    #define the initial mask
    mask_ini1 = maskMAG & maskBRIGHTSTAR & maskGrr
    mask_ini2 = maskMAG & maskBRIGHTSTAR & maskNoGaia
    mask_ini = mask_ini1 | mask_ini2
    print('Lenght DR7 sample within (r < 20) & (Grr > 0.6) | (gaiagmag == 0):', len(cat0S[mask_ini]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[mask_ini], mask_list[i][mask_ini])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[mask_ini] & mask_rz[mask_ini]
    mask_list_MAG = np.array([maskNOBS[mask_ini], maskFRACMASKED[mask_ini], maskFRACFLUX[mask_ini], maskFRACIN[mask_ini], maskFLUX_IVAR[mask_ini], maskBRIGHTSTAR[mask_ini], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[mask_ini]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & mask_ini & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR 
    #mask1 = mask & maskGrr
    #mask2 = mask & maskNoGaia
    #mask = mask1 | mask2
    
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    cat0S = Table(cat0S)
    cat0S['Grr'] = Grr
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')
    
    return cat0S[mask], cat0S[mask_ini], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped







def load_cat_caseA_SV(catalog, catT, FILE=True, desitarget=True):
    """Process input catalogue from SWEEPS and apply star-galaxy separation cuts

    Parameters
    ----------
    catalog : :class:`array` 
        A array catalog from SWEEPS or TRACTOR
    catT : :class:`array` 
        Adam M. DESI TARGET catalogue with same footprint as ``caatalog``
    FILE : :class:`boolean`, optional, defaults to ``True``
        If ``True``, read ``catalog`` from  fits file. 
        If ``False``, read from defined array 
    desitarget : :class:`boolean`, optional, defaults to ``True``
        If ``True``, match to desitarget catalog and append ``DESI_TARGET`` column
        
    Returns
    -------   
    cat0S[mask] : :class: `astropy table`
        ``catalog`` after aplying all star-galaxy separation cuts
    cat0S[maskMAG] : :class: `astropy table`
        ``catalog`` after aplying the 15 < r-band < 20 cuts
    cat0S : :class: `astropy table`
        ``catalog`` without any cuts
    df_M_in : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (15 < r-band < 20)
    df_M_out : :class: `pandas data frame`
        Matrix of overlaps cuts for the rmag mask (r-band < 15, r-band > 20). A.K.A the drops out.
    mask_list_MAG : :class: `boolean`
        An array of all the cuts for the ``df_M_in`` and ``df_M_out`` matrices
    df_dropped : :class: `pandas data frame`
        Contain the information of the dropped objects by each of the star-galaxy separation cuts for cat0S[maskMAG]. 
        'drop_r': number of drops out  
        'drop_r_per': percentage of drops out for the total of len(cat0S[maskMAG]) 
        'PSF_r_per': percentage of the `PSF` part
        'NOPSF_r_per': percentage of the `no-PSF` parts
    """
    
    # Load DESI target catalog
    columns0 = ['RA', 'DEC', 'TYPE', 'DESI_TARGET', 'BRIGHTSTARINBLOB', 'GAIA_PHOT_G_MEAN_MAG', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'FLUX_G', 
                'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1',
               'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z', 'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
               'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z']
    
    if FILE:
        cat0 = fitsio.read(catalog, columns=columns0)
        print('Lenght full DR7:',len(cat0))
        cat0S = cut(200, 230, -2,5, cat0)
    else:
        cat0S = catalog
        print('Lenght raw catalogue:',len(cat0S))
        
    objects =  cat0S
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    objtype = objects['TYPE']
    release = objects['RELEASE']

    gfluxivar = objects['FLUX_IVAR_G']
    rfluxivar = objects['FLUX_IVAR_R']
    zfluxivar = objects['FLUX_IVAR_Z']

    gnobs = objects['NOBS_G']
    rnobs = objects['NOBS_R']
    znobs = objects['NOBS_Z']

    gfracflux = objects['FRACFLUX_G']
    rfracflux = objects['FRACFLUX_R']
    zfracflux = objects['FRACFLUX_Z']

    gfracmasked = objects['FRACMASKED_G']
    rfracmasked = objects['FRACMASKED_R']
    zfracmasked = objects['FRACMASKED_Z']

    gfracin = objects['FRACIN_G']
    rfracin = objects['FRACIN_R']
    zfracin = objects['FRACIN_Z']
    
    gallmask = objects['ALLMASK_G']
    rallmask = objects['ALLMASK_R']
    zallmask = objects['ALLMASK_Z']

    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    BRIGHTSTARINBLOB = objects['BRIGHTSTARINBLOB']
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    maskNOBS = (gnobs>=1) & (rnobs>=1) & (znobs>=1)
    maskMAG = rflux > 10**((22.5-20.2)/2.5)
    #maskMAG &= rflux <= 10**((22.5-20)/2.5)
    maskFRACMASKED = (gfracmasked<0.4) & (rfracmasked<0.4) & (zfracmasked<0.4)
    maskFRACFLUX = (gfracflux<5.0) & (rfracflux<5.0) & (zfracflux<5.0)
    maskFRACIN = (gfracin>0.3) & (rfracin>0.3) & (zfracin>0.3)
    maskFLUX_IVAR = (gfluxivar>0) & (rfluxivar>0) & (zfluxivar>0)
    mask_gr = np.logical_and(rflux > gflux * 10**(-1.0/2.5), rflux < gflux * 10**(4.0/2.5))
    mask_rz = np.logical_and(zflux > rflux * 10**(-1.0/2.5), zflux < rflux * 10**(4.0/2.5))
    maskBRIGHTSTAR = np.array([not i for i in BRIGHTSTARINBLOB])
    
    def dropped(catraw, MASK):
        ND = len(catraw[[not i for i in MASK]])#len(catraw) - np.sum(MASK) # Number objects dropped
        Per = ND*100/len(catraw) # Percentage of dropped objects from a total in catraw
        if ND == 0:
            PSF_per = 0
            NOPSF_per = 0
        else:
            PSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] == b'PSF ')])*100/ND
            NOPSF_per = len(catraw[[not i for i in MASK] & (catraw['TYPE'] != b'PSF ')])*100/ND
        
        return np.array([ND, Per, PSF_per, NOPSF_per])
    
    print('Lenght DR7 sample within r < 20.2:', len(cat0S[maskMAG]))
    
    #creating Dropped table -----------------
    mask_list = np.array([maskNOBS, maskMAG, maskFRACMASKED, maskFRACFLUX, maskFRACIN, maskFLUX_IVAR, mask_gr, mask_rz, maskBRIGHTSTAR])
    mask_list_names = ( 'maskNOBS', 'maskMAG', 'maskFRACMASKED', 'maskFRACFLUX', 'maskFRACIN', 'maskFLUX_IVAR', 'mask_gr', 'mask_rz', 'maskBRIGHTSTAR')
    Dropped = [[0 for x in range(4)] for y in range(len(mask_list))] 
    from itertools import product
    for i in range(len(mask_list)):
        Dropped[i] = dropped(cat0S[maskMAG], mask_list[i][maskMAG])
    df_dropped = pd.DataFrame(Dropped, index=mask_list_names, columns= ('drop_r', 'drop_r_per', 'PSF_r_per', 'NOPSF_r_per'))
    
    #------------------------------
    
    #creating correlation matrix IN and OUT
    mask_gr_rz = mask_gr[maskMAG] & mask_rz[maskMAG]
    mask_list_MAG = np.array([maskNOBS[maskMAG], maskFRACMASKED[maskMAG], maskFRACFLUX[maskMAG], maskFRACIN[maskMAG], maskFLUX_IVAR[maskMAG], maskBRIGHTSTAR[maskMAG], mask_gr_rz])
    mask_list_MAG_names_in = ('NOBS', 'FRACMASKED', 'FRACFLUX', 'FRACIN', 'FLUX_IVAR', 'BRIGHTSTAR', 'gr-rz')
    mask_list_MAG_names_out = ('NOBS OUT', 'FRACMASKED OUT', 'FRACFLUX OUT', 'FRACIN OUT', 'FLUX_IVAR OUT', 'BRIGHTSTAR OUT', 'gr-rz OUT')

    cattmp = cat0S[maskMAG]
    Matrix_in = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))]
    Matrix_out = [[0 for x in range(len(mask_list_MAG))] for y in range(len(mask_list_MAG))] 
    
    from itertools import product
    for i,j in product(range(len(mask_list_MAG)), range(len(mask_list_MAG))):
        Matrix_in[i][j] = len(cattmp[(mask_list_MAG[i]) & (mask_list_MAG[j])])
        if i == j:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]]])
        else:
            Matrix_out[i][j] = len(cattmp[[not i for i in mask_list_MAG[i]] & (mask_list_MAG[j])])
    #-------------------------------------------
            
    #print('MATRIX header:%s' %(str(mask_list_MAG_names)))

    df_M_in = pd.DataFrame(Matrix_in, index=mask_list_MAG_names_in, columns=mask_list_MAG_names_in)
    df_M_out = pd.DataFrame(Matrix_out, index=mask_list_MAG_names_out, columns=mask_list_MAG_names_in)
    
    mask = maskNOBS & maskMAG & maskFRACMASKED & maskFRACFLUX & maskFRACIN & maskFLUX_IVAR & mask_gr & mask_rz & maskBRIGHTSTAR
    print('Lenght DR7 sample after cuts', len(cat0S[mask]))
    
    cat0S = Table(cat0S)
    cat0S['Grr'] = Grr
    
    if desitarget:
        print('adding desitarget and bgstarget column to catalogue...')
        desitarget, bgstarget, idx, d2d = add_desitarget(cat0S, catT)
        
        cat0S['desitarget'] = desitarget
        cat0S['bgstarget'] = bgstarget
    print('adding Grr column to catalogue...')

    return cat0S[mask], cat0S[maskMAG], cat0S, df_M_in, df_M_out, mask_list_MAG, df_dropped