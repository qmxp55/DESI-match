
def make_cutout_table(ra_in, dec_in, other=None, othername=None, table=(2,7), compare=False, scale_unit='pixscale', scale=0.25, layer='decals-dr7', layer2=None, savefile=None):
    """
    Produces a table comparing LEGACY survey images.
    To see the different layers see: https://github.com/yymao/decals-image-list-tool/blob/master/index.html
    
    Keyword arguments:
    ra_in            -- array-like: RA positions in degrees 
    dec_in           -- array-like: DEC positions in degrees 
    other            -- array-like: any other parameter, e.g. magnitude
    othername        -- string: name or label of the 'other' parameter
    table            -- 2D integer array: arrange of the output images in (rows * columns) form
    compare          -- boolean: True if want to get output images comparing with other layer (currently in test mode)
    scale_unit       -- striing: pixel scale default
    scale            -- float: 1:180 arcsec
    layer            -- string: type of layer from legacysurvey. e.g:
                            ls-dr67:Legacy Survey DR6+7
                            decals-dr7:DECaLS DR7
                            decals-dr7-model:DECaLS DR7 Model
                            decals-dr7-resid:DECaLS DR7 Residual
                            mzls+bass-dr6:MzLS+BASS DR6
                            mzls+bass-dr6-model:MzLS+BASS DR6 Model
                            mzls+bass-dr6-resid:MzLS+BASS DR6 Residual
    layer2            -- string: second layer to compare if 'compare=True'
    savefile          -- string: path of output image and pdf if any
    """    
    de_img = []
    wi_img = []
    N = table[0]*table[1]
    
    for i in range(N):
        de_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%g&dec=%g&%s=%g&layer=%s&size=180' % (ra_in[i],dec_in[i], scale_unit, scale, layer)
        img = plt.imread(download_file(de_cutout_url,cache=True,show_progress=False,timeout=120))
        de_img.append(img)
        
        if compare:
            wi_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%g&dec=%g&%s=%g&layer=%s&size=180' % (ra_in[i],dec_in[i], scale_unit, scale, layer2)
            img = plt.imread(download_file(wi_cutout_url,cache=True,show_progress=False,timeout=120))
            wi_img.append(img)
            
    fig = plt.figure(figsize=(4*table[1],4*table[0]))

    for i in range(len(de_img)):
        ax = fig.add_subplot(table[0],table[1],i+1)
        ax.imshow(de_img[i])
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        if other[i] != None:
            ax.text(0.1,0.9,'%s=%.1f'%(othername,other[i]),transform=ax.transAxes,fontsize=14,color='white')

    plt.subplots_adjust(wspace=0.07, hspace=0.07)
    
    if compare:
        fig = plt.figure(figsize=(4*table[1],4*table[0]))
        for i in range(len(wi_img)):
            ax = fig.add_subplot(table[0],table[1],i+1)
            ax.imshow(wi_img[i])
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.text(0.1,0.9,'r=%.1f'%(mag[i]),transform=ax.transAxes,fontsize=14,color='white')

        plt.subplots_adjust(wspace=0.07, hspace=0.07)
    
    if savefile != None:
        fig.savefig(savefile +'.png')
        fig.savefig(savefile +'.pdf')


import numpy as np
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')
from astropy.utils.data import download_file  #import file from URL
from matplotlib.ticker import NullFormatter

if __name__ == "__main__":

    file = np.loadtxt('FRACFLUX_123.dat', skiprows=1)
    file_out = 'images_test/test'

    RA = file[:,0]
    DEC = file[:,1]
    MAG = file[:,2]

    n = 7   #rows
    m = 5   #columns
    table = (n,m)

    make_cutout_table(RA, DEC, MAG, 'other', table, scale=0.5, layer='decals-dr7', savefile=file_out)