import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
#import fitsio
#import pandas as pd
#from astropy.io import ascii
#import sys, os, time, argparse, glob
import warnings; warnings.simplefilter('ignore')
from astropy.utils.data import download_file  #import file from URL

#from match_coord import search_around, scatter_plot, match_coord
import healpy as hp
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy import units as u

def relative_density_plot(d_ra, d_dec, d2d, search_radius, ref_density, nbins=101, return_res=False, show=True):

    bins = np.linspace(-search_radius, search_radius, nbins)
    bin_spacing = bins[1] - bins[0]
    bincenter = (bins[1:]+bins[:-1])/2
    mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
    mesh_d2d = np.sqrt(mesh_ra**2 + mesh_dec**2)
    mask = (d2d>2.)
    #taking the 2d histogram and divide by the area of each bin to get the density
    density, _, _ = np.histogram2d(d_ra[mask], d_dec[mask], bins=bins)/(bin_spacing**2)
    #ignoring data outside the circle with radius='search radius'
    mask = mesh_d2d >= bins.max()-bin_spacing
    density[mask] = np.nan
    density_ratio = density/ref_density
    plt.figure(figsize=(8, 8))
    plt.imshow(density_ratio.transpose()-1, origin='lower', aspect='equal',
               cmap='seismic', extent=bins.max()*np.array([-1, 1, -1, 1]), vmin=-3, vmax=3)
    plt.colorbar(fraction=0.046, pad=0.04)
    if show:
        plt.show()

    if return_res:
        return bins, mesh_d2d, density_ratio


def circular_mask_radii_func(MAG):
    '''
    Define mask radius as a function of the magnitude

    Inputs
    ------
    magnitude: Magnitude in (array);

    Output
    ------
    radii: mask radii (array)
    '''
    #add the data to interpolate in format ['MAG', DISTANCE (arcsec)]
    #For LSLGA
    #x, y = np.transpose([[9.5,100], [10.5, 90], [11.5, 80], [12.5, 70], [13.5, 60], [14.5, 50], [15.5, 35], [16.5, 25], [17.5, 15], [17.5, 10], [18, 8]])
    #For TWOMASS
    x, y = np.transpose([[9.5,100], [10.5, 80], [11.5, 70], [12.5, 50], [13.5, 30], [14.5, 20], [15.5, 10], [16.5, 5]])
    circular_mask_radii_func = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))

    MAG = np.array(MAG)
    radii = np.zeros(len(MAG))
    mask = MAG>10 # set maximum mask radius
    if np.sum(mask)>0:
        radii[mask] = circular_mask_radii_func(MAG[mask]) #pa*(w1_ab[mask])**3 + pb*(w1_ab[mask])**2 +pc*(w1_ab[mask]) + pd
    if np.sum(~mask)>0:
        radii[~mask] = 100

    # mask radius in arcsec
    return radii


def search_around(ra1, dec1, ra2, dec2, search_radius=1., verbose=True):
    '''
    Using the astropy.coordinates.search_around_sky module to find all pairs within
    some search radius.
    Inputs:
    RA and Dec of two catalogs;
    search_radius (arcsec);
    Outputs:
        idx1, idx2: indices of matched objects in the two catalogs;
        d2d: angular distances (arcsec);
        d_ra, d_dec: the differences in RA and Dec (arcsec);
    '''

    # protect the global variables from being changed by np.sort
    ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])

    # Matching catalogs
    sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')
    idx1, idx2, d2d, d3d = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
    if verbose:
        print('%d nearby objects ~ %g %%'%(len(idx1), 100*len(idx1)/len(ra2)))

    # convert distances to numpy array in arcsec
    d2d = np.array(d2d.to(u.arcsec))


    d_ra = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
    d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask = d_ra > 180*3600
    d_ra[mask] = d_ra[mask] - 360.*3600
    mask = d_ra < -180*3600
    d_ra[mask] = d_ra[mask] + 360.*3600
    d_ra = d_ra * np.cos(dec1[idx1]/180*np.pi)
    ##########################################

    return idx1, idx2, d2d, d_ra, d_dec

def overdensity(cat, star, nameMag, slitw, density=False, magbins=(8,14,4)):
    '''
    Get scatter and density plots of objects of cat1 around objects of cat2 within a search radius in arcsec.

    Inputs
    ------
    cat: (array) catalogue 1;
    star: (array) catalogue 2;
    nameMag: (string) label of magnitude in catalogue 2;
    slitw: (float, integer) slit widht;
    density: (boolean) True to get the density as function of distance (arcsec) within shells;
    magbins: (integers) format to separate the magnitude bins in cat2 (min, max, number bins);

    Output
    ------
    (distance (arcsec), density) if density=True
    '''
    # define the slit width for estimating the overdensity off diffraction spikes
    slit_width = slitw
    search_radius = 240.

    # Paramater for estimating the overdensities
    annulus_min = 1
    annulus_max = 240.

    ra2 = star['RA']
    dec2 = star['DEC']
    ra1 = cat['RA']
    dec1 = cat['DEC']

    if density:

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2, dec2, ra1, dec1,
                                                 search_radius=search_radius)
        density = []
        shells = np.linspace(1, search_radius, search_radius)
        for i in range(len(shells)-1):

            ntot_annulus = np.sum((d2d>shells[i]) & (d2d<shells[i+1]))
            density_annulus = ntot_annulus/(np.pi*(shells[i+1]**2 - shells[i]**2))
            bincenter = (shells[i]+shells[i+1])/2

            density.append([bincenter, density_annulus])

        density = np.array(density).transpose()
        plt.figure(figsize=(12, 8))
        plt.semilogy(density[0], density[1])
        plt.xlabel(r'r(arcsec)')
        plt.ylabel(r'N/($\pi r^2$)')
        plt.grid()
        plt.show()

        return density


    mag_bins = np.linspace(magbins[0], magbins[1], magbins[2])

    for index in range(len(mag_bins)):

        if index==0:
            mask_star = (star[nameMag]<mag_bins[index])
            title = '{} < {:.2f}'.format(nameMag,mag_bins[0], np.sum(mask_star))
        else:
            mask_star = (star[nameMag]>mag_bins[index-1]) & (star[nameMag]<mag_bins[index])
            title = '{:.2f} < {} < {:.2f}'.format(mag_bins[index-1], nameMag, mag_bins[index], np.sum(mask_star))

        print(title)

        #get the mask radii from the mean magnitude
        mag_mean = np.mean(star[nameMag][mask_star])
        mask_radius = circular_mask_radii_func([mag_mean])[0]

        idx2, idx1, d2d, d_ra, d_dec = search_around(ra2[mask_star], dec2[mask_star], ra1, dec1,
                                                 search_radius=search_radius)

        print('%d sources ~%g %% ' %(len(ra2[mask_star]),100*len(ra2[mask_star])/len(ra2)))

        markersize = np.max([0.01, np.min([10, 0.3*100000/len(idx2)])])
        axis = [-search_radius*1.05, search_radius*1.05, -search_radius*1.05, search_radius*1.05]
        axScatter = scatter_plot(d_ra, d_dec, markersize=markersize, alpha=0.4, figsize=6.5, axis=axis, title=title)

        ntot_annulus = np.sum((d2d>annulus_min) & (d2d<annulus_max))
        density_annulus = ntot_annulus/(np.pi*(annulus_max**2 - annulus_min**2))

        bins, mesh_d2d, density_ratio = relative_density_plot(d_ra, d_dec, d2d, search_radius,
                                                              ref_density=density_annulus, return_res=True,
                                                              show=False, nbins=101)

        plt.axvline(slit_width)
        plt.axvline(-slit_width)
        plt.axhline(slit_width)
        plt.axhline(-slit_width)    
        angle_array = np.linspace(0, 2*np.pi, 240)
        x = mask_radius * np.sin(angle_array)
        y = mask_radius* np.cos(angle_array)
        plt.plot(x, y, 'k', lw=1)
        plt.title(title)
        plt.show()

        #bincenter = (bins[1:]+bins[:-1])/2
        #mesh_ra, mesh_dec = np.meshgrid(bincenter, bincenter)
        #mask = ((mesh_ra<slit_width) & (mesh_ra>-slit_width)) | ((mesh_dec<slit_width) & (mesh_dec>-slit_width))
        #plt.figure(figsize=(7, 5))
        #plt.plot(mesh_d2d[mask].flatten(), density_ratio[mask].flatten()-1, '.', markersize=1)
        #plt.axvline(mask_radius, lw=1, color='k')
        #plt.xlabel('distance (arcsec)')
        #plt.ylabel('fractional overdensity')
        #plt.grid(alpha=0.5)
        #plt.ylim(ymax=5)
        #plt.show()


def extract_annulus(cat, star, nameMag, plot2d='circle', mag=(8,10), annulus=(35,40)):
    '''
    Get RA, DEC and r-magnitude of selected sample

    Inputs
    ------
    cat: (array) catalogue 1;
    star: (array) catalogue 2;
    nameMag: (string) label of magnitude in catalogue 2;
    plot2d: (string) (='circle', 'square') select objects in concentric circles or concentric squares shells from the 2d density histogram
    mag: (integers) magnitude limits from catalogue 1 in format (min, max)
    annulus: (integer) (inner, outter) circle radius or square lenght in arcsec

    Output
    ------
    RA, DEC, magnitude

    '''
    search_radius = 240.
    #Paramater for estimating the overdensities
    annulus_min = 1
    annulus_max = 240.

    ra2 = star['RA']
    dec2 = star['DEC']
    ra1 = cat['RA']
    dec1 = cat['DEC']

    mask_star = (star[nameMag]>mag[0]) & (star[nameMag]<mag[1])

    idx2, idx1, d2d, d_ra, d_dec = search_around(ra2[mask_star], dec2[mask_star], ra1, dec1,
                                                 search_radius=search_radius)

    print('%d sources ~%g %% ' %(len(ra2[mask_star]),100*len(ra2[mask_star])/len(ra2)))

    catN = cat[idx1]
    catN['d2d'] = d2d
    catN['d_ra'] = d_ra
    catN['d_dec'] = d_dec

    ntot_annulus = np.sum((d2d>annulus_min) & (d2d<annulus_max))
    density_annulus = ntot_annulus/(np.pi*(annulus_max**2 - annulus_min**2))

    bins, mesh_d2d, density_ratio = relative_density_plot(d_ra, d_dec, d2d, search_radius,
                                                              ref_density=density_annulus, return_res=True,
                                                              show=False, nbins=101)
    import matplotlib.patches as patches

    if plot2d == 'circle':

        mask = catN['d2d'] > annulus[0]
        mask &= catN['d2d'] < annulus[1]

        print('Inside shell:',len(catN[mask]))

        angle_array = np.linspace(0, 2*np.pi, 240)
        x1 = annulus[0] * np.sin(angle_array)
        y1 = annulus[0] * np.cos(angle_array)
        x2 = annulus[1] * np.sin(angle_array)
        y2 = annulus[1] * np.cos(angle_array)
        plt.plot(x1, y1, 'k', lw=1)
        plt.plot(x2, y2, 'k', lw=1)
        plt.show()

    if plot2d == 'square':

        mask_in = ((catN['d_ra'] < annulus[0]) & (catN['d_ra'] > -annulus[0])) & ( (catN['d_dec'] < annulus[0]) & (catN['d_dec'] > -annulus[0]))
        mask_out = ((catN['d_ra'] < annulus[1]) & (catN['d_ra'] > -annulus[1])) & ( (catN['d_dec'] < annulus[1]) & (catN['d_dec'] > -annulus[1]))
        mask = (~mask_in) & (mask_out)
        #print(np.sum(mask_in), np.sum(mask_out))

        print('Inside shell:',len(catN[mask]))

        plt.axvline(annulus[0], -annulus[0]/2, annulus[0]/2, c='k')
        plt.axvline(-annulus[0], -annulus[0]/2, annulus[0]/2, c='k')

        plt.axvline(annulus[1], -annulus[1]/2, annulus[1]/2, c='k')
        plt.axvline(-annulus[1], -annulus[1]/2, annulus[1]/2, c='k')

        plt.axhline(annulus[0], -annulus[0]/2, annulus[0]/2, c='k')
        plt.axhline(-annulus[0], -annulus[0]/2, annulus[0]/2, c='k')

        plt.axhline(annulus[1], -annulus[1]/2, annulus[1]/2, c='k')
        plt.axhline(-annulus[1], -annulus[1]/2, annulus[1]/2, c='k')

        plt.show()

    g, r, z, w1, G, rr = get_mag_decals(catN[mask])

    return catN[mask]['RA'], catN[mask]['DEC'], r


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

def scatter_plot(d_ra, d_dec, markersize=1, alpha=1, figsize=8, axis=None, title='', show=True,
    xlabel='RA2 - RA1 (arcsec)', ylabel=('DEC2 - DEC1 (arcsec)')):
    '''
    INPUTS:
     d_ra, d_dec (arcsec): array of RA and Dec difference in arcsec
     (optional): dec (degrees): if specificied, d_ra's are plotted in actual angles

    OUTPUTS:
     axScatter: scatter-histogram plot
    '''

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.85
    bottom, height = 0.1, 0.85

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom, width, 0.3]
    rect_histy = [left, bottom, 0.3, height]

    # start with a rectangular Figure
    plt.figure(figsize=(figsize, figsize))

    axScatter = plt.axes(rect_scatter)
    axScatter.set_title(title)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axScatter.plot(d_ra, d_dec, 'k.', markersize=markersize, alpha=alpha)

    axHistx.hist(d_ra, bins=100, histtype='step', color='r', linewidth=2)
    axHisty.hist(d_dec, bins=100, histtype='step', color='r', linewidth=2, orientation='horizontal')

    if axis is None:
        axHistx.set_xlim(axScatter.get_xlim())
        axHisty.set_ylim(axScatter.get_ylim())
    else:
        axHistx.set_xlim(axis[:2])
        axHisty.set_ylim(axis[2:])
        axScatter.set_xlim(axis[:2])
        axScatter.set_ylim(axis[2:])

    axHistx.axis('off')
    axHisty.axis('off')

    axScatter.axhline(0, color='r', linestyle='--', linewidth=1.2)
    axScatter.axvline(0, color='r', linestyle='--', linewidth=1.2)
    axScatter.set_xlabel(xlabel)
    axScatter.set_ylabel(ylabel)

    if show==True:
        plt.show()
    else:
          return axScatter


def make_cutout_table(ra_in, dec_in, other=None, othername=None, table=(2,7), compare=False,
                      scale_unit='pixscale', boxsize=30, scale=0.262, layer='decals-dr7', layer2=None, savefile=None):
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
    size = int(round(boxsize/scale))
    print('pixels:',size)

    if isinstance(ra_in, float):
        de_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%g&dec=%g&%s=%g&layer=%s&size=%g' % (ra_in,dec_in, scale_unit, scale, layer, size)
        img = plt.imread(download_file(de_cutout_url,cache=True,show_progress=False,timeout=120))

        fig = plt.figure(figsize=(10,10))
        plt.imshow(img)
        if other != None:
            plt.text(0.1,0.9,'%s=%.1f'%(othername,other),fontsize=14,color='white')

        return


    for i in range(N):
        de_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%g&dec=%g&%s=%g&layer=%s&size=%g' % (ra_in[i],dec_in[i], scale_unit, scale, layer, size)
        img = plt.imread(download_file(de_cutout_url,cache=True,show_progress=False,timeout=120))
        de_img.append(img)

        if compare:
            wi_cutout_url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra=%g&dec=%g&%s=%g&layer=%s&size=%g' % (ra_in[i],dec_in[i], scale_unit, scale, layer2, size)
            img = plt.imread(download_file(wi_cutout_url,cache=True,show_progress=False,timeout=120))
            wi_img.append(img)

    fig = plt.figure(figsize=(4*table[1],4*table[0]))

    for i in range(len(de_img)):
        ax = fig.add_subplot(table[0],table[1],i+1)
        ax.imshow(de_img[i])
        #ax.xaxis.set_major_formatter(NullFormatter())
        #ax.yaxis.set_major_formatter(NullFormatter())
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
            if other[i] != None:
                ax.text(0.1,0.9,'%s=%.1f'%(othername,other[i]),transform=ax.transAxes,fontsize=14,color='white')

        plt.subplots_adjust(wspace=0.07, hspace=0.07)

    if savefile != None:
        fig.savefig(savefile +'.png')
        fig.savefig(savefile +'.pdf')


def make_plot_zoom(cat, BGAL, veto, ra, dec, m, TWOMASS=True):
    """
    Produces the ellipses of a specific galaxy (if exists) from two different vetoes: TWOMASS or LSLGA.

    Keyword arguments:

    """
    from matplotlib.patches import Ellipse
    #print not mask[0]

    if TWOMASS:
        RA, DEC, major, minor, angle = twomass_fit(BGAL)
        l0 = BGAL['Jpa']
        j_ext = BGAL['J_ext']
    else:
        RA, DEC, major, minor, angle = LSLGA_fit(BGAL)
        l0 = BGAL['PA']
        j_ext = BGAL['MAG']

    x = cat['RA'][veto]
    y = cat['DEC'][veto]
    h = RA
    z = DEC

    index = min(range(len(h)), key=lambda i: abs(h[i]-ra))
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, marker='.', color='royalblue')
    plt.scatter(cat['RA'][~veto], cat['DEC'][~veto], marker='.', color='g')
    plt.scatter(h, z, marker='*', color='r')
    if TWOMASS:
        plt.title(r'$RA = %f$, $DEC = %f$, $J_{ext}=%3f$, $angle = %d$'%(h[index], z[index], j_ext[index], l0[index]))
    else:
        plt.title(r'$RA = %f$, $DEC = %f$, $MAG=%3f$, $angle = %d$'%(h[index], z[index], j_ext[index], l0[index]))

    plt.xlim(ra-m*major[index], ra+m*major[index])
    plt.ylim(dec-m*major[index], dec+m*major[index])

    ax = plt.gca()
    ellipse1 = Ellipse((h[index], z[index]), width=2*major[index], height=2*minor[index], angle=angle[index],
                       edgecolor='r', fc='None', lw=1)

    ax.add_patch(ellipse1)

    ax.legend([r'$a =%f\,(arcsec)$'%(major[index]*3600)], loc = 'upper right')
    ax.invert_xaxis()
    plt.show()


def make_plot_zoom_comp(cat, BGAL1, BGAL2, veto1, veto2, ra, dec, m):
    """
    Produces the ellipses of a specific galaxy (if exists)for two different vetoes: TWOMASS and LSLGA.

    Keyword arguments:

    """
    from matplotlib.patches import Ellipse

    RA1, DEC1, major1, minor1, angle1 = twomass_fit(BGAL1)
    l01 = BGAL1['Jpa']
    j_ext1 = BGAL1['J_ext']

    RA2, DEC2, major2, minor2, angle2 = LSLGA_fit(BGAL2)
    l02 = BGAL2['PA']
    j_ext2 = BGAL2['MAG']

    #vetoes in common
    x = cat['RA'][(veto1) & (veto2)]
    y = cat['DEC'][(veto1) & (veto2)]

    h1 = RA1
    z1 = DEC1

    h2 = RA2
    z2 = DEC2

    index1 = min(range(len(h1)), key=lambda i: abs(h1[i]-ra))
    index2 = min(range(len(h2)), key=lambda i: abs(h2[i]-ra))
    plt.figure(figsize=(10,10))
    plt.scatter(x, y, marker='.', color='orange')
    plt.scatter(cat['RA'][(~veto1) & (~veto2)], cat['DEC'][(~veto1) & (~veto2)], marker='.', color='g')
    plt.scatter(cat['RA'][(~veto1) & (veto2)], cat['DEC'][(~veto1) & (veto2)], marker='.', color='blue')
    plt.scatter(cat['RA'][(veto1) & (~veto2)], cat['DEC'][(veto1) & (~veto2)], marker='.', color='blue')
    plt.scatter(h1, z1, marker='*', color='r')
    plt.scatter(h2, z2, marker='+', color='black')

    print('TWOMASS: RA = %2.5g, DEC = %2.5g, J_{ext}=%2.5g, angle = %d'%(h1[index1], z1[index1],
                                                                          j_ext1[index1], l01[index1]))
    print('LSLGA: RA = %2.5g, DEC = %2.5g, MAG=%2.5g, angle = %d'%(h2[index2], z2[index2],
                                                                    j_ext2[index2], l02[index2]))
    #print(title1)
    #print(title2)
    plt.xlim(ra-m*major1[index1], ra+m*major1[index1])
    plt.ylim(dec-m*major1[index1], dec+m*major1[index1])

    ax = plt.gca()
    ellipse1 = Ellipse((h1[index1], z1[index1]), width=2*major1[index1], height=2*minor1[index1], angle=angle1[index1],
                       edgecolor='r', fc='None', lw=1)

    ellipse2 = Ellipse((h2[index2], z2[index2]), width=2*major2[index2], height=2*minor2[index2], angle=angle2[index2],
                       edgecolor='r', fc='None', lw=1, ls='--')
    #ellipse3 = Ellipse((h[index], z[index]), width=2*a_best_fit[index], height=2*b_best_fit[index], angle=l[index], edgecolor='black', fc='None', lw=1)
    #ellipse2 = Ellipse((h[index], z[index]), width=2*a_r_fe[index], height=2*b_r_fe[index], angle=l[index], edgecolor='blue', fc='None', lw=1)

    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    #ax.add_patch(ellipse3)
    ax.legend([r'TWOMASS: $a =%2.4g\,(arcsec)$'%(major1[index1]*3600), r'LSLGA: $a =%2.4g\,(arcsec)$'%(major2[index2]*3600)], loc = 'upper right')
    ax.invert_xaxis()
    plt.show()
