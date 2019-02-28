import numpy as np
from astropy.table import Table

def _is_row(table):
    """Return True/False if this is a row of a table instead of a full table.
    supports numpy.ndarray, astropy.io.fits.FITS_rec, and astropy.table.Table
    """
    import astropy.io.fits.fitsrec
    import astropy.table.row
    if isinstance(table, (astropy.io.fits.fitsrec.FITS_record, astropy.table.row.Row)) or \
       np.isscalar(table):
        return True
    else:
        return False

def _get_colnames(objects):
    """Simple wrapper to get the column names."""

    # ADM capture the case that a single FITS_REC is passed
    import astropy.io.fits.fitsrec
    if isinstance(objects, astropy.io.fits.fitsrec.FITS_record):
        colnames = objects.__dict__['array'].dtype.names
    else:
        colnames = objects.dtype.names

    return colnames


def _prepare_optical_wise(objects, colnames=None):
    """Process the Legacy Surveys inputs for target selection."""

    if colnames is None:
        colnames = _get_colnames(objects)

    # ADM flag whether we're using northen (BASS/MZLS) or
    # ADM southern (DECaLS) photometry
    #photsys_north = _isonnorthphotsys(objects["PHOTSYS"])
    #photsys_south = ~_isonnorthphotsys(objects["PHOTSYS"])

    # ADM rewrite the fluxes to shift anything on the northern Legacy Surveys
    # ADM system to approximate the southern system
    # ADM turn off shifting the northern photometry to match the southern
    # ADM photometry. The consensus at the May, 2018 DESI collaboration meeting
    # ADM in Tucson was not to do this.
#    wnorth = np.where(photsys_north)
#    if len(wnorth[0]) > 0:
#        gshift, rshift, zshift = shift_photo_north(objects["FLUX_G"][wnorth],
#                                                   objects["FLUX_R"][wnorth],
#                                                   objects["FLUX_Z"][wnorth])
#        objects["FLUX_G"][wnorth] = gshift
#        objects["FLUX_R"][wnorth] = rshift
#        objects["FLUX_Z"][wnorth] = zshift

    # ADM the observed r-band flux (used for F standards and MWS, below)
    # ADM make copies of values that we may reassign due to NaNs
    obs_rflux = objects['FLUX_R']

    # - undo Milky Way extinction
    flux = unextinct_fluxes(objects)

    gflux = flux['GFLUX']
    rflux = flux['RFLUX']
    zflux = flux['ZFLUX']
    w1flux = flux['W1FLUX']
    w2flux = flux['W2FLUX']
    rfiberflux = flux['RFIBERFLUX']
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

    gsnr = objects['FLUX_G'] * np.sqrt(objects['FLUX_IVAR_G'])
    rsnr = objects['FLUX_R'] * np.sqrt(objects['FLUX_IVAR_R'])
    zsnr = objects['FLUX_Z'] * np.sqrt(objects['FLUX_IVAR_Z'])
    w1snr = objects['FLUX_W1'] * np.sqrt(objects['FLUX_IVAR_W1'])
    w2snr = objects['FLUX_W2'] * np.sqrt(objects['FLUX_IVAR_W2'])

    # For BGS target selection
    brightstarinblob = objects['BRIGHTSTARINBLOB']
    
    gaiagmag = objects['GAIA_PHOT_G_MEAN_MAG']
    # For BGS target selection
    Grr = gaiagmag - 22.5 + 2.5*np.log10(objects['FLUX_R'])

    # Delta chi2 between PSF and SIMP morphologies; note the sign....
    dchisq = objects['DCHISQ']
    deltaChi2 = dchisq[..., 0] - dchisq[..., 1]

    # ADM remove handful of NaN values from DCHISQ values and make them unselectable
    w = np.where(deltaChi2 != deltaChi2)
    # ADM this is to catch the single-object case for unit tests
    if len(w[0]) > 0:
        deltaChi2[w] = -1e6

    return (obs_rflux, gflux, rflux, zflux,
            w1flux, w2flux, rfiberflux, objtype, release, gfluxivar, rfluxivar, zfluxivar,
            gnobs, rnobs, znobs, gfracflux, rfracflux, zfracflux,
            gfracmasked, rfracmasked, zfracmasked,
            gfracin, rfracin, zfracin, gallmask, rallmask, zallmask,
            gsnr, rsnr, zsnr, w1snr, w2snr, dchisq, deltaChi2, brightstarinblob, gaiagmag, Grr)


def unextinct_fluxes(objects):
    """Calculate unextincted DECam and WISE fluxes.

    Args:
        objects: array or Table with columns FLUX_G, FLUX_R, FLUX_Z,
            MW_TRANSMISSION_G, MW_TRANSMISSION_R, MW_TRANSMISSION_Z,
            FLUX_W1, FLUX_W2, MW_TRANSMISSION_W1, MW_TRANSMISSION_W2

    Returns:
        array or Table with columns GFLUX, RFLUX, ZFLUX, W1FLUX, W2FLUX

    Output type is Table if input is Table, otherwise numpy structured array
    """
    dtype = [('GFLUX', 'f4'), ('RFLUX', 'f4'), ('ZFLUX', 'f4'),
             ('W1FLUX', 'f4'), ('W2FLUX', 'f4'), ('RFIBERFLUX', 'f4')]
    if _is_row(objects):
        result = np.zeros(1, dtype=dtype)[0]
    else:
        result = np.zeros(len(objects), dtype=dtype)

    result['GFLUX'] = objects['FLUX_G'] / objects['MW_TRANSMISSION_G']
    result['RFLUX'] = objects['FLUX_R'] / objects['MW_TRANSMISSION_R']
    result['ZFLUX'] = objects['FLUX_Z'] / objects['MW_TRANSMISSION_Z']
    result['W1FLUX'] = objects['FLUX_W1'] / objects['MW_TRANSMISSION_W1']
    result['W2FLUX'] = objects['FLUX_W2'] / objects['MW_TRANSMISSION_W2']
    result['RFIBERFLUX'] = objects['FIBERFLUX_R'] / objects['MW_TRANSMISSION_R']

    if isinstance(objects, Table):
        return Table(result)
    else:
        return result
    

def _psflike(psftype):
    """ If the object is PSF """
    # ADM explicitly checking for NoneType. I can't see why we'd ever want to
    # ADM run this test on empty information. In the past we have had bugs where
    # ADM we forgot to pass objtype=objtype in, e.g., isSTD
    if psftype is None:
        raise ValueError("NoneType submitted to _psfflike function")

    psftype = np.asarray(psftype)
    # ADM in Python3 these string literals become byte-like
    # ADM so to retain Python2 compatibility we need to check
    # ADM against both bytes and unicode
    # ADM, also 'PSF' for astropy.io.fits; 'PSF ' for fitsio (sigh)
    psflike = ((psftype == 'PSF') | (psftype == b'PSF') |
               (psftype == 'PSF ') | (psftype == b'PSF '))
    return psflike
