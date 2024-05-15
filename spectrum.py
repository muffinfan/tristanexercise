import numpy as np
from scipy.integrate import quad

#-------------------------------------------------------------------------------------------------
# D.H. Wilkinson, Small terms in the beta-decay spectrum of tritium, Nucl. Phys. A 526 (1991) 131.
# or https://iopscience.iop.org/article/10.1088/0034-4885/71/8/086201

M_ELECTRON = 510998.95        # electron mass, eV
ALPHA      = 7.2973525698e-3  # fine-structure constant
ENDPOINT   = 18575            # endpoint energy, eV
A_CONST    = 1.002037         # constant, see reference
B_CONST    = 0.001427         # constant, see reference
Z_DAUGHTER = 2                # daughter nucleus charge

def fermi(ekin,Z=Z_DAUGHTER):
    """ 
    Fermi function (unscreened coulomb field)
    
    Parameters:
      Z : Atomic charge of daughter nucleus
      E : Kinetic energy of electron in eV (>0)
    
    Returns:
      Fermi correction factor    
    """
    
    # total energy
    etot = ekin + M_ELECTRON
    
    # relativistic beta
    beta = np.sqrt(1 - (M_ELECTRON/(etot))**2)
    
    # Sommerfeld parameter eta
    eta = ALPHA*Z/beta
    
    # non-relativistic fermi correction    
    fermicorr = 2*np.pi*eta/(1-np.exp(-2*np.pi*eta))
    
    # relativistic approximation
    fermicorr = fermicorr * (A_CONST-B_CONST*beta)
    
    return fermicorr


def diffspec_base(ekin, mnu=0):
    
    '''
    Differential tritium spectrum with 
    fermi correction (relativistic approximation).
    
      ekin : Kinectic energy of electron in eV
      mnu  : Neutrino mass in eV 
    '''
    
    # convert types for convenience
    if type(ekin)==float or type(ekin)==int:
        ekin = np.array([ekin])
    
    # initialize output spectrum array
    spec = np.zeros(len(ekin))
    
    # non-zero region
    nz = (ekin>0) & (ekin<(ENDPOINT-mnu))

    # electron total energy
    ee = (ekin[nz]+M_ELECTRON)
    
    # neutrino total energy
    enu = ENDPOINT-ekin[nz]
    
    # electron momentum
    pe = np.sqrt((ekin[nz]+M_ELECTRON)**2 - M_ELECTRON**2)
    
    # neutrino momentum
    pnu = np.sqrt((ENDPOINT-ekin[nz])**2-mnu**2)

    # fermicorrection
    fermicorr = fermi(ekin[nz])

    # the spectrum
    spec[nz] = fermicorr*ee*enu*pe*pnu*np.heaviside(ekin[nz],1)
    
    return spec


def diffspec_mixed(ekin, mActive=0, mSterile=0, sin2theta=0):
    
    # active neutrino shape
    active = (1-sin2theta)*diffspec_base(ekin,mnu=mActive)
    
    # sterile neutrino shape
    sterile = (sin2theta)*diffspec_base(ekin,mnu=mSterile)
    
    # sum according to mixing amplitude
    spec = active + sterile
    
    return spec


def integrate_over_bins(func,binEdges):
    '''
    Integrate 1D function over the given bins.
    
    Parameters:
      func:     The function / probability density
      binEdges: 1D array of bin edges
    
    Returns: The binned spectrum as 1D np.array
    '''    
    
    nBins = len(binEdges)-1
    binnedSpec = np.zeros(nBins)
    
    for j in range(nBins):
        binnedSpec[j] = quad(func,binEdges[j], binEdges[j+1])[0]
    
    return binnedSpec


def diffspec_mixed_binned(energyBinEdges, mSterile=0, sin2theta=0, fast=False):
    
    # define function for bin integration
    spec = lambda e: diffspec_mixed(e, mSterile=mSterile, sin2theta=sin2theta)
    
    # integrate over bins
    if fast:
        energyBinCenters = (eBinEdges[:-1]+eBinEdges[1:])/2
        binnedSpec = integrate_bins_fast(spec,energyBinCenters)
    else:
        binnedSpec = integrate_over_bins(spec,energyBinEdges)
    
    return binnedSpec