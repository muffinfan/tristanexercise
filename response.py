from binning import binning
import numpy as np

def calculate_braodening_response(eBins,sigma):
    """
    Calculate the response matrix for a simple broadening
    taking into account the bin width.
    Method: Integral of gaussian in target bin averaged 
            over all possible energies in input bin.
    
    Parameters:
        - eBins: Binning object
        - sigma: Width of the gaussian
    """
    
    if sigma==0:
        return np.eye(eBins.nbin)

    nbin = eBins.nbin
    dv   = eBins.dv
    
    # calculate one extended row
    x = np.arange(-nbin,nbin+1)*dv
    i = np.arange(len(x)-1)
    y = gaussian_smearing_matrix_element(x,i,nbin,sigma)
    
    # each row of the matrix is the same only shifted
    matrix = np.zeros((nbin,nbin))
    for i,row in enumerate(matrix):
        row[:] = y[nbin-i:2*nbin-i]
    
    return matrix