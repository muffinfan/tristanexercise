from scipy.special import erfinv
import numpy as np
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

def integrated_eta_distribution(eIn,eOut1,eOut2): 
    return 2*np.sqrt(2)*(erfinv(1-2*eOut1/eIn)-erfinv(1-2*eOut2/eIn))

def calculate_cs_response(wcc,rpx,thr,eBinsIn,eBinsOut):
    resp = np.zeros((eBinsIn.nbin,eBinsOut.nbin))
    
    # index of lowest edge above threshold 
    jmin = int(np.argwhere(eBinsOut.edges>=thr)[0])
    
    # loop over input energy bins
    for i in range(0,eBinsIn.nbin):
        
        # calculate response matrix elements
        # - exclude first few and last element due to divergence
        # - take center energy for input bin (averaging would be more difficult)
        for j in range(jmin,i):
            resp[i,j] = wcc/rpx * integrated_eta_distribution(eBinsIn.centers[i], eBinsOut.edges[j], eBinsOut.edges[j+1])
        
        # set first and last element such that 
        norm = 1-resp[i,int(i/2):-1].sum()
        resp[i,0] = 0#norm/2
        resp[i,i] = norm
    
    return resp

