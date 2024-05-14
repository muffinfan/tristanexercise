import numpy as np
from typing import Tuple
from numba import njit

#-------------------------------------------------------------------------
# Binning class

class binning:
    """
    A class to store binning information of a histogram.
    
    Stored class variables:
     - vmin:    left edge of first bin
     - vmax:    right edge of right bin
     - nbin:    number of bins
     - edges:   1d-array of bin edge positions, size nbin+1
     - centers: 1d-array of bin centers, size nbin
     - dv:      bin width
    """
    def __init__(self,vmin,vmax,nbin):
        """
        Parameters
         - vmin: left edge of first bin
         - vmax: right edge of right bin
         - nbin: number of bins
        """
        self.vmin       = vmin
        self.vmax       = vmax
        self.nbin       = nbin
        self.edges      = np.linspace(vmin,vmax,nbin+1)
        self.centers    = 0.5*(self.edges[1:]+self.edges[:-1])
        self.dv         = (vmax-vmin)/nbin

    @classmethod
    def from_centers(cls,centers):
        """
        Factory method for creating a binning class from given centers
        Parameters:
         - centers: 1d-array of bin centers
        """
        centers=np.array(centers)
        dv = np.mean(np.diff(centers))
        nbin = len(centers)
        return cls(centers[0]-0.5*dv,centers[-1]+0.5*dv,nbin)

    @classmethod
    def from_edges(cls, edges: "array_like") -> "binning":
        """
        Factory method for creating a binning class from given edges.
        
        Parameters:
         - centers: 1d-array of bin edges
        """
        
        nbin  = len(edges)-1
        
        return cls(edges[0], edges[-1], nbin)
    
    @classmethod
    def from_binning_new_range(cls, oldBins: "binning", newRange: Tuple[float,float]) -> "binning":
        '''
        Extend or shrink an array of bin edges to a new range, 
        while retaining bin width and edges within that range.
        '''

        # to be clear
        oldEdges = oldBins.edges

        # move inwards by machine epsilon to avoid creating too many bins by accident
        eps = np.finfo(float).eps
        newRange = (newRange[0]+eps, newRange[1]-eps)

        # bin width (assuming equal size)
        width = oldEdges[1]-oldEdges[0]

        # create extensions if new range is bigger on either side
        extensionLower = np.flip(np.arange(oldEdges[0] -width, newRange[ 0]-width,-width))
        extensionUpper =         np.arange(oldEdges[-1]+width, newRange[-1]+width, width)

        # create masks for the case that new range is smaller for eiter side
        mask1 = (extensionLower>newRange[0]-width) & (extensionLower<newRange[1]+width)
        mask2 = (oldEdges      >newRange[0]-width) & (oldEdges      <newRange[1]+width)
        mask3 = (extensionUpper>newRange[0]-width) & (extensionUpper<newRange[1]+width)

        # combine into one
        newEdges = np.concatenate((extensionLower[mask1], oldEdges[mask2], extensionUpper[mask3]))
        
        return cls.from_edges(newEdges)

    def __repr__(self):
        return f"binning(nbin={self.nbin}, vmin={self.vmin}, vmax={self.vmax}, dv={self.dv})"

    def __len__(self):
        return self.nbin



#-------------------------------------------------------------------------
# Binning utilities

@njit
def centers(binBorders: np.ndarray) -> np.ndarray:
    """
    Calculate bin centers from borders/edges,
    assuming equally sized bins.
    """

    return (binBorders[1:]+binBorders[:-1])/2.

@njit
def borders(binCenters: np.ndarray) -> np.ndarray:
    """
    Calculate bin borders/edges from centers,
    assuming equally sized bins.
    """
    
    width = binCenters[1]-binCenters[0]
    return np.linspace(binCenters[0]-width/2., binCenters[-1]+width/2., len(binCenters)+1)


def change_bin_range(binEdges: np.ndarray, newRange: Tuple[float,float]) -> np.ndarray:
    '''
    Extend or shrink an array of bin edges to a new range, 
    while retaining bin width and edges within that range.
    '''
    
    # to be clear
    oldBins = binEdges
    
    # bin width (assuming equal size)
    width = oldBins[1]-oldBins[0]

    # create extensions if new range is bigger on either side
    extensionLower = np.flip(np.arange(oldBins[0] -width, newRange[ 0]-width,-width))
    extensionUpper =         np.arange(oldBins[-1]+width, newRange[-1]+width, width)

    # create masks for the case that new range is smaller for eiter side
    mask1 = (extensionLower>newRange[0]-width) & (extensionLower<newRange[1]+width)
    mask2 = (oldBins       >newRange[0]-width) & (oldBins       <newRange[1]+width)
    mask3 = (extensionUpper>newRange[0]-width) & (extensionUpper<newRange[1]+width)

    # combine into one
    newBins = np.concatenate((extensionLower[mask1], oldBins[mask2], extensionUpper[mask3]))
    
    return newBins


#-------------------------------------------------------------------------
# Integration over bins

def integrate_bins_fast(f,eBinCenters,n=10):
    """
    Integrates a function f within the bins defined by eBinCenters
    Parameters:
        - f: function to be integrated
        - eBinCenters: 1d-array, must the contain evenly spaced bin centers
        - n=10: int, number of point within each bin used for integrations
    """
    #get the evaluation points for the array. this could be given as parameter,
    #which would speed up the calculation a lot.
    de=eBinCenters[1]-eBinCenters[0]
    delta=((np.arange(n)-(n-1)/2)/n)*de
    e_eval=np.repeat(eBinCenters,n)+np.tile(delta,eBinCenters.shape[0])
    
    #evaluate the function and sum over each bin
    f_out=f(e_eval)
    return np.reshape(f_out,(eBinCenters.shape[0],n)).sum(axis=1)*de/n


#-------------------------------------------------------------------------
# Bin to bin mapping for response construction

@njit
def intervalOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@njit
def map_bin_to_bin(oldBins,newBins):
    
    # initialize angle bin to angle bin response
    response = np.zeros((len(oldBins)-1,len(newBins)-1))
    
    # bin width
    w = (oldBins[1]-oldBins[0])
    
    # loop over input bins to see which fall into range
    for j,b in enumerate(zip(oldBins, oldBins[1:])):
        
        # loop over ranges corresponding to output bins
        for i,r in enumerate(zip(newBins, newBins[1:])):
            r = list(r)
            
            # matrix element from overlap of bin and range
            response[j,i] = intervalOverlap(b,r)/w
    
    return response