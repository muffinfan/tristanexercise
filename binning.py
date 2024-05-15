import numpy as np
from typing import Tuple

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
# Integration over bins

def integrate_bins_fast(f,bins,n=10):
    """
    Integrates a function f within the bins defined by binCenters
    Parameters:
        - f: function to be integrated
        - bins: binning object or 1d-array of evenly spaced bin centers
        - n=10: int, number of point within each bin used for integrations
    """

    # check if binning object or centers given
    if hasattr(bins, "edges") and hasattr(bins, "centers"):
        binCenters = bins.centers
    else:
        binCenters = bins

    #get the evaluation points for the array. this could be given as parameter,
    #which would speed up the calculation a lot.
    de=binCenters[1]-binCenters[0]
    delta=((np.arange(n)-(n-1)/2)/n)*de
    e_eval=np.repeat(binCenters,n)+np.tile(delta,binCenters.shape[0])
    
    #evaluate the function and sum over each bin
    f_out=f(e_eval)
    return np.reshape(f_out,(binCenters.shape[0],n)).sum(axis=1)*de/n

