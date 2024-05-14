from matplotlib import pyplot as plt
from collections.abc import Iterable
import numpy as np


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def init(usetex=True, serif=False, usewidget=False, labelsize='medium', sidescroll=False):
    """
    Set some matplotlib rc settings.

    usetex: set text interpreter for latex usage
    serif:  use serif font
    """
    if usetex:
        import os
        try:
            if '/usr/bin' not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + '/usr/bin'
        except:
            os.environ["PATH"] = '/usr/bin'

    if is_notebook():
        import matplotlib as mpl
        if usewidget:
            mpl.use('module://ipympl.backend_nbagg')
            plt.ioff()
        else:
            mpl.use('module://ipykernel.pylab.backend_inline')

        if sidescroll:
            from IPython.core.display import display, HTML
            display(HTML("<style>div.jp-OutputArea-output pre {white-space: pre;}</style>"))



    plt.rc('text', usetex=usetex)
    if usetex: plt.rcParams['text.latex.preamble'] = r'\usepackage{xfrac} \usepackage{sansmath} \sansmath'      
    plt.rcParams['font.family'] = 'serif' if serif else 'sans-serif'


def plotty(xlabel="", ylabel="", title="", suptitle="",
           fig=None, size=[8,5], axes=None, 
           xlim=None, ylim=None, log=None, sci=None, grid='major', gridAlpha=0.3,
           legend=False, legloc='best', legncol=1, 
           fontsize=18, legfontsize=16, tickfontsize=16,
           show=True, tight=True, file='', preset='', useOffset=None):
    
    '''  
    Apply common matplotlib plotting options in a compact way.
    Can be used instead of plt.show().

    Optional parameters:
    xlabel:       set x-label on axis
    ylabel:       set y-label on axis
    title:        set title on axis
    suptitle:     set suptitle for figure
    legend:       enable the legend
    legloc:       set legend position
    legncol:      set the number of legend columns
    size:         set figure size
    log:          set logarithmic scale on axis ('xy'/'x'/'y')
    fig:          set specific figure handle
    axes:         set specific figure handle or list of handles
    fontsize:     set base font size for figure/axis text
    tickfontsize: set font size of tick labels
    legfontsize:  set font size of legend
    xlim:         set manual x limit on axis
    ylim:         set manual y limit on axis
    sci:          turn on scientific axis notation ('xy'/'x'/'y')
    show:         if true call plt.show()
    tight:        if true apply plt.tight_layout()
    file:         filename string for figure saving
    grid:         set grid ('both'/'minor'/'major'/'off')
    '''

    if preset=='halfpage':
        size=[6.66,4]
    
    if fig == None: fig = plt.gcf();
    if suptitle != "": fig.suptitle(suptitle,fontsize=fontsize, y=0.91)
    if size != [None,None]: fig.set_size_inches(size)


    if isinstance(axes, Iterable):
        for i,axis in enumerate(axes):
            axis.tick_params(which='major',direction='out',width=1,length=6,labelsize=tickfontsize)
            axis.tick_params(which='minor',direction='out',width=.5,length=3,labelsize=tickfontsize)

            if grid!=None:
                if isinstance(grid, str): grid = [grid,grid]
                if grid[i] in ['major','both']: axis.grid(b=True, which='major', color='k', alpha=gridAlpha, linestyle='-')
                if grid[i] in ['minor','both']: axis.grid(b=True, which='minor', color='k', alpha=gridAlpha*3/5, linestyle=':')
            
            if sci!=None:
                if isinstance(sci, str): sci = [sci,sci]
                if 'x' in sci[i]: axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                if 'y' in sci[i]: axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            if log!=None:
                if isinstance(log, str): log = [log,log]
                if 'x' in log[i]: axis.set_xscale('log',nonpositive='clip'); axis.autoscale()
                if 'y' in log[i]: axis.set_yscale('log',nonpositive='clip'); axis.autoscale()
            
            if xlim!=None : axis.set_xlim(xlim[i])
            if ylim!=None : axis.set_ylim(ylim[i])

            if title:  axis.set_title(title[i],   fontsize=fontsize)
            if xlabel: axis.set_xlabel(xlabel[i], fontsize=fontsize)
            if ylabel: axis.set_ylabel(ylabel[i], fontsize=fontsize)

            if useOffset!=None: axis.ticklabel_format(useOffset=useOffset)

            if legend:
                axis.legend(loc=legloc,ncol=legncol,fontsize=legfontsize); #use inbuilt labels: e.g. from plt.plot(...,label='name')
    else:
        if axes == None: 
            axis = plt.gca()
        axis.tick_params(which='major',direction='out',width=1,length=6,labelsize=tickfontsize)
        axis.tick_params(which='minor',direction='out',width=.5,length=3,labelsize=tickfontsize)
        if grid in ['major','both']: axis.grid(b=True, which='major', color='k', alpha=gridAlpha, linestyle='-')
        if grid in ['minor','both']: axis.grid(b=True, which='minor', color='k', alpha=gridAlpha*3/5, linestyle=':')
        
        if sci!=None:
            if 'x' in sci: axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            if 'y' in sci: axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if log!=None:
            if 'x' in log: axis.set_xscale('log',nonpositive='clip'); axis.autoscale()
            if 'y' in log: axis.set_yscale('log',nonpositive='clip'); axis.autoscale()
        
        if xlim!=None : axis.set_xlim(xlim)
        if ylim!=None : axis.set_ylim(ylim)

        if title:  axis.set_title(title,   fontsize=fontsize)
        if xlabel: axis.set_xlabel(xlabel, fontsize=fontsize)
        if ylabel: axis.set_ylabel(ylabel, fontsize=fontsize)

        if useOffset!=None: axis.ticklabel_format(useOffset=useOffset)

        if legend:
            axis.legend(loc=legloc,ncol=legncol,fontsize=legfontsize); #use inbuilt labels: e.g. from plt.plot(...,label='name')    
    
    plt.gcf().set_facecolor('w')
    if tight: plt.tight_layout()    
    if file:  plt.savefig(file,bbox_inches='tight',dpi=300)
    if show:  plt.show()


def plot_hist(bins, contents, axis=None, outline=True, fill=True, fillalpha=0.4, label=None, hatch='', **kwargs):
    
    # get axis if not specified
    if axis==None: axis = plt.gca()
    
    # bin edges or centers given?
    if len(bins)==len(contents):
        x = bins
        y = contents
    elif len(bins)==(len(contents)+1):
        x = (bins[:-1]+bins[1:])/2.
        y = contents
    else:
        raise ValueError('Bins and contents have unmatching lengths. Bins must refer to edges or centers.')
    
    # append and prepend
    dx = x[1]-x[0]
    xp = np.pad(x+dx/2., (1, 1), 'constant', constant_values=(x[0]-dx/2., x[-1]+dx/2.))
    yp = np.pad(y,       (1, 1), 'constant', constant_values=(0, 0))
    
    # plot outline
    if outline:
        axis.step(xp,yp, **kwargs)
    
    # plot fill
    if fill:
        axis.fill_between(xp, yp, alpha=fillalpha, step='pre', label=label, hatch=hatch, **kwargs)