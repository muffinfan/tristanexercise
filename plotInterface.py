import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba, SymLogNorm
from matplotlib.patches import Rectangle
from collections.abc import Iterable
from scipy.stats import chi2

def init(usetex=False, interactive=False, sidescroll=False, serif=False, dpi=100):
    """
    Set some matplotlib rc settings.
    
    Arguments:
        - usetex:      set text interpreter for latex usage
        - interactive: use interactive matplotlib nbagg backend if in notebook
        - serif:       use serif font
        - sidescroll:  may fix issues with sidescrolling if True
        - dpi:         default dpi for inline plotting
                       (adjusts size of plots, 72 for mpl default)
    """

    # latex options
    if usetex:
        # make sure /usr/bin is in path
        import os
        try:
            if '/usr/bin' not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + '/usr/bin'
        except:
            os.environ["PATH"] = '/usr/bin'

        # enable some latex packages
        plt.rcParams['text.latex.preamble'] = (
             # r'\usepackage{lmodern} '+
            r'\usepackage{xfrac} '+
            r'\usepackage{sansmath} \sansmath'+
            r'\usepackage[detect-all]{siunitx}'
            )

    # enable latex
    plt.rc('text', usetex=usetex)

    # for notebook select backend
    if is_notebook():
        if interactive:
            mpl.use('module://ipympl.backend_nbagg')
            plt.ioff()
        else:
            mpl.use('module://ipykernel.pylab.backend_inline')

        if sidescroll:
            from IPython.core.display import display, HTML
            display(HTML("<style>div.jp-OutputArea-output pre {white-space: pre;}</style>"))

    # set rc params        
    plt.rcParams['font.family'] = 'serif' if serif else 'sans-serif'
    mpl.rcParams['figure.dpi']  = dpi
    mpl.rcParams['savefig.dpi'] = 300


def is_notebook():
    """
    Utility to check if code is executed in a jupyter notebook.

    Returns: bool
    """
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


def plotty(fig=None, axes=None, size=[8,5],
           xlabel="", ylabel="", title="", suptitle="", suptitlepos=1.0,
           xlim=None, ylim=None, log=None, sci=None, useOffset=None,
           grid='major', gridAlpha=0.2,
           legend=False, legloc='best', legncol=1,
           fontsizeAxes=14, fontsizeTicks=12, fontsizeLegend=12, fontsizeTitle = 14, fontsizeSuptitle = 14,
           whiteBg=True, tight=True, show=True,
           savefile='', png=True, pdf=True, dpi=300):
    
    '''  
    Apply common matplotlib plotting options in a compact way.
    Can be used in place of plt.show().

    For multiple sublplots/axes:
      Options can be given as list to specify for each individual
      subplot. If no list is provided, same option is applied for
      each subplot.

    Arguments:
        - fig:              set specific figure handle (otherwise gets current fig)
        - axes:             set specific axis handle (otherwise gets current axes)
        - size:             set figure size in inches
        - xlabel:           set x-label on axis (two component list/tuple)
        - ylabel:           set y-label on axis (two component list/tuple)
        - title:            set title on axis
        - suptitle:         set suptitle for figure
        - suptitlepos:      position of suptitle (fractional)
        - xlim:             set manual x limit on axis
        - ylim:             set manual y limit on axis
        - log:              set logarithmic scale on axis ('xy'/'x'/'y')
        - sci:              turn on scientific axis notation ('xy'/'x'/'y')
        - useOffset:        set useOffset in axis.ticklabel_format()
        - grid:             set grid ('both'/'minor'/'major'/'off')
        - gridAlpha:        grid opacity
        - legend:           enable the legend
        - legloc:           set legend position
        - legncol:          set the number of legend columns
        - fontsizeAxes:     font size axis labels
        - fontsizeTicks:    font size axis ticks
        - fontsizeLegend:   font size legend 
        - fontsizeTitle:    font size title
        - fontsizeSuptitle: font size suptitle
        - tight:            if true apply plt.tight_layout()
        - whiteBg:          make background white (otherwise transparent)
        - show:             if true call plt.show() at the end
        - savefile:         filename string for figure saving
        - png:              save as png (png + pdf is default)
        - pdf:              save as pdf (png + pdf is default)
        - dpi:              resolution for savefile
                            (default=300 vs. mpl default=72)
    '''
    
    # set up figure and get axes
    if fig == None: fig = plt.gcf();
    if suptitle != "": fig.suptitle(suptitle,fontsize=fontsizeSuptitle, y=suptitlepos)
    if size != [None,None]: fig.set_size_inches(size)
    if axes is None: axes = fig.get_axes()

    # loop over all axes
    nPlots = len(axes)
    for i,axis in enumerate(axes):
        axis.tick_params(which='major',direction='out',width=1,length=6,labelsize=fontsizeTicks)
        axis.tick_params(which='minor',direction='out',width=.5,length=3,labelsize=fontsizeTicks)

        if grid!=None:
            if isinstance(grid, str): grid = [grid]*nPlots
            if grid[i] in ['major','both']: axis.grid(which='major', color='k', alpha=gridAlpha, linestyle='-')
            if grid[i] in ['minor','both']: axis.grid(which='minor', color='k', alpha=gridAlpha*3./5., linestyle=':')
        
        if sci!=None:
            if isinstance(sci, str): sci = [sci]*nPlots
            if 'x' in sci[i]: axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            if 'y' in sci[i]: axis.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if log!=None:
            if isinstance(log, str): log = [log]*nPlots
            if 'x' in log[i]: 
                axis.set_xscale('log',nonpositive='clip'); axis.autoscale()
            else:
                axis.set_xscale('linear'); axis.autoscale()
            if 'y' in log[i]: 
                axis.set_yscale('log',nonpositive='clip'); axis.autoscale()
            else:
                axis.set_yscale('linear'); axis.autoscale()
        
        if xlim!=None:
            if len(np.shape(xlim))==1: xlim = [xlim]*nPlots
            axis.set_xlim(xlim[i])

        if ylim!=None: 
            if len(np.shape(ylim))==1: ylim = [ylim]*nPlots
            axis.set_ylim(ylim[i])

        if title:
            if isinstance(title, str): title = [title]*nPlots
            axis.set_title(title[i],   fontsize=fontsizeTitle)
        
        if xlabel: 
            if isinstance(xlabel, str): xlabel = [xlabel]*nPlots
            axis.set_xlabel(xlabel[i], fontsize=fontsizeAxes)
        
        if ylabel: 
            if isinstance(ylabel, str): ylabel = [ylabel]*nPlots
            axis.set_ylabel(ylabel[i], fontsize=fontsizeAxes)

        if useOffset!=None: 
            if isinstance(useOffset, bool): useOffset = [useOffset]*nPlots
            axis.ticklabel_format(useOffset=useOffset)

        if legend!=False:
            if isinstance(legend, bool): legend = [legend]*nPlots
            if isinstance(legloc, str): legloc = [legloc]*nPlots
            if legend[i]: axis.legend(loc=legloc[i],ncol=legncol,fontsize=fontsizeLegend)

    # make background white
    if whiteBg: fig.set_facecolor('w')

    # activate tight layout
    if tight: plt.tight_layout()
    
    # save to file
    if savefile: savefig(savefile, png=png, pdf=pdf, dpi=dpi)
    
    # finally show plot
    if show:  plt.show()


def savefig(savefile, png=True, pdf=True, dpi=300):
    '''
    Save matplotlib figure with high resolution as pdf and png.

    Arguments:
        - savefile: Path to file including filename 
                    (file extension is added automatically)
        - png:      Save as png (png + pdf is default)
        - pdf:      Save as pdf (png + pdf is default)
        - dpi:      increases resolution
                    (default=300 vs. mpl default=72)
    '''

    # remove file extension just in case
    import os
    savefile = os.path.splitext(savefile)[0]
    if '/' in savefile: 
        filepath = savefile
    else:
        filepath = './'+savefile

    if (not png) and (not pdf):
        print('Figure not saved since all formats (png, pdf) are disabled.')

    # save as png and/or pdf
    if png: 
        plt.savefig(filepath+'.png',bbox_inches='tight',dpi=dpi)
        print(f'Figure saved to {filepath}.png')
    if pdf: 
        plt.savefig(filepath+'.pdf',bbox_inches='tight',dpi=dpi)
        print(f'Figure saved to {filepath}.pdf')


def plot_hist(bins, contents, bottom=None, axis=None, outline=True, smooth=False, fill=True, fillalpha=0.3, 
    label=None, hatch='', fc=None, ec=None, hc=None, fillkwargs={},**kwargs):
    '''
    Plotting nice looking histograms made simple.

    Arguments:
        - bins:       Auto-detect Binning object / np.array of edges / np.array of centers
        - contents:   np.array of contents in each bin
        - bottom:     Starting point for shading (float or np.array)
        - axis:       Set pre-existing axis (otherwise retrieves current axis via plt.gca())
        - outline:    Draw outpline above shading
        - smooth:     False -> plt.steps(), True -> plt.plot() (for the outline)
        - fill:       Draw shading
        - fillalpha:  Alpha for shading
        - label:      Label for legend
        - hatch:      Apply hatching (see matplotlib hatching options)
        - fc:         face color -> shading (RGBA)
        - ec:         edge color -> outline (RGBA)
        - hc:         hatch color (RGBA)
        - fillkwargs: Further kwargs dict for shading (plt.fill_between())
        - kwargs:     Further kwargs dict for outline (plt.step() or plt.show())

    '''
    
    # get axis if not specified
    if axis==None: axis = plt.gca()

    # bining object, edges, or centers given?
    if hasattr(bins, "edges") and hasattr(bins, "centers"):
        xc = bins.centers
        yc = contents
    elif len(bins)==len(contents): # centers given
        xc = bins
        yc = contents
    elif len(bins)==(len(contents)+1): # edges given
        xc = (bins[:-1]+bins[1:])/2.
        yc = contents
    else:
        raise ValueError('Bins and contents have unmatching lengths. Bins must refer to edges or centers.')

    # set botton automatically
    if bottom is None:
        ybe = np.zeros(len(xc)+2)
        ybc = np.zeros(len(xc))
    else:
        if not isinstance(bottom, Iterable):
            ybc = np.array([bottom]*len(xc))
            ybe = np.array([bottom]*(len(xc)+2))
        else:
            ybc = bottom
            ybe = np.pad(ybc, (1, 1), 'constant', constant_values=(ybc[0], ybc[-1]))
            #ValueError('bottom cannot be Iterable')

    
    # append and prepend
    dx = xc[1]-xc[0]
    xe = np.pad(xc+dx/2., (1, 1), 'constant', constant_values=(xc[0]-dx/2., xc[-1]+dx/2.))
    ye = np.pad(yc,       (1, 1), 'constant', constant_values=(ybe[0], ybe[-1]))
    
    # override color with ec
    if ec is not None: kwargs['color'] = ec

    # plot outline
    if outline:

        # kwargs are used only for line
        if smooth:
            line, = axis.plot(xc,yc, **kwargs)
        else:        
            line, = axis.step(xe,ye, **kwargs)
    
    # plot fill
    if fill:
        if fc is None:
            if 'line' in locals(): fillkwargs['fc'] = line.get_color()
        else:
            fillkwargs['fc'] = fc

        if smooth:
            face = axis.fill_between(xc, yc, ybc, alpha=fillalpha, **fillkwargs)
        else:
            face = axis.fill_between(xe, ye, ybe, alpha=fillalpha, step='pre', **fillkwargs)

    if hatch!='':
        al = kwargs['alpha'] if 'alpha' in kwargs.keys() else 1.0
        if hc is None: 
            if 'line' in locals(): hc = to_rgba(line.get_color(),al)

        if smooth:
            htch = axis.fill_between(xc, yc, ybc, fc=(0,0,0,0), hatch=hatch, ec=hc, step='pre')
        else:
            htch = axis.fill_between(xe, ye, ybe, fc=(0,0,0,0), hatch=hatch, ec=hc, step='pre')

    # make patch for legend
    fc = face.get_facecolor() if 'face' in locals() else (0,0,0,0)
    ec = to_rgba(line.get_color(),line.get_alpha()) if 'line' in locals() else None
    lw = line.get_linewidth() if 'line' in locals() else 0
    ls = line.get_linestyle() if 'line' in locals() else None

    rect = Rectangle((0, 0), 0, 0, fc=fc, ec=ec, hatch=hatch*2, lw=lw, visible=True, label=label, ls=ls)
    axis.add_patch(rect)

