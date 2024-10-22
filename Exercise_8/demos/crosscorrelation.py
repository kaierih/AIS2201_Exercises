from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget


# Interactive stem plot with matlab-ish default config
class interactiveStem:
    def __init__(self, ax, n, xn, color='tab:blue', marker='o', label=None, filled=False):
        self.ax = ax
        self.samples = self.ax.stem(n, # 
                                    xn, # Nullsampler
                                    basefmt='black', # Farge på y=0 aksen
                                    label=label
                                    )
        self.samples.baseline.set_linewidth(0.5)
        self.samples.baseline.set_xdata([0, len(n)])
        self.samples.markerline.set_color(color)
        self.samples.markerline.set_marker(marker)
        if not filled:
            self.samples.markerline.set_markerfacecolor('none')
        self.samples.stemlines.set_color(color)
        self.ax.grid(True)
        
    def update(self, n, xn):
        self.N = len(n)
        
        # Make new line collection
        points = np.array([n, xn]).T.reshape(-1, 1, 2)
        start_points = np.array([n, np.zeros(len(n))]).T.reshape(-1, 1, 2)
        segments = np.concatenate([start_points, points], axis=1)
        
        # Adjust markers and lines
        self.samples.stemlines.set_segments(segments)
        self.samples.markerline.set_xdata(n)
        self.samples.markerline.set_ydata(xn)
        
class CrossCorrelationDemo:
    def __init__(self, xn, hn, fig_num = 1, figsize=(8, 6)):
        self.xn = xn
        self.L = len(xn)
        self.hn = hn
        self.M = len(hn)
        self.yn = np.correlate(self.xn, self.hn, mode="full")
        
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
             
        ### Subplot 1
        # Plot Input signal x[k]
        ax1 = plt.subplot(3,1,1)
        ax1.set_title(r'Cross-correlation: $r_{xy}[l] = \sum_{n}x[n]\cdot y[n-l]$')
        xn_samples = ax1.stem(xn, 
                               linefmt='C0', # Linjestil stolper
                               markerfmt='oC0', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                               basefmt='black', # Farge på y=0 aksen
                               label=r'$x[n]$'
                               )
        xn_samples.baseline.set_linewidth(0.5)
        xn_samples.baseline.set_xdata([-self.M, self.L+self.M])
        xn_samples.markerline.set_markerfacecolor('none')
        ax1.set_xlim([-self.M, self.L+self.M-1])
        
        
        # Plot reversed impulse response h[n-k]
        self.hn_samples = interactiveStem(ax1, np.arange(len(self.hn)), self.hn, color='C3', marker='x', label=r'$h[n-k]$')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ### Subplot 2
        # Plot x[k]*h[n-k]
        ax2 = plt.subplot(3,1,2)
        ax2.grid(True)
        self.xn_yn = interactiveStem(ax2, [0], [self.hn[0]*self.xn[0]], color='C4', label=r'$x[k]\cdot x[n-k]$', filled=True)
        self.xn_yn.ax.set_xlim([-self.M, self.L+self.M-1])
        self.xn_yn.samples.baseline.set_xdata([-self.M, self.L+self.M])
        self.xn_yn.ax.set_ylim([-max(abs(xn))*max(abs(hn))*1.05, max(abs(xn))*max(abs(hn))*1.05])
        self.xn_yn.ax.legend(loc='upper left')
        ax2.set_xlabel(r'$n$')

        
        ### Subplot 3
        # Plot y[n]
        ax3 = plt.subplot(3,1,3)
        n = np.arange(-self.M+1, self.L)
        yn_samples = ax3.stem(n,
                              self.yn,
                              linefmt='C2', # Linjestil stolper
                               markerfmt='oC2', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                               basefmt='black', # Farge på y=0 aksen
                               label=r'$r_{xy}[l]$'
                               )
        yn_samples.baseline.set_linewidth(0.5)
        yn_samples.baseline.set_xdata([-self.M, self.L+self.M])
        yn_samples.markerline.set_markerfacecolor('none')
        ax3.set_xlim([-self.M, self.L])
        ax3.grid(True)
        ax3.set_xlabel(r'$l$')
        self.yn_active = interactiveStem(ax3, [0], [self.yn[0]], color='C3', label=r'$y[n]$', filled=True)
        ax3.legend(loc='upper left')
        if max(np.abs(ax3.get_ylim())) < max(np.abs(ax1.get_ylim())):
            ax3.set_ylim(np.array(ax1.get_ylim()))
        
        # Confiugre Layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
        
        #Set up slider panel
        # Set up UI panel
        sample_num = widget.IntSlider(  value = 0,
                                        min=-self.M+1,
                                        max=self.L-1,
                                        step = 1,
                                        description=r'Displacement $l$',
                                        disabled=False,
                                        style = {'description_width': 'initial'},
                                        layout=Layout(width='95%'),
                                        continuous_update=True
                                        )
        self.layout = VBox([sample_num])
        self.userInput = {'l': sample_num}
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        
    def update(self, l):
        n_1 = max(0, l)
        n_2 = min(self.L, l+self.M)
        l_1 = max(0, -l)
        l_2 = min(self.M, self.L - l)
        n_array = np.arange(n_1, n_2)
        noverlap = min(self.L, n_2-n_1)
        self.xn_yn.update(n=n_array, xn=self.xn[n_1:n_2]*self.hn[l_1:l_2])
        self.yn_active.update([l], [self.yn[l+self.M-1]])
        self.hn_samples.update(np.arange(l, l+self.M), self.hn)
        
        
        self.hn_samples.samples.set_label(f'$y[n-{l}]$'.replace("--","+"))
        self.hn_samples.ax.legend(loc='upper left')
        self.xn_yn.samples.set_label(f'$x[n]\\cdot y[n-{l}]$'.replace("--","+"))
        self.xn_yn.ax.legend(loc='upper left')
        self.yn_active.samples.set_label(r'$r_{xy}['+str(l)+']$')
        self.yn_active.ax.legend(loc='upper left')
        self.fig.axes[0].set_title(('Cross-correlation: $r_{xy}[%d] = \\sum_{n}x[n]\\cdot y[n-%d]$'%(l, l)).replace("--","+"))
