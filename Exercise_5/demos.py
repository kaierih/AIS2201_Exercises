from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget

# Funksjoner og klassedefinisjoner tilhørende demoer om frekvensanalyse    

def getImpulseLines(f, A, f_max):
    assert len(f)==len(A), "Error, arrays must be same length"
    f_line = np.concatenate(([-f_max], np.outer(f, np.ones(3)).flatten(), [f_max]))
    A_line = np.concatenate(([0], np.outer(A, [0, 1, 0]).flatten(), [0]))   
    return [f_line, A_line]


class dualSpectrumPlot:
    def __init__(self, ax, f_max, A_max=1, A_min=0, N=1):
        self.N = N
        self.ax = ax
        self.f_max =f_max
        self.A_max = A_max
        
        f_nd = np.outer([-f_max, f_max], np.ones(N))
        A_nd = np.zeros((2, self.N))
   
        self.lines = plt.plot(f_nd, A_nd, linewidth=2)
    
        self.ax.axis([-f_max, f_max, A_min, A_max])
        self.ax.grid(True)
        self.ax.set_xlabel("Frekvens $f$ (Hz)")
    
    def update(self, new_x, new_y):
        assert self.N == len(new_x) == len(new_y), "Error: Parameter lenght different from number of sines."
        for i in range(self.N):
            self.lines[i].set_xdata(new_x[i])  
            self.lines[i].set_ydata(new_y[i])  
            
    def setLabels(self, names):
        self.ax.legend(self.lines, names, loc='upper right')
        
    def setStyles(self, styles):
        for i in range(min(len(styles), len(self.lines))):
            try:
                self.lines[i].set_color(styles[i]['color'])
            except:
                pass
            
            try:
                self.lines[i].set_linestyle(styles[i]['linestyle'])
            except:
                pass    


def make_stem_segments(n, xn):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([n, xn]).T.reshape(-1, 1, 2)
    start_points = np.array([n, np.zeros(len(n))]).T.reshape(-1, 1, 2)

    segments = np.concatenate([start_points, points], axis=1)
    return segments

class interactiveStem:
    def __init__(self, ax, N = 1, A_max=1, A_min=-1):
        self.N = N
        self.ax = ax
        self.n = np.arange(self.N)
        self.samples = self.ax.stem(self.n, # 
                                    np.zeros(self.N), # Nullsampler
                                    linefmt='C3', # Linjestil stolper
                                    markerfmt='xC3', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                                    basefmt='black' # Farge på y=0 aksen
                                    )
        self.samples.baseline.set_linewidth(0.5)
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([0, self.N, A_min, A_max])
        #self.ax.set_xticks(np.arange(N+1))
        #self.ax.grid(True)
        self.ax.set_xlabel("Samplenummer $n$")
        
    def update(self, n, xn):
        self.N = len(n)
        
        # Adjust stemlines, markerline, baseline in stemcontainer
        segments = make_stem_segments(n, xn)
        self.samples.stemlines.set_segments(segments)
        self.samples.markerline.set_xdata(n)
        self.samples.markerline.set_ydata(xn)
        self.samples.baseline.set_xdata([0, self.N])
        self.samples.baseline.set_ydata([0, 0])
        
        # Adjust sample markers
        #self.ax.set_xticks(np.arange(self.N+1))
        
        # Adjust axis limits
        #self.ax.set_xlim([-0.0, self.N])

# Spektral lekasje demo
class SpectralLeakageDemo:
    def __init__(self, fig_num=1, figsize=(8,6)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)

        # Set up subplot with amplitude spectrum
        ax1 = plt.subplot(1, 1, 1)
        #ax2.set_title(r"Amplitudespekter til sinussignal")
        #ax2.set_ylabel(r'$\left|X\left(e^{j 2\pi f}\right)\right|$')
        
        self.AmpSpectrum = dualSpectrumPlot(ax1, f_max=1, A_max = 1,  N = 1)
        self.AmpSpectrum.ax.set_xticks(np.pi*np.linspace(-1,1,9))
        self.AmpSpectrum.ax.set_xticklabels([str(round(x, 2)) + '$\\pi$' for x in np.linspace(-1,1,9)])
        self.AmpSpectrum.ax.set_xlabel('Normalized angular frequency $\\hat{\\omega}$')
        self.AmpSpectrum.ax.set_ylabel('Magnitude')
        self.AmpSpectrum.setStyles([{'color': 'tab:blue'}])
        self.AmpSpectrum.lines[0].set_label(r'"True" frequency spectrum for $x[n]$')
        self.AmpSpectrum.ax.set_title("blablabla\n")
        # Set up subplot with phase spectrum
        ax2 = ax1.twiny()
        
        self.DFT_Amp = interactiveStem(ax2, A_max=1, A_min=0)
        self.DFT_Amp.ax.set_xlabel("Frequency index $m$")
        self.DFT_Amp.samples.set_label('$|X[m]|$ for $N$-point DFT')
        self.fig.legend(bbox_to_anchor=(0.42, 0.80), loc=1)
        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up UI panel
        window_len = widget.BoundedIntText(
                                        value = 16,
                                        min=0,
                                        max=64,
                                        step = 1,
                                        description='DFT window length $N$:',
                                        disabled=False,
                                        style = {'description_width': 'initial'},
                                        layout=Layout(width='95%'),
                                        continuous_update=False
        )
        signal_freq = widget.FloatSlider(
                                        value = 0.2,
                                        min=0,
                                        max=1,
                                        step = 0.01,
                                        description='Normalized angular frequency $\\hat{\\omega}\ (\\times \\pi)$:',
                                        disabled=False,
                                        style = {'description_width': '40%'},
                                        layout=Layout(width='95%'),
                                        continuous_update=False
        )

        self.layout = HBox([VBox([signal_freq], layout=Layout(width='140%')), window_len])
        self.userInput = {
            'N': window_len,
            'F': signal_freq,
        }

        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        
    def update(self, F, N):
        n = np.arange(N)
        xn = cos(np.pi*F*n)
        Xk = np.fft.fft(xn)
        Xk_amp = np.absolute(np.fft.fftshift(Xk))

        k = np.arange(-(N//2), (N+1)//2)
        self.DFT_Amp.ax.set_xlim([-N/2, N/2])
        self.DFT_Amp.update(k, Xk_amp)
        self.AmpSpectrum.ax.set_title('$x[n] = \\cos ('+str(F)+'\\pi \\cdot n)$ \n')
               
        
        if F==0:
            f_line, A_line = getImpulseLines([0],[N], self.AmpSpectrum.f_max)
        else:
            f_line, A_line = getImpulseLines([-F*np.pi, F*np.pi],[N/2, N/2], self.AmpSpectrum.f_max)
                                            
        self.AmpSpectrum.update([f_line],[A_line])
        self.AmpSpectrum.ax.set_ylim(ymax=N/1.7)