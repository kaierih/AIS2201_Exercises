from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget

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

class timeSeriesPlot:
    def __init__(self, ax, t, A_max, N=1, t_unit='s'):
        res  = len(t)
        self.N = N
        t_nd = np.outer(t, np.ones(self.N))
        x_t = np.zeros((res, self.N))          

        self.ax = ax
        self.lines = self.ax.plot(t_nd, x_t)
        
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([t[0], t[-1], -A_max, A_max])
        self.ax.grid(True)
        self.ax.set_xticks(np.linspace(t[0],t[-1],11))
        self.ax.set_xlabel("Tid (" + t_unit + ")")
        
    def update(self, new_lines):
        assert self.N == len(new_lines), "Error: Parameter lenght different from number of sines."
        for i in range(self.N):
            self.lines[i].set_ydata(new_lines[i])
            
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
                                    markerfmt='oC3', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                                    basefmt='black', # Farge på y=0 aksen
                                    )
        self.samples.baseline.set_linewidth(0.5)
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([0, self.N, A_min, A_max])
        self.ax.set_xticks(np.arange(N+1))
        self.ax.grid(True)
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
        self.ax.set_xticks(np.arange(self.N+1))
        
        # Adjust axis limits
        self.ax.set_xlim([-0.0, self.N])

# Samplet sinussignal
class DFT_Demo():
    def __init__(self, fig_num=1, fig_size=(9,5)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        # Set up sine waves on canvas
        ax1 = plt.subplot(3,1,1)
        self.discreteSignal = interactiveStem(ax1, A_max=1.1, A_min=-1.1)
        self.discreteSignal.ax.set_ylabel('$x[n]$')
        self.discreteSignal.ax.set_title('Sinusoid')
        

        
        # Set up sine waves on canvas
        ax2 = plt.subplot(3,1,2)
        self.DFT_Amp = interactiveStem(ax2, A_min=0, A_max=10)
        self.DFT_Amp.ax.set_xlabel("Frequency index $m$")
        self.DFT_Amp.ax.set_ylabel('$|X[m]|$')
        self.DFT_Amp.ax.set_title('Magnitude Spectrum')

        # Set up sine waves on canvas
        ax3 = plt.subplot(3,1,3)
        self.DFT_Phase = interactiveStem(ax3, A_max=np.pi, A_min=-np.pi)
        self.DFT_Phase.ax.set_xlabel("Frekvens-indeks $m$")
        self.DFT_Phase.ax.set_ylabel('$\\angle X[m]$')
        self.DFT_Phase.ax.set_yticks(np.pi*np.linspace(-1,1,9))
        self.DFT_Phase.ax.set_yticklabels([f'${x:.2f}\\pi$' for x in np.linspace(-1,1,9)])
        self.DFT_Phase.ax.set_title('Phase Spectrum')
        
        # Tilpass figur-layout

        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        # Set up UI panel
        window_len = widget.BoundedIntText(
                                        value = 10,
                                        min=0,
                                        max=64,
                                        step = 1,
                                        description='DFT window length $N$:',
                                        disabled=False,
                                        style = {'description_width': 'initial'},
                                        layout=Layout(width='95%'),
                                        continuous_update=True
        )
        signal_freq = widget.FloatSlider(
                                        value = 0.2,
                                        min=0,
                                        max=1,
                                        step = 0.1,
                                        description='Frequency $\\hat{\\omega}\\ (\\times \\pi)$:',
                                        disabled=False,
                                        style = {'description_width': '30%'},
                                        layout=Layout(width='95%'),
                                        continuous_update=False
        )
        signal_amp = widget.FloatSlider(
                                        value = 1.0,
                                        min=0,
                                        max=1,
                                        step = 0.05,
                                        description='Ampltiude $A$:',
                                        disabled=False,
                                        style = {'description_width': '30%'},
                                        layout=Layout(width='95%'),
                                        continuous_update=False
        )
        signal_phase = widget.FloatSlider(
                                        value = 0,
                                        min=-1,
                                        max=1,
                                        step = 1/12,
                                        description='Phase $\\phi$:',
                                        disabled=False,
                                        style = {'description_width': '30%'},
                                        layout=Layout(width='95%'),
                                        continuous_update=False
        )
        self.layout = HBox([VBox([signal_amp, signal_freq, signal_phase], layout=Layout(width='140%')), window_len])
        self.userInput = {
            'N': window_len,
            'F': signal_freq,
            'A': signal_amp,
            'phi': signal_phase
        }
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        

        
    def update(self, N, F, A, phi):

        self.userInput['F'].step = 2/N
        F = 2/N*np.round(N/2*F)
        self.userInput['F'].value = F
        # Update discrete samples
        n = np.arange(0, N)
        w_d = pi*F
        xn = A*cos(w_d*n+phi*pi)
        self.discreteSignal.update(n, xn)
        self.discreteSignal.ax.set_title(r'Sinusoid $x[n]='+str(round(A,1))+'\\cdot \\cos('+str(round(F,2))+'\\pi\\cdot n+'+str(round(phi,2))+'\\pi) $')
       
        Xk = np.fft.fft(xn)
        self.DFT_Amp.update(n, np.absolute(Xk))
        self.DFT_Amp.ax.set_ylim(ymax=N)
        
        self.DFT_Phase.update(n, np.angle(Xk)*(np.absolute(Xk)>1e-10))