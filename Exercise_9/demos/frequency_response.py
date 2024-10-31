from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
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

class timeSeriesPlot:
    def __init__(self, ax, t, A_max, N=1, t_unit='s'):
        res  = len(t)
        self.N = N
        t_nd = np.outer(t, np.ones(self.N))
        x_t = np.zeros((res, self.N))          

        self.ax = ax
        self.lines = self.ax.plot(t_nd, x_t, zorder=10)
        
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([t[0], t[-1], -A_max, A_max])
        self.ax.grid(True)
        #self.ax.set_xticks(np.linspace(t[0],t[-1],11))
        
    def update(self, new_lines):
        assert self.N == len(new_lines), "Error: Parameter lenght different from number of sines."
        for i in range(self.N):
            self.lines[i].set_ydata(new_lines[i])
            
    def setLabels(self, names):
        #self.ax.legend(self.lines, names, loc='upper right')
        for i in range(len(names)):
            self.lines[i].set_label(names[i])
        
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

class FreqRespDemo:
    def __init__(self, b, a=[1], fig_num=1, figsize=(8,6)):
        
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
        
        self.b = b
        self.a = a
        self.M = len(b)
        self.N = len(a)
        self.w, Hw = sig.freqz(b, a, worN=512)
        self.Hw_amp = np.abs(Hw)
        self.Hw_phase = np.unwrap(np.angle(Hw))
        
        self.t_n = np.linspace(0, 16, 501)
        self.n = np.arange(16)
        
        
        # Amplituderespons
        ax11 = plt.subplot(2,2,1)
        ax11.plot(self.w, self.Hw_amp)
        ax11.set_xlim([0, pi])
        ax11.set_ylim(ymin=0)
        ax11.set_xticks(np.linspace(0, 1, 5)*pi)
        ax11.set_xticklabels(['$'+str(round(i,2))+'\\pi$' for i in np.linspace(0, 1, 5)])
        ax11.set_xlabel('Normalized Angular Frequency $\\hat{\\omega}$')
        ax11.set_ylabel('Magnitude Response $\\left| H\\left(\\hat{\\omega} \\right)\\right|$')
        ax11.grid(True)
        ax11.set_title('placeholder')
        self.ax11 = ax11
        
        # Markør for valgt frekvens:
        self.ampMarker, = ax11.plot([0], [1], 'oC3')
        

        # Frekvensrespons
        ax12 = plt.subplot(2,2,2)
        ax12.plot(self.w, self.Hw_phase/pi)
        phaseLabels = ax12.get_yticks()
        phaseLim = ax12.get_ylim()
        ax12.set_yticks(phaseLabels)
        ax12.set_ylim(phaseLim)
        ax12.set_yticklabels([r'$'+str(round(i,2))+'\pi$' for i in phaseLabels])
        ax12.set_xlim([0, pi])
        ax12.set_xticks(np.linspace(0, 1, 5)*pi)
        ax12.set_xticklabels([r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
        ax12.set_xlabel('Normalized Angular Frequency $\\hat{\\omega}$')
        ax12.set_ylabel('Phase Response $\\angle H\\left(\\hat{\\omega} \\right)$')
        ax12.grid(True)
        ax12.set_title('placeholder')
        self.ax12 = ax12
        
        # Markør for valgt frekvens:
        self.phaseMarker, = ax12.plot([0], [0], 'oC3')

        # Sinusfigurer
        ax2 = plt.subplot(2,2,(3,4))
        ax2.set_title('placeholder')
        self.waveforms = timeSeriesPlot(ax2, self.t_n, A_max=1.1, N=2)
        self.waveforms.setStyles([{'color':'tab:blue', 'linestyle':'-.'}, 
                                  {'color':'tab:red', 'linestyle':'-.'}])
        self.waveforms.lines[0].set_linewidth(0.5)
        self.waveforms.lines[1].set_linewidth(0.5)
        
        self.xn_stem = interactiveStem(ax2, self.n, sin(0*self.n), color='tab:blue', label='$x[n]$')
        self.yn_stem = interactiveStem(ax2, self.n, sin(0*self.n), color='tab:red', label='$y[n]$')
        ax2.legend(loc='upper right')
        self.ax2 = ax2
        
        # Confiugre Layout
        self.fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        
        #Set up slider panel
        normFreq = widget.FloatSlider(
                                    value = 1/8,
                                    min=0,
                                    max=127/128,
                                    step = 1/128,
                                    description=r'Normalized Angular Frequency',
                                    disabled=False,
                                    style = {'description_width': 'initial'},
                                    layout=Layout(width='95%'),
                                    continuous_update=False
                                    )
        self.layout = VBox([normFreq])
        self.userInput = {'w': normFreq}
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)

    
    def update(self, w):
        index = int(w*128)*4
        
        self.ampMarker.set_xdata(self.w[index])
        self.ampMarker.set_ydata(self.Hw_amp[index])
        self.phaseMarker.set_xdata(self.w[index])
        self.phaseMarker.set_ydata(self.Hw_phase[index]/pi)
        self.ax11.set_title(r"$\left| H \left( "+str(round(w,2))+r"\pi \right) \right| = "+str(round(self.Hw_amp[index],2))+"$")
        self.ax12.set_title(r"$\angle H \left( "+str(round(w,2))+r"\pi \right) = "+str(round(self.Hw_phase[index]/pi,2))+"\pi$")
        titlestr = (r"$x[n] = \sin("+str(round(w,2))+r"\pi \cdot n), \ \ \ \ y[n]="+
                          str(round(self.Hw_amp[index],2))+r"\cdot\sin("+str(round(w,2))+r"\pi \cdot n +"+str(round(self.Hw_phase[index]/pi,2))+r"\pi)$")
        titlestr=titlestr.replace("+-", "-")
        self.ax2.set_title(titlestr)        
        
        xt = sin(pi*w*self.t_n)
        yt = self.Hw_amp[index]*sin(pi*w*self.t_n+self.Hw_phase[index])
        self.waveforms.update([xt, yt])
        
        xn = sin(pi*w*self.n)
        yn = self.Hw_amp[index]*sin(pi*w*self.n+self.Hw_phase[index])
        self.xn_stem.update(self.n, xn)
        self.yn_stem.update(self.n, yn)

