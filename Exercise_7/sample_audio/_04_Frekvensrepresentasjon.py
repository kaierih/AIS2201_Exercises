from numpy import sin, cos, pi, exp
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget

# Funksjon brukt i eksempler om frekvensforskyvning
def displayDualSpectrum(x, fs, color=None, label=None, linestyle=None):
    N = len(x)
    Xk = np.fft.fft(x)/N
    Xk = np.fft.fftshift(Xk)
    f = np.array(np.arange(-fs/2, fs/2, fs/N))
    plt.xlim([-fs/2, fs/2])
    plt.plot(f, np.abs(Xk), color=color, label=label, linestyle=linestyle)
    plt.xlabel("Frekvens (Hz)")
    plt.grid(True)
    plt.ylim(ymin=0)


# UI tool to inspect segments of a larger signal in time and frequency domain   
class signalAnalyzer:
    def __init__(self, x_n, f_s, fig_num=1):
        self.x_n = x_n
        self.f_s = f_s
        self.t = np.linspace(0, len(self.x_n)/self.f_s, len(self.x_n), endpoint=False)
        
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(12, 8))
        
        
        # Set up plot showing signal selection
        self.ax1 = plt.subplot(4, 1, 1)
        self.ax1.set_title(r"Full Signal Plot")
        self.ax1.plot(self.t, self.x_n,  color='tab:blue')
        self.ax1.grid(True)
        self.ax1.set_xlabel('Time t (seconds)')
        self.highlight, = self.ax1.plot(self.t, self.x_n, color='tab:orange')
        self.ax1.axis(xmin=self.t[0], xmax=self.t[-1])
        
        
        
        # Set up signal segment inspection plot
        self.ax2 = plt.subplot(4, 1, (2,4))
        
        self.selectionCurve, = self.ax2.plot(self.t, self.x_n, color='tab:blue')
        self.ax2.grid(True)
        
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=2.0)
        
        # Set up UI panel
        domainSelection = widget.RadioButtons(
            options=['Time Trace', 'Frequency Spectrum'],
            value='Time Trace',
            description='Display: ',
            disabled=False,
            continuous_update=False
        )
        win_start = widget.FloatSlider(
            value = 0.0,
            min=0.0,
            max=(len(self.x_n)-1)/self.f_s,
            step = 0.01,
            description='Signal segment start (seconds):',
            disabled=False,
            style = {'description_width': 'initial'},
            layout=Layout(width='95%'),
            continuous_update=False
        )
        win_length = widget.BoundedFloatText(    
            value=0.1,
            min=0.0,
            max=(len(self.x_n)-1)/self.f_s,
            step=0.01,
            description='Signal segment lenght (seconds):',
            disabled=False,
            style = {'description_width': 'initial'},
            #layout=Layout(width='95%'),
            continuous_update=False
        )
        self.layout = VBox(
            [win_start, 
             HBox([win_length, domainSelection])]
        )
        self.userInput = {
            't_start': win_start,
            't_length': win_length,
            'domain': domainSelection
        }
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        # Run demo
        #out = interactive_output(self.update, self.sliders)
        #display(self.layout, out)
        
    def update(self, t_start, t_length, domain):
        n_start = int(t_start*self.f_s)
        n_stop = int((t_start+t_length)*self.f_s)
        self.highlight.set_ydata(self.x_n[n_start:n_stop])
        self.highlight.set_xdata(self.t[n_start:n_stop])
        if domain=='Time Trace':
            self.ax2.set_xlabel('Time t (seconds)')
            self.ax2.set_ylabel('Value x(t)')
            self.selectionCurve.set_ydata(self.x_n[n_start:n_stop])
            self.selectionCurve.set_xdata(self.t[n_start:n_stop])
            self.ax2.set_xlabel("Time t (seconds)")
            self.ax2.set_ylabel("Value x(t)")
            self.ax2.set_title("Time plot of selected signal segment")
            self.ax2.axis(xmin=self.t[n_start], xmax=self.t[n_stop], ymin=min(self.x_n)*1.1, ymax=max(self.x_n)*1.1)
            
        elif domain=='Frequency Spectrum':
            M = n_stop-n_start
            f, Sxx_sub = welch(self.x_n[n_start:n_stop], self.f_s, 'hamming', int(M/4), int(M/8), int(M/2))
            Sxx_sub_dB = 10*np.log10(Sxx_sub)
            
            self.ax2.set_xlabel("Frequency f (Hz)")
            self.ax2.set_ylabel("Power Pxx(f) (dB)")
            self.ax2.set_title("Frequency content of selected signal segment")
            self.selectionCurve.set_ydata(Sxx_sub_dB)
            self.selectionCurve.set_xdata(f)
            self.ax2.axis(xmin=0, xmax=self.f_s/2, ymin=min(Sxx_sub_dB), ymax=max(Sxx_sub_dB)+5)
            
        else:
            pass
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=2.0)

# Funksjoner og klassedefinisjoner tilhørende demo om Frekvensmiksing        

def getImpulseLines(f, A, f_max):
    assert len(f)==len(A), "Error, arrays must be same length"
    f_line = np.concatenate(([-f_max], np.outer(f, np.ones(3)).flatten(), [f_max]))
    A_line = np.concatenate(([0], np.outer(A, [0, 1, 0]).flatten(), [0]))   
    return [f_line, A_line]

def sliderPanelSetup(set_details, n_of_sets=1, slider_type='float'):
    panel_col = []
    sliders = {}
    for i in range(n_of_sets):
        panel_row = []
        for item in set_details:
            mathtext = item['description']
            mathtext = mathtext.strip('$')
            if n_of_sets > 1:
                if mathtext.find(" ") == -1:
                    mathtext = '$' + mathtext + '_' + str(i+1) + '$' 
                else:
                    mathtext = '$' + mathtext.replace(" ", '_'+str(i+1)+'\ ', 1) + '$'
            else:
                mathtext = '$' + mathtext + '$'
            #mathtext = r'{}'.format(mathtext)

            panel_row.append(FloatSlider(value=item['value'], 
                                         min=item['min'],
                                         max = item['max'], 
                                         step = item['step'], 
                                         description=mathtext, 
                                         layout=Layout(width='95%')))
            
            sliders[item['keyword']+str(i+1)] = panel_row[-1]
        panel_col.append(HBox(panel_row, layout = Layout(width='100%')))
    layout = VBox(panel_col, layout = Layout(width='90%'))
    return sliders, layout


class dualSpectrumPlot:
    def __init__(self, ax, f_max, A_max=1, N=1):
        self.N = N
        self.ax = ax
        self.f_max =f_max
        self.A_max = A_max
        
        f_nd = np.outer([-f_max, f_max], np.ones(N))
        A_nd = np.zeros((2, self.N))
   
        self.lines = plt.plot(f_nd, A_nd, linewidth=2)
    
        self.ax.axis([-f_max, f_max, 0, A_max])
        self.ax.grid(True)
        self.ax.set_xlabel("Frekvens (Hz)")
    
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

# Frekvensmiksing      
class FrequencyMixingDemo:
    def __init__(self, fig_num=4, figsize=(12,6)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
        
        
        # Set up subplot with sine waves
        ax1 = plt.subplot(2, 1,1)
        ax1.set_title(r"Frekvensmiksing i tidsplan")
        
        self.t = np.linspace(0, 1, 501)
        self.SineWaves = timeSeriesPlot(ax1, self.t, A_max = 1,  N = 3)
        
        self.SineWaves.setStyles([{'color': 'tab:green', 'linestyle': ':'},
                                  {'color': 'tab:orange', 'linestyle': ':'},
                                  {'color': 'tab:blue'}])
        
        self.SineWaves.setLabels([r'$x_1(t) = \cos(2\pi \cdot f_1 \cdot t + \phi_1)$',
                                  r'$x_2(t) = \cos(2\pi \cdot f_2 \cdot t + \phi_2)$', 
                                  r'$y(t)=x_1(t)\cdot x_2(t)$'])
       # Set up subplot with spectrum
        ax2 = plt.subplot(2, 1,2)
        ax2.set_title(r"Frekvensmiksing i frekvensplan")
        
        self.Spectrum = dualSpectrumPlot(ax2, f_max=41, A_max = 1,  N = 3)
        
        self.Spectrum.setStyles([{'color': 'tab:green', 'linestyle': ':'},
                                  {'color': 'tab:orange', 'linestyle': ':'},
                                  {'color': 'tab:blue'}])
        
        self.Spectrum.setLabels([r'$x_1(t) = \cos(2\pi \cdot f_1 \cdot t + \phi_1)$',
                                  r'$x_2(t) = \cos(2\pi \cdot f_2 \cdot t + \phi_2)$', 
                                  r'$y(t)=x_1(t)\cdot x_2(t)$'])        
        


        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'F', 'value': 1, 'min': 0, 'max': 20, 'step': 1, 'description': r'f'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}],
            n_of_sets = 2)
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, F1, F2, phi1, phi2):

        x1 = cos(2*pi*F1*self.t + phi1*pi)
        x2 = cos(2*pi*F2*self.t + phi2*pi)
        y = x1 * x2
        
        self.SineWaves.update([x1, x2, y])
        f1_line, A1_line = getImpulseLines([-F1, F1],[0.5, 0.5], self.Spectrum.f_max)
        f2_line, A2_line = getImpulseLines([-F2, F2],[0.5, 0.5], self.Spectrum.f_max)
                                 
        if F1==F2:
            f3_line, A3_line = getImpulseLines(np.array([-F1-F2, 0, F1+F2]),
                                               np.array([0.25, 0.5*abs(np.cos(np.pi*(phi1-phi2))), 0.25]),
                                               self.Spectrum.f_max)
        elif F1 > F2:
            f3_line, A3_line = getImpulseLines(np.array([-F1-F2, -F1+F2, F1-F2, F1+F2]),
                                               np.array([0.25, 0.25, 0.25, 0.25]),
                                               self.Spectrum.f_max)
        else:
            f3_line, A3_line = getImpulseLines(np.array([-F1-F2, -F2+F1, F2-F1, F1+F2]),
                                               np.array([0.25, 0.25, 0.25, 0.25]),
                                               self.Spectrum.f_max)                
        self.Spectrum.update([f1_line, f2_line, f3_line],
                            [A1_line, A2_line, A3_line])
        
# Funksjoner og klassedefinisjon for fourier demonstrasjon.

def getDiamond(x, y, x_size=0.1, y_size=0.1):
    x_cross = np.array([1, 0, -1, 0, 1])*x_size+x
    y_cross = np.array([0, 1, 0, -1, 0])*y_size+y
    return [x_cross, y_cross]

def getCross(x, y, x_size=0.1, y_size=0.1):
    x_cross = np.array([1, 0, -1, 0, -1, 0, 1])*x_size+x
    y_cross = np.array([1, 0, 1, 0, -1, 0, -1])*y_size+y
    return [x_cross, y_cross]

# UI tool to inspect segments of a larger signal in time and frequency domain   
class FourierDemo:
    def __init__(self, x_t, f_s, max_k=10, fig_num=1):
        self.L = len(x_t)
        self.r_t = np.absolute(x_t)
        self.phi_t = np.angle(x_t)
        self.f_s = f_s
        self.t = np.linspace(0, 1, self.L, endpoint=False)
        self.mixer = 2*pi*self.t
        self.k = 0
        self.k_vect = np.arange(-max_k, max_k+ 1)
        self.animationSteps = np.linspace(1/20, 1, 20)
        self.a_k = []
        self.max_k = max_k
        self.coeff_win = np.array([-3, 3])
        
        for i in range(-max_k, max_k+1):
            self.a_k.append(np.mean(x_t*np.exp(-2j*pi*self.t*i)))
        self.a_k = np.array(self.a_k)
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(12, 8))
        
        
        # Set up plot showing signal
        self.ax1 = plt.subplot(2, 3, (1,5))
        self.ax1.set_title(r"$x(t)\cdot e^{-j\frac{2\pi\cdot 0}{T_0}\cdot t}$", fontsize=18)
        self.signalCurve, = self.ax1.plot(np.cos(self.phi_t)*self.r_t, np.sin(self.phi_t)*self.r_t,  color='tab:blue')
        self.ax1.grid(True)
        self.ax1.set_aspect(1)
        self.ax1.set_xlabel('Real axis')
        self.ax1.set_ylabel('Imaginary axis')
        axisLimit = max(self.r_t)
        self.ax1.axis([-axisLimit, axisLimit, -axisLimit, axisLimit])
        
        x, y = getCross(np.real(self.a_k[max_k]), np.imag(self.a_k[max_k]), 0.2, 0.2)
        self.coeffSquare, = self.ax1.plot(x, y, color='tab:red')
        
        # Set up plot showing magnitude spectrum
        self.ax2 = plt.subplot(2,3, 3)
        self.ampCoeffMarker, = self.ax2.plot([0, 0], [0, 10], ':', color='tab:red', linewidth=2)
        self.ax2.set_ylim([0, round(max(abs(self.a_k))+0.1,1)])
        self.ax2.set_xlim([-max_k, max_k])
        self.ax2.set_xticks(self.k_vect)
        self.ax2.set_xticklabels([r'$\frac{'+str(i)+'}{T_0}$' for i in self.k_vect])
        self.ax2.set_xlim(self.coeff_win)
        self.ax2.grid(True)
        self.ax2.set_ylabel(r'$A_k$')
        self.ax2.set_xlabel("Frekvens")

        self.ax2b = self.ax2.twiny()
        self.ampCoeff = self.ax2b.stem(self.k_vect, # Samplenummer
                                      np.absolute(self.a_k), # Signalverdier gitt samplenummer (x[n])
                                      linefmt='-C0', # Linjestil stolper
                                      markerfmt='.', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                                      basefmt='grey', # Farge på y=0 aksen
                                      use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                                      )
        self.ax2b.set_ylim(ymin=0)
        self.ax2b.set_xticks(self.k_vect)
        self.ax2b.set_xlim(self.coeff_win)
        self.ax2b.set_xlabel("Koeffisient-nummer 'k'")
        
        # Set up plot showing phase spectrum
        self.ax3 = plt.subplot(2,3, 6)
        self.angleCoeffMarker, = self.ax3.plot([0, 0], [-pi, pi], ':', color='tab:red', linewidth=2)
        self.ax3.set_ylim([-np.pi, np.pi])
        self.ax3.set_xticks(self.k_vect)
        self.ax3.set_xlim([-max_k, max_k])
        self.ax3.set_xticklabels([r'$\frac{'+str(i)+'}{T_0}$' for i in self.k_vect])
        self.ax3.set_yticks(np.linspace(-np.pi, np.pi, 5))
        self.ax3.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
        self.ax3.grid(True)
        self.ax3.set_ylabel(r'$\phi_k$')
        self.ax3.set_xlim(self.coeff_win)
        self.ax3.set_xlabel("Frekvens")
        
        self.ax3b = self.ax3.twiny()
        self.angleCoeff = self.ax3b.stem(np.arange(-max_k, max_k+1), # Samplenummer
                                         np.angle(self.a_k)*(np.absolute(self.a_k)>1e-6), # Signalverdier gitt samplenummer (x[n])
                                         linefmt='-C0', # Linjestil stolper
                                         markerfmt='.', # Punktstil for stem-markere. Default er 'o' (stor prikk)
                                         basefmt='grey', # Farge på y=0 aksen
                                         use_line_collection=True # Hvordan "stem" skal håndtere dataene. Bruk alltid True.
                                         )        
        self.ax3b.set_ylim([-np.pi, np.pi])
        self.ax3b.set_xticks(self.k_vect)
        self.ax3b.set_xlim(self.coeff_win)
        self.ax3b.set_xlabel("Koeffisient-nummer 'k'")

        
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
        
        # Set up UI panel
        coeff_k = widget.BoundedIntText(    
            value=0,
            min=-max_k,
            max=max_k,
            step=1,
            description="Antall Rotasjoner 'k':",
            disabled=False,
            style = {'description_width': 'initial'},
            #layout=Layout(width='95%'),
            continuous_update=False
        )
        self.layout = HBox(
            [coeff_k]
        )
        self.userInput = {
            'k': coeff_k
        }
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)
        # Run demo
        #out = interactive_output(self.update, self.sliders)
        #display(self.layout, out)
        #print("Hello")

        
    def update(self, k):
        self.coeffSquare.set_color('w')
        self.coeffSquare.set_linewidth(0)
      
        for i in (self.animationSteps*(k-self.k)+self.k):
            # Påfør rotasjon på kurven
            phi_t = self.phi_t-self.mixer*i
            self.signalCurve.set_xdata(np.cos(phi_t)*self.r_t)
            self.signalCurve.set_ydata(np.sin(phi_t)*self.r_t)
            
            # Forskyv amplitudevisningen til koeffisientene
            self.ax2b.set_xlim(self.coeff_win+i)
            
            # Forskyv fasevisningen til koeffisientene
            self.ax3b.set_xlim(self.coeff_win+i)
            
            # Oppdater figur
            self.fig.canvas.draw()
            
        titleStr = r"$x(t)\cdot e^{-j\frac{2\pi \cdot" +str(k)+"}{T_0}\cdot t}$"
        self.ax1.set_title(titleStr, fontsize=18)
        self.k = k
        
      
        x, y = getCross(np.real(self.a_k[self.max_k+self.k]), np.imag(self.a_k[self.max_k+self.k]), 0.2, 0.2)
        self.coeffSquare.set_color('tab:red')
        self.coeffSquare.set_linewidth(2)
        self.coeffSquare.set_xdata(x)
        self.coeffSquare.set_ydata(y)