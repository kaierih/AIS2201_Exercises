from numpy import sin, cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout


def getArrow(x, y, dx, dy, arrowhead_scale=1):
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy,dx)
    len_arrowhead = min(arrowhead_scale/16, r/2)
    x_arrow = np.array([x, x+dx, x+dx+len_arrowhead*cos(theta-4*pi/5), x+dx, x+dx+len_arrowhead*cos(theta+4*pi/5)])
    y_arrow = np.array([y, y+dy, y+dy+len_arrowhead*sin(theta-4*pi/5), y+dy, y+dy+len_arrowhead*sin(theta+4*pi/5)])
    return x_arrow, y_arrow

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

class vectorPlot:
    def __init__(self, ax, A_max, N=1):
        self.ax = ax
        self.N = N
        init_values = np.zeros((2, N))
        self.lines = self.ax.plot(init_values, init_values)
        self.ax.grid(True)
        self.ax.set_xlabel("Reell akse")
        self.ax.set_ylabel("Imaginær akse")
        self.ax.axis([-A_max, A_max, -A_max, A_max])
        
    def update(self, x_new_lines, y_new_lines):
        assert len(x_new_lines)==len(y_new_lines)==self.N, 'Error: mismatch between x and y dimensions.'
        for i in range(self.N):
            x_line = x_new_lines[i]
            y_line = y_new_lines[i]
            L = len(x_line)
            assert len(y_line)==L, 'Error: mismatch between x and y dimensions.'
            x_arrows = np.zeros((L-1)*5)
            y_arrows = np.zeros((L-1)*5)
            for j in range(1, L):
                b = j*5
                a = b-5
                x_arrows[a:b], y_arrows[a:b] = getArrow(x_line[j-1], y_line[j-1], x_line[j]-x_line[j-1], y_line[j]-y_line[j-1])
            self.lines[i].set_xdata(x_arrows)
            self.lines[i].set_ydata(y_arrows)
            
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
        
# Demo 4:
# Sum av sinusbølger med vektoraddisjon        
class VectorSumDemo():
    def __init__(self, fig_num=1, fig_size=(9, 4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        
        # Set up subplot with sine waves
        ax1 = plt.subplot(1, 5, (1,3))
        ax1.set_title(r"Sum av sinusbølger med frekvens $f=1Hz$")
        
        self.t = np.linspace(0, 2, 501)
        self.SineWaves = timeSeriesPlot(ax1, self.t, A_max = 2,  N = 3)
        
        self.SineWaves.setStyles([{'color': 'tab:green', 'linestyle': '-.'},
                                  {'color': 'tab:orange', 'linestyle': '-.'},
                                  {'color': 'tab:blue'}])
        
        self.SineWaves.setLabels([r'$x_1(t) = A_1\cdot \cos(2\pi t + \phi_1)$',
                                  r'$x_2(t) = A_2\cdot \cos(2\pi t + \phi_2)$', 
                                  r'$y(t)=x_1(t)+x_2(t)$'])

        
        # Set up vector subplot
        ax2 = plt.subplot(1, 5, (4,5))
        ax2.set_title("Kompleks amplitude $a_k = A_ke^{j\phi_k}$")
        ax2.set_aspect(1)
        
        self.VectorSumPlot = vectorPlot(ax2, A_max = 2, N = 3)
        
        self.VectorSumPlot.setStyles([{'color': 'tab:green', 'linestyle': '-.'},
                                      {'color': 'tab:orange', 'linestyle': '-.'},
                                      {'color': 'tab:blue'}])
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'A', 'value': 1, 'min': 0, 'max': 1, 'step': 0.1, 'description': r'A'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}],
            n_of_sets = 2)
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x1 = kwargs['A1']*cos(2*pi*self.t + kwargs['phi1']*pi)
        x2 = kwargs['A2']*cos(2*pi*self.t + kwargs['phi2']*pi)
        y = x1 + x2
        
        self.SineWaves.update([x1, x2, y])
        
        v1_x = np.array([0, kwargs['A1']*cos(kwargs['phi1']*pi)])
        v1_y = np.array([0, kwargs['A1']*sin(kwargs['phi1']*pi)])
        
        v2_x = np.array([0, kwargs['A2']*cos(kwargs['phi2']*pi)])+v1_x[-1]
        v2_y = np.array([0, kwargs['A2']*sin(kwargs['phi2']*pi)])+v1_y[-1]
        
        v3_x = np.array([0, v2_x[-1]])
        v3_y = np.array([0, v2_y[-1]])
        
        self.VectorSumPlot.update([v1_x, v2_x, v3_x], [v1_y, v2_y, v3_y])
              
        

# Demo 6
# Visualisering av frekvensmiksing
class FrequencyMixingDemo():
    def __init__(self, fig_num=1, fig_size = (9, 4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        # Set up subplot with sine wave 1
        ax1 = plt.subplot(2,2,1)
        ax1.set_title(" ")
        
        self.t_x = np.linspace(0, 1, 201)
        self.SineWave1 = timeSeriesPlot(ax1, self.t_x, A_max = 1.2)
        
        # Set up subplot with sine wave 2 
        ax2 = plt.subplot(2,2,2)
        ax2.set_title(" ")
        
        self.SineWave2 = timeSeriesPlot(ax2, self.t_x, A_max = 1.2)
        
        # Set up subplot with product 
        ax3 = plt.subplot(2,2,(3,4))
        ax3.set_title(r"$y(t) = x_1(t)\cdot x_2(t)$")
        
        self.t_y = np.linspace(0, 2, 401)
        self.MixedWaves = timeSeriesPlot(ax3, self.t_y, A_max = 1.2)
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
        
        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'f', 'value': 1, 'min': 0.5, 'max': 15, 'step': 0.5, 'description': r'f'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}],
            n_of_sets=2)
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x1 = cos(2*pi*self.t_y*kwargs['f1'] + kwargs['phi1']*pi)
        x2 = cos(2*pi*self.t_y*kwargs['f2'] + kwargs['phi2']*pi)
        
        y = x1*x2
        
        titleStr1 = "$x_1(t)=\cos(2\pi\cdot"+str(kwargs['f1'])+"\cdot t +"+str(round(kwargs['phi1'],2))+"\pi)$" # Plot-tittel
        titleStr1 = titleStr1.replace("+-", "-")
        self.SineWave1.ax.set_title(titleStr1)
        self.SineWave1.update([x1[0:201]])
        
        titleStr2 = "$x_2(t)=\cos(2\pi\cdot"+str(kwargs['f2'])+"\cdot t +"+str(round(kwargs['phi2'],2))+"\pi)$" # Plot-tittel
        titleStr2 = titleStr2.replace("+-", "-")
        self.SineWave2.ax.set_title(titleStr2)
        self.SineWave2.update([x2[0:201]])
        
        self.MixedWaves.update([y])

# Frekvensmiksing      
class FrequencyMixingSpectrumDemo:
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