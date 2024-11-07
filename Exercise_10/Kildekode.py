import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt 

def tfPlot(b, a, ax=None):
    if ax == None:
        ax = plt.axes(projection='3d')
    res=122
    x = np.linspace(-1.2, 1.2, res)
    y = np.linspace(-1.2, 1.2, res)
    x,y = np.meshgrid(x,y)
    z = x + 1j*y
    Bz = np.zeros((res, res))*1j
    for i in range(len(b)):
        Bz += b[i]/(z**i)
    Az = np.zeros((res, res))*1j
    for i in range(len(a)):
        Az += a[i]/(z**i)
    Hz = Bz/Az
    
    ax.set_xlabel(r'Re(z)')
    ax.set_xlim(-1.2,1.2)
    ax.set_ylabel(r'Im(z)')
    ax.set_ylim(-1.2,1.2)
    ax.set_zlim([-60, 40])

    plt.title(r'$\left| H(z) \right|$ in Z-plane')
    
    ax.plot_surface(x, y, 20*np.log10(np.abs(Hz)), rstride=1, cstride=1, cmap='viridis', edgecolor='none' )
    w, Hw = sig.freqz(b, a, worN=509, whole=True)
    x_w = np.cos(w)
    y_w = np.sin(w)
    ax.plot(x_w, y_w, 20*np.log10(np.abs(Hw)), linewidth=3, color='tab:red')
    
def pzPlot(b, a):
    zeroes, poles, k = sig.tf2zpk(b, a)

    plt.plot(np.real(zeroes), np.imag(zeroes),'C0o', markersize=8, linewidth=0, markerfacecolor='none')
    plt.plot(np.real(poles), np.imag(poles),'C0x', markersize=8, linewidth=0, markerfacecolor='none')

    plt.grid(True)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.plot(np.cos(np.linspace(0, 2*np.pi, 513)), np.sin(np.linspace(0, 2*np.pi, 513)),'C3:')
    plt.title('Pole zero map')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    
def Magnitude_dB(b, a):
    w, Hw = sig.freqz(b, a, worN=509, whole=True)

    plt.plot(w, 20*np.log10(np.abs(Hw)))
    plt.grid(True)
    plt.ylim(ymin=-40)
    plt.xlim([0, np.pi])
    plt.title('Magnitude Response')
    plt.xticks(np.linspace(0, 1, 5)*np.pi, [r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
    plt.xlabel(r'Digital Frequency $\hat{\omega}$')
    plt.ylabel(r'Magnitude (dB)')

    
def visualizeTF(b, a ,fig_num=None, figsize=(16,9)):
    plt.close(fig_num)
    if fig_num:
        
        fig = plt.figure(fig_num, figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)
    
    ax = plt.subplot(2,3,(2,6),projection = '3d')
    tfPlot(b, a, ax)

    plt.subplot(2,3,1) 
    plt.gca().set_aspect('equal')
    pzPlot(b, a)
    
    plt.subplot(2,3,4)   
    Magnitude_dB(b, a)
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)