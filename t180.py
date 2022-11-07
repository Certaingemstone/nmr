import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class SpinEnsemble:
    '''Ideal spin ensemble with no diffusion or inhomogeneity
    Incorporates spin-lattice (t1) and spin-spin (t2) effects'''
    def __init__(self, t1, t2):
        self.mz_eq = 1.
        self.mz = 1.
        self.mxy = 0
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.z_history = []
        self.xy_history = []
    
    def record(self):
        self.z_history.append(self.mz)
        self.xy_history.append(self.mxy)
        
    def pulse(self, theta):
        '''Rotate spin ensemble by angle'''
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        res = np.matmul(R, np.array((self.mz, self.mxy)))
        self.mz = res[0]
        self.mxy = res[1]
        
    def relax(self, time):
        '''Evolve self.m via relaxation processes'''
        self.mz = self.mz_eq - (self.mz_eq - self.mz) * np.exp(-time/self.t1)
        self.mxy = self.mxy * np.exp(-time/self.t2)
        
    def multi_relax(self, n, time=1, record=True):
        for i in range(n):
            self.relax(time)
            if record:
                self.record()
        
    def plot(self, ax):
        sizes = [ x for x in np.linspace(1, 16, len(self.z_history)) ]
        colors = [ x for x in np.linspace(0, len(self.z_history), len(self.z_history)) ]
        ax.scatter(self.xy_history, self.z_history, s=sizes, c=colors, cmap=cm.plasma)
        ax.axhline(0, color='k')
        ax.axvline(0, color='k')
        ax.set_xlabel("Transverse Magnetization")
        ax.set_ylabel("Longitudinal Magnetization")
        ax.set_title("Simulated NMR Response")

if __name__ == "__main__":
    
    fig, ax = plt.subplots()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    T1 = 20
    T2 = 60
    t180 = np.pi * 1.1
    t90 = t180/2

    ens = SpinEnsemble(T1, T2)

    ens.record()
    ens.pulse(t90)
    ens.record()
    ens.multi_relax(8, record=True)
    ens.pulse(t180)
    ens.record()

    for i in range(10):
        ens.multi_relax(16, record=True)
        ens.pulse(t180)
        ens.record()
        
    #ens.multi_relax(5, time=20, record=True)

    ens.plot(ax)
    plt.show()
