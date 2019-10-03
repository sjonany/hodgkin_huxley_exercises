import scipy as sp
import pylab as plt
from scipy.integrate import odeint

# Original code is from https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Exercises.html
class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    def __init__(self):
        self.C_m  =   1.0
        """membrane capacitance, in uF/cm^2"""

        self.g_Na = 120.0
        """Sodium (Na) maximum conductances, in mS/cm^2"""

        self.g_K  =  36.0
        """Postassium (K) maximum conductances, in mS/cm^2"""

        self.g_L  =   0.3
        """Leak maximum conductances, in mS/cm^2"""

        self.E_Na =  50.0
        """Sodium (Na) Nernst reversal potentials, in mV"""

        self.E_K  = -77.0
        """Postassium (K) Nernst reversal potentials, in mV"""

        self.E_L  = -54.387
        """Leak Nernst reversal potentials, in mV"""

        self.t = sp.arange(0.0, 450.0, 0.01)
        """ The time to integrate over """

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_K  * n**4 * (V - self.E_K)
    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        |  :param t: time
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X

        dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def gen_plots(self):
        init_states = [-65, 0.05, 0.6, 0.32]
        X = odeint(self.dALLdt, init_states, self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)


        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,10))

        ax[0].plot(self.t, V, 'k')
        ax[0].set_title('Hodgkin-Huxley Neuron')
        ax[0].set_ylabel('V (mV)')

        ax[1].plot(self.t, ina, 'c', label='$I_{Na}$')
        ax[1].plot(self.t, ik, 'y', label='$I_{K}$')
        ax[1].plot(self.t, il, 'm', label='$I_{L}$')
        ax[1].set_ylabel('Current')
        ax[1].legend()

        ax[2].plot(self.t, m, 'r', label='m')
        ax[2].plot(self.t, h, 'g', label='h')
        ax[2].plot(self.t, n, 'b', label='n')
        ax[2].set_ylabel('Gating Value')
        ax[2].legend()

        i_inj_values = [self.I_inj(t) for t in self.t]
        ax[3].plot(self.t, i_inj_values, 'k')
        ax[3].set_xlabel('t (ms)')
        ax[3].set_ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        ax[3].set_ylim(-1, 40)

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
