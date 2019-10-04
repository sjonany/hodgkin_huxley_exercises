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

        self.I_inj = lambda t: 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)
        """
        External Current

        |  :param t: time (ms)
        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """

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

    @staticmethod
    def count_peaks(V):
      peak_count = 0
      for t in sp.arange(1, len(V)-1, 1):
        if V[t] > V[t-1] and V[t] > V[t+1]:
          peak_count += 1
      return peak_count

    def gen_plots(self, plots_to_include):
        init_states = [-65, 0.05, 0.6, 0.32]
        X = odeint(self.dALLdt, init_states, self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        self.V = V
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        i_inj_values = [self.I_inj(t) for t in self.t]

        fig, axes = plt.subplots(nrows=len(plots_to_include), ncols=1,
          figsize=(10, 3 * len(plots_to_include)), sharex=True)

        for i in range(len(plots_to_include)):
          plot_command = plots_to_include[i]
          if len(plots_to_include) == 1:
            ax = axes
          else:
            ax = axes[i]

          if plot_command == "V":
            ax.plot(self.t, V, 'k')
            ax.set_title('Hodgkin-Huxley Neuron')
            ax.set_ylabel('V (mV)')
          elif plot_command == "I":
            ax.plot(self.t, ina, 'c', label='$I_{Na}$')
            ax.plot(self.t, ik, 'y', label='$I_{K}$')
            ax.plot(self.t, il, 'm', label='$I_{L}$')
            ax.set_ylabel('Current')
            ax.legend()
          elif plot_command == "gate":
            ax.plot(self.t, m, 'r', label='m')
            ax.plot(self.t, h, 'g', label='h')
            ax.plot(self.t, n, 'b', label='n')
            ax.set_ylabel('Gating Value')
            ax.legend()
          else:
            ax.plot(self.t, i_inj_values, 'k')
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
            ax.set_ylim(-1, 40)

if __name__ == '__main__':
    runner = HodgkinHuxley()
    runner.Main()
