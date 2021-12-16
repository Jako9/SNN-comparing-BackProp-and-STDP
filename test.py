# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import time

# @title Figure Settings
import ipywidgets as widgets       # interactive display

# use NMA plot style
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/master/nma.mplstyle")
my_layout = widgets.Layout()

# @title Helper functions

def default_pars_STDP(**kwargs):
  pars = {}

  # typical neuron parameters
  pars['V_th'] = -55.     # spike threshold [mV]
  pars['V_reset'] = -75.  # reset potential [mV]
  pars['tau_m'] = 10.     # membrane time constant [ms]
  pars['V_init'] = -65.   # initial potential [mV]
  pars['V_L'] = -75.      # leak reversal potential [mV]
  pars['tref'] = 2.       # refractory time (ms)

  # STDP parameters
  pars['A_plus'] = 0.008                   # magnitude of LTP
  pars['A_minus'] = pars['A_plus'] * 1.10  # magnitude of LTD
  pars['tau_stdp'] = 20.                   # STDP time constant [ms]

  # simulation parameters
  pars['T'] = 400.  # Total duration of simulation [ms]
  pars['dt'] = .1   # Simulation time step [ms]

  # external parameters if any
  for k in kwargs:
    pars[k] = kwargs[k]

  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])  # Vector of discretized time points [ms]

  return pars

def Poisson_generator(pars, rate, time_steps, neurons,myseed=False):
  """Generates poisson trains

  Args:
    pars            : parameter dictionary
    rate            : noise amplitute [Hz]
    time_steps      : number of time-steps simulated
    neurons         : number of neurons simulated
    myseed          : random seed. int or boolean

  Returns:
    pre_spike_train : spike train matrix, ith row represents whether
                      there is a spike in ith spike train over time
                      (1 if spike, 0 otherwise)
  """

  # Retrieve simulation parameters
  dt, range_t = pars['dt'], pars['range_t']
  Lt = range_t.size

  # set random seed
  if myseed:
    np.random.seed(seed=myseed)
  else:
    np.random.seed()

  # generate uniformly distributed random variables
  u_rand = np.random.rand(time_steps, neurons)

  # generate Poisson train
  poisson_train = 1. * (u_rand < rate * (dt / 1000.))

  return poisson_train


def mySTDP_plot(A_plus, A_minus, tau_stdp, time_diff, dW):
  plt.figure()
  plt.plot([-5 * tau_stdp, 5 * tau_stdp], [0, 0], 'k', linestyle=':')
  plt.plot([0, 0], [-A_minus, A_plus], 'k', linestyle=':')

  plt.plot(time_diff[time_diff <= 0], dW[time_diff <= 0], 'ro')
  plt.plot(time_diff[time_diff > 0], dW[time_diff > 0], 'bo')

  plt.xlabel(r't$_{\mathrm{pre}}$ - t$_{\mathrm{post}}$ (ms)')
  plt.ylabel(r'$\Delta$W', fontsize=12)
  plt.title('Biphasic STDP', fontsize=12, fontweight='bold')
  plt.show()

def Delta_W(pars, A_plus, A_minus, tau_stdp):
  """
  Plot STDP biphasic exponential decaying function
  Args:
    pars       : parameter dictionary
    A_plus     : (float) maxmimum amount of synaptic modification
                 which occurs when the timing difference between pre- and
                 post-synaptic spikes is positive
    A_plus     : (float) maxmimum amount of synaptic modification
                 which occurs when the timing difference between pre- and
                 post-synaptic spikes is negative
    tau_stdp   : the ranges of pre-to-postsynaptic interspike intervals
                 over which synaptic strengthening or weakening occurs
  Returns:
    dW         : instantaneous change in weights
  """

  # STDP change
  dW = np.zeros(len(time_diff))
  # Calculate dW for LTP
  dW[time_diff <= 0] = A_plus * np.exp(time_diff[time_diff <= 0] / tau_stdp)
  # Calculate dW for LTD
  dW[time_diff > 0] = -A_minus * np.exp(-time_diff[time_diff > 0] / tau_stdp)

  return dW

def generate_potential(pars, pre_spike_train_ex,pre=True):
  """
  track potential of neurons
  Args:
    pars               : parameter dictionary
    pre_spike_train_ex : binary spike train input from
                         presynaptic excitatory neuron
  Returns:
    P                  : LTP ratio
  """

  # Get parameters
  A_plus, A_minus, tau_stdp, dt = pars['A_plus'], pars['A_minus'], pars['tau_stdp'], pars['dt']
  # Initialize
  P = np.zeros(pre_spike_train_ex.shape)
  for it in range(pre_spike_train_ex.shape[0] - 1):
    # Calculate the delta increment dP
    if pre:
        dP = -(dt / tau_stdp) * P[it, :] + A_plus * pre_spike_train_ex[it+1, :]
    else:
        dP = -(dt / tau_stdp) * P[it, :] - A_minus * pre_spike_train_ex[it+1, :]
    # Update P
    P[it+1, :] = P[it, :] + dP

  return P

def main():
    pars = default_pars_STDP(T=200., dt=1.)
    pre_spike_train_ex = Poisson_generator(pars, rate=100, time_steps=200,neurons = 1000, myseed=2020)
    post_neuron = Poisson_generator(pars, rate=100, time_steps=200,neurons = 1, myseed=2020)
    print(pre_spike_train_ex)
    pre_pot = generate_potential(pars, pre_spike_train_ex,pre=True)
    print(pre_pot)
    post_pot = generate_potential(pars, post_neuron,pre=False)
    print(post_pot)

if __name__ == '__main__':
    main()
