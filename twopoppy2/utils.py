import numpy as np
import warnings

from scipy.constants import golden as gr
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

from .constants import au, year


def get_powerlaw_dust_distribution(sigma_d, a_max, q=3.5, na=10, a0=None, a1=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array

    a_max : array
        maximum particle size array

    Keywords:
    ---------

    q : float
        particle size index, n(a) propto a**-q

    na : int
        number of particle size bins

    a0 : float
        minimum particle size

    a1 : float
        maximum particle size

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution of size (len(sigma_d), na)
    """

    if a0 is None:
        a0 = a_max.min()

    if a1 is None:
        a1 = 2 * a_max.max()

    nr = len(sigma_d)
    sig_da = np.zeros([nr, na]) + 1e-100

    a_i = np.logspace(np.log10(a0), np.log10(a1), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    for ir in range(nr):

        if a_max[ir] <= a0:
            sig_da[ir, 0] = 1
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            for ia in range(i_up):
                sig_da[ir, ia] = a_i[ia + 1]**(4 - q) - a_i[ia]**(4 - q)

            # filling the bin that contains a_max
            sig_da[ir, i_up] = a_max[ir]**(4 - q) - a_i[i_up]**(4 - q)

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da


def solution_youdinshu2002(r, sig0, t, A, d):
    """
    An implementation of the analytical solution of Youdin & Shu, 2002
    for a any dust surface density profile with a power-law gas and
    temperature profile if the velocity can be written as v = - A * r^d.
    Usually, for drift and a fixed size, one gets d = p-q+1/2

    Arguments:
    ----------

    r : array
        radial grid

    sig0 : dust surface density, initial condition on grid r
    t    = time of the solution
    A    = velocity offset (see above)
    d    = velocity exponent (see above)

    Returns:
    analytical solution on r at time t
    """

    # get the initial position of each radius

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        if d == 1:
            ri = r * np.exp(-A * r**(d - 1.) * t)
        else:
            ri = (r**(1. - d) - A * t * (1 - d))**(1. / (1. - d))

    # cut off unphysical part of solution

    ri[r**(1 - d) <= A * t * (1 - d)] = 2 * r[-1]

    # get the initial condition at
    # ri, take a floor value for extrapolation

    sig_i = 10**np.interp(np.log10(ri), np.log10(r), np.log10(sig0), left=-100, right=-100)

    # return the solution

    return r**(-d - 1) * ri**(d + 1) * sig_i


def lbp_solution(R, gamma, nu1, mstar, mdisk, RC0, time=0):
    """
    Calculate Lynden-Bell & Pringle self similar solution.
    All values need to be either given with astropy-units, or
    in as pure float arrays in cgs units.

    Arguments:
    ----------

    R : array,  radius array
    
    gamma : float, viscosity exponent

    nu1 : float, viscosity at R[0]

    mstar : float, stellar mass

    mdisk : float, disk mass at t=0

    RC0 : float, critical radius at t=0

    Keywords:
    ---------

    time : float, physical "age" of the analytical solution

    Output:
    -------
    sig_g,RC(t)

    sig_g : array
    gas surface density, with or without unit, depending on input

    RC : the critical radius

    """
    import astropy.units as u
    import numpy as np

    # assume cgs if no units are given

    units = True
    if not hasattr(R, 'unit'):
        R = R * u.cm
        units = False
    if not hasattr(mdisk, 'unit'):
        mdisk = mdisk * u.g
    if not hasattr(mstar, 'unit'):
        mstar = mstar * u.g
    if not hasattr(nu1, 'unit'):
        nu1 = nu1 * u.cm**2 / u.s
    if not hasattr(RC0, 'unit'):
        RC0 = RC0 * u.cm
    if time is None:
        time = 0
    if not hasattr(time, 'unit'):
        time = time * u.s

    # convert to variables as in Hartmann paper

    R1 = R[0]
    r = R / R1
    ts = 1. / (3 * (2 - gamma)**2) * R1**2 / nu1

    T0 = (RC0 / R1)**(2. - gamma)
    toff = (T0 - 1) * ts

    T1 = (time + toff) / ts + 1
    RC1 = T1**(1. / (2. - gamma)) * R1

    # the normalization constant

    C = (-3 * mdisk * nu1 * T0**(1. / (4. - 2. * gamma)) * (-2 + gamma)) / 2. / R1**2

    # calculate the surface density

    sig_g = C / (3 * np.pi * nu1 * r**gamma) * T1**(-(5. / 2. - gamma) / (2. - gamma)) * np.exp(-(r**(2. - gamma)) / T1)

    if units:
        return sig_g, RC1
    else:
        return sig_g.cgs.value, RC1.cgs.value


def t_acc0(sim, r_c0=60):

    r = sim.r
    r_c0 = r_c0 * au
    
    alpha = sim.alpha_gas
    alpha_dw = sim.alpha_dw
    alpha_tilda = alpha + alpha_dw

    r_c0_ind = np.argmin(np.abs(r-r_c0))
    
    cs = sim.cs
    hp = sim.hp
    asp = sim.hp/sim.r
    
    
    t_acc0  = r_c0/ (3*cs[r_c0_ind]*asp[r_c0_ind]* alpha_tilda)
    return t_acc0

def r_ct_func(sim, r_c0=60, t=0):

    r_c0 = r_c0 * au
    alpha = sim.alpha_gas
    alpha_dw = sim.alpha_dw
    alpha_phi = alpha_dw/alpha
    return r_c0 * (1+ t/((1+alpha_phi)*t_acc0(sim)))

def Tabone22_analytical(sim, r_c0 = 60, mode='wind', t=0):
    '''mode can be wind/ hybrid/ sigma_dep
    
    input:
    --------
    r_c0: float, initial characteristic radius;

    
    '''
    
    M_0 = (np.pi * np.diff(sim.ri**2) * sim.data['sigma_g'][0]).sum()  # inital disc mass 
    #r_c0 = sim.ini.gas.SigmaRc
    r_c0 = r_c0 * au
    alpha = sim.alpha_gas
    alpha_dw = sim.alpha_dw
    leverarm = sim._leverarm
    omega = 0 # describe how the magnetic field changes with time

    if mode != 'wind':
        #alpha_tilda = alpha + alpha_dw
        alpha_phi = alpha_dw/alpha
    else:
        pass
    ksi =  1/(2*(leverarm-1)) * (alpha_dw/(alpha_dw+alpha)) # when lambda>=2
    #nu = sim._nu
    r = sim.r
    #cs = sim.cs
    #hp = sim.hp
    #asp = sim.hp/sim.r

    #r_c0_ind = np.argmin(np.abs(r-r_c0))
    t_acc  = t_acc0(sim, r_c0 = r_c0) #r_c/ (3*cs[r_c_ind]*asp[r_c_ind]* alpha_tilda[0]) # here we assume alpha and alpha_dw are constants
    r_ct   = r_ct_func(sim,r_c0 = r_c0, t=t)

    if mode == 'wind':
        sigma = M_0/(2*np.pi*r_c0**2) * np.exp(-t/2/t_acc) * (r/r_c0)**(-1+ksi) * np.exp(-r/r_c0) 
    elif mode == 'hybrid':
        sigma = M_0/(2*np.pi*r_c0**2) * (1 + t/((1+alpha_phi)* t_acc)) ** (-1/2*(alpha_phi+2*ksi+5))  * (r/r_ct)**(-1+ksi) * np.exp(-r/r_ct)# here we assume alpha and alpha_dw are constants 
    elif mode == 'sigma_dep':
        sigma = M_0/(2*np.pi*r_c0**2) * (1- omega * t/2/t_acc) ** (1/omega) * (r/r_ct)**(-1+ksi) * np.exp(-r/r_ct)
    return sigma





class Widget():

    def __init__(self, m):
        self.data = m.data
        self.height = 3
        f, axs = plt.subplots(1, 2, figsize=(2 * self.height * gr, self.height * 1.2), sharex=True)
        self.f = f
        self.axs = axs
        f.subplots_adjust(bottom=0.2)

        # left plot

        self.line_d0, = self.axs[0].loglog(m.r / au, m.data['sigma_g'][0], c='0.4', ls='-')
        self.line_d1, = self.axs[0].loglog(m.r / au, m.data['sigma_g'][0], c='C0', ls='-', label='gas')

        self.line_g0, = self.axs[0].loglog(m.r / au, m.data['sigma_d'][0], c='0.8', ls='-')
        self.line_g1, = self.axs[0].loglog(m.r / au, m.data['sigma_d'][0], c='C1', ls='-', label='dust')

        self.axs[0].plot([], [], '0.4', label='0 years')
        self.axs[0].set_ylim(1e-4, 1e4)
        self.axs[0].legend()

        # right plot

        self.line_a_1, = self.axs[1].loglog(m.r / au, m.data['a_1'][0])
        self.line_afr, = self.axs[1].loglog(m.r / au, m.data['a_fr'][0])
        self.line_adr, = self.axs[1].loglog(m.r / au, m.data['a_dr'][0])
        self.axs[1].set_ylim(9e-6, 1e2)

        # time slider

        pos = self.axs[0].get_position()
        self.time_slider_ax = self.f.add_axes([pos.x0, 0.05, pos.width, 0.03])
        self.time_slider = Slider(
            self.time_slider_ax, 'time', 0, len(m.data['time']), valinit=0,
            valstep=1, color='turquoise', valfmt='%d')
        self.time_slider.on_changed(self.update)

    def update(self, val):
        it = int(np.floor(self.time_slider.val))

        # left plot

        self.time_slider_ax.set_title('time = {:.2g}'.format(self.data['time'][it][0] / year), fontsize='small')
        self.line_g1.set_ydata(self.data['sigma_g'][it])
        self.line_d1.set_ydata(self.data['sigma_d'][it])

        # right plot

        self.line_a_1.set_ydata(self.data['a_1'][it])
        self.line_afr.set_ydata(self.data['a_fr'][it])
        self.line_adr.set_ydata(self.data['a_dr'][it])


def get_interfaces_from_log_cell_centers(x):
    """
    Returns the cell interfaces for an array of logarithmic
    cell centers.

    Arguments:
    ----------

    x : array
    :   Array of logarithmically spaced cell centers for
        which the interfaces should be calculated

    Output:
    -------

    xi : array
    :    Array of length len(x)+1 containing the grid interfaces
         defined such that 0.5*(xi[i]+xi[i+1]) = xi
    """
    x = np.asarray(x)
    B = x[1] / x[0]
    A = (B + 1) / 2.
    xi = np.append(x / A, x[-1] * B / A)
    return xi


def get_interfaces_from_lin_cell_centers(x):
    """
    Returns the cell interfaces for an array of linear
    cell centers.

    Arguments:
    ----------

    x : array
    :   Array of logarithmically spaced cell centers for
        which the interfaces should be calculated

    Output:
    -------

    xi : array
    :    Array of length len(x)+1 containing the grid interfaces
         defined such that 0.5 * (xi[i] + xi[i + 1]) = xi
    """
    x  = np.asarray(x)
    dx = np.diff(x).mean()
    xi = np.append(x[0] - dx / 2, x + dx / 2)
    return xi
