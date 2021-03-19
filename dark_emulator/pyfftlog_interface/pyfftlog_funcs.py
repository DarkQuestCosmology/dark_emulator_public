from .pyfftlog_class import fftlog
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def get_arr(func, x, args):
    """getting arr from func with args

    Args:
        funcs (function): function with input x and args
        x (ndarray): input x for funcs
        args (tuple): tuple of arguments for funcs

    Returns:
        arr (ndarray): array of func(x, args).

    """
    if not isinstance(args, tuple):
        args = (args,)
    if len(args) == 0:
        arr = func(x)
    else:
        arr = func(x, *args)
    return arr

def pk2xi_pyfftlog(pk_func, num=1, logkmin=-3.0, logkmax=3.0, args=()):
    """converting P(k) to xi(r)

    Converting 3-dimensional power spectrum into 3-dimensional correlation function, defined as

    .. math::
        \\xi(r) = \\int \\frac{{\\rm d}^{3}k}{(2\\pi)^3} P(k) e^{i k\cdot r}
                = \\frac{1}{(2\\pi r)^{3/2}} \\int r{\\rm d}k [k^{3/2}P(k)] J_{1/2}(kr)

    Args:
        pk_func (function): function of 3d power spectrum
        num (int, optional): binning number of fftlog in unit of 2048
        logkmin (float): Minimum k over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logkmax (float): Minimum k over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for pk_func

    Returns:
        xi (function): function of xi

    """
    _fftlog = fftlog(num, -logkmax, -logkmin)
    pk = get_arr(pk_func, _fftlog.k, args)
    return ius( _fftlog.r, _fftlog.pk2xi(pk) )

def pk2wp_pyfftlog(pk_func, num=1, logkmin=-4.0, logkmax=5.0, args=()):
    """converting P(k) to wp(r)

    Converting 3-dimensional power spectrum into 2-dimensional projected correlation function, defined as

    .. math::
        w_{\\rm p}(r) = \\int_0^{\\inf} {\\rm d}\\Pi \\xi(\\sqrt{r^2+\\Pi})
                          = \\frac{1}{2\\pi r} \\int r{\\rm d}k [kP(k)] J_{0}(kr).

    Args:
        pk_func (function): function of 3d power spectrum
        num (int, optional): binning number of fftlog in unit of 2048
        logkmin (float): Minimum k over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logkmax (float): Minimum k over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for pk_func

    Returns:
        wp (function): function of wp

    """
    _fftlog = fftlog(num, -logkmax, -logkmin)
    dump = np.exp(-(_fftlog.k/1e4)**2) # high k (>1e4) dump
    pk = get_arr(pk_func, _fftlog.k, args) * dump
    return ius( _fftlog.r, _fftlog.pk2wp(pk) )
pk2xiproj_J0_pyfftlog = pk2wp_pyfftlog

def pk2dwp_pyfftlog(pk_func, num=1, logkmin=-3.0, logkmax=5.0, args=()):
    """converting P(k) to dwp(r)

    Converting 3-dimensional power spectrum into 2-dimensional excess projected correlation function, defined as

    .. math::
        \\Delta w_{\\rm p}(r) = \\bar{w_{\\rm p}}(r) - w_{\\rm p}(r)
                           = \\frac{1}{2\\pi r} \\int r{\\rm d}k [kP(k)]J_2(kr).

    Args:
        pk_func (function): function of 3d power spectrum
        num (int, optional): binning number of fftlog in unit of 2048
        logkmin (float): Minimum k over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logkmax (float): Minimum k over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for pk_func

    Returns:
        dwp (function): function of dwp

    """
    _fftlog = fftlog(num, -logkmax, -logkmin)
    dump = np.exp(-(_fftlog.k/1e4)**2) # high k (>1e4) dump
    pk = get_arr(pk_func, _fftlog.k, args) * dump
    return ius( _fftlog.r, _fftlog.pk2dwp(pk) )
pk2xiproj_J2_pyfftlog = pk2dwp_pyfftlog

def xi2pk_pyfftlog(xi_func, num=1, logrmin=-3.0, logrmax=3.0, args=()):
    """converting P(k) to wp(r)

    Converting 3-dimensional correlation function into 3-dimensional power spectrum, defined as

    .. math::
        P(k) = \\int {\\rm d}^{3}r \\xi(r) e^{-i k\cdot r}
                 = \\frac{(2\\pi)^{3/2}}{k^{3/2}} \\int r{\\rm d}k [r^{3/2}\\xi(r)] J_{1/2}(kr)

    Args:
        xi_func (function): function of 3d power spectrum
        num (int, optional): binning number of fftlog in unit of 2048
        logrmin (float): Minimum r over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logrmax (float): Minimum r over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for xi_func

    Returns:
        pk (function): function of pk

    """
    _fftlog = fftlog(num, logrmin, logrmax)
    xi = get_arr(xi_func, _fftlog.r, args)
    return ius( _fftlog.k, _fftlog.xi2pk(xi) )

def HT_pyfftlog(Ar_func, mu, q, beta, prefactor, num=1, logrmin=-3.0, logrmax=3.0, args=()):
    """Hankel transform

    Executing Hankel transformation, defined as

    .. math::
        A(k) = p k^{\\beta} \\int_0^{\\infty} k {\\rm d}r A(r) (kr)^{q} J_{\\mu}(kr).

    where :math:`p` is arbitrary prefactor.

    Args:
        Ar_func (function): function of :math:`A(r)`
        mu (float): Order of the first kind of Bessel function, :math:`J_\\mu`.
        q (float): Bias parameter in FFTLog
        beta (float): Power index of k to be multiplied to output
        prefactor (float, ndarray): Constant prefactor to be multiplied to output
        num (int, optional): binning number of fftlog in unit of 2048
        logrmin (float): Minimum r over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logrmax (float): Minimum r over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for xi_func

    Returns:
        Ak (function): function of :math:`A(k)`, which is Hankel transformed :math:`A(r)`.

    """
    _fftlog = fftlog(num, logrmin, logrmax)
    Ar = get_arr(Ar_func, _fftlog.r, args)
    return ius( _fftlog.k, _fftlog.HT(Ar, mu, q, beta=beta, prefactor=prefactor) )

def iHT_pyfftlog(Ak_func, mu, q, beta, prefactor, num=1, logrmin=-3.0, logrmax=3.0, args=()):
    """inverse Hankel transform

    Executing inverse Hankel transformation, defined as

    .. math::
        A(r) = p r^{\\beta} \\int_0^{\\infty} r {\\rm d}k A(k) (kr)^{q} J_{\\mu}(kr).

    where :math:`p` is arbitrary prefactor.

    Args:
        Ak_func (function): function of :math:`A(k)`
        mu (float): Order of the first kind of Bessel function, :math:`J_\\mu`.
        q (float): Bias parameter in FFTLog
        beta (float): Power index of k to be multiplied to output
        prefactor (float, ndarray): Constant prefactor to be multiplied to output
        num (int, optional): binning number of fftlog in unit of 2048
        logrmin (float): Minimum r over which scale fftlog will be executed. Default choice, -3.0, is for darkemu.
        logrmax (float): Minimum r over which scale fftlog will be executed. Default choice, 3.0, is for darkemu.
        args (tuple): tuple of arguments for xi_func

    Returns:
        Ar (function): function of :math:`A(r)`, which is Hankel transformed :math:`A(r)`.

    """
    _fftlog = fftlog(num, logrmin, logrmax)
    Ak = get_arr(Ak_func, _fftlog.k, args)
    return ius( _fftlog.r, _fftlog.iHT(Ak, mu, q, beta=beta, prefactor=prefactor) )
