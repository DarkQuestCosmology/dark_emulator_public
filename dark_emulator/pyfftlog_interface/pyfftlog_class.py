import pyfftlog
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def hankel_transform(n, kr, q, mu, dlnr, Ar):
    """executing Hankel transformation

    Args:
        n (int): binning number of fftlog
        kr (float): Product of k_c and r_c where 'c' indicate the center of scale array, r and k.
        q (flaot): bias index for fftlog.
        mu (float): mu of Bessel function.
        dlnr (float): bin width in logarithmic
        Ar (ndarray): array of input function
    Returns:
        Ak (ndarray): Hankel-transformed array.

    """
    _kr, xsave = pyfftlog.fhti(n, mu, dlnr, q, kr, 0)
    Ak = pyfftlog.fht(Ar.copy(), xsave, 1) # execute fftlog
    return Ak

class fftlog:
    """pyfftlog wrapper class
    """
    def __init__(self, num=1, logrmin=-3.0, logrmax=3.0, kr=1):
        """
        Args:
            num (int,optional): Binning number of fftlog in unit of 2048.
            logrmin (float, optional): Minimum r over which scale fftlog will be executed.
            logrmax (float, optional): Maximum r under which scale fftlog will be executed.
            kr (float, optional): Product of k_c and r_c where 'c' indicate the center of scale array, r and k.

        """
        self.n = n = 2048 * num
        logrmin = logrmin
        logrmax = logrmax
        self.kr = kr
        nc = float(n+1.)/2.

        # array in real space
        logrc = (logrmin + logrmax)/2.
        dlogr = (logrmax - logrmin)/n
        self.dlnr = dlogr*np.log(10.0)
        self.r = 10.**(logrc + (np.arange(1, n+1) - nc) * dlogr)

        # array in Fourier space
        logkc = np.log10(kr) - logrc
        logkmin = np.log10(kr) - logrmax
        logkmax = np.log10(kr) - logrmin
        dlogk = (logkmax - logkmin)/n
        self.dlnk = dlogk * np.log(10.)
        self.k = 10.**(logkc + (np.arange(1, n+1) - nc) * dlogk)

    def HT(self, Ar, mu, q, beta=0, prefactor=1):
        """Hankel transform

        Executing Hankel transformation, defined as

        .. math::
            A(k) = p k^{\\beta} \\int_0^{\\infty} k {\\rm d}r A(r) (kr)^{q} J_{\\mu}(kr).

        where :math:`p` is arbitrary prefactor.

        Args:
            Ar (ndarray): Array of numeric values of function A(r=self.r).
            mu (float): Order of the first kind of Bessel function, :math:`J_\\mu`.
            q (float): Bias parameter in FFTLog
            beta (float): Power index of k to be multiplied to output
            prefactor (float, ndarray): Constant prefactor to be multiplied to output

        Returns:
            Ak (ndarray): Approximated Hannkel transformation of Ar on k=self.k, defined as

        """
        return prefactor * self.k**beta * hankel_transform(self.n, self.kr, q, mu, self.dlnr, Ar)

    def iHT(self, Ak, mu, q, beta, prefactor):
        """inverse Hankel transform

        Executing inverse Hankel transformation, defined as

        .. math::
            A(r) = p r^{\\beta} \\int_0^{\\infty} r {\\rm d}k A(k) (kr)^{q} J_{\\mu}(kr).

        where :math:`p` is arbitrary prefactor.

        Args:
            Ak (ndarray): Array of numeric values of function A(k=self.rk).
            mu (float): Order of the first kind of Bessel function, :math:`J_\\mu`.
            q (float): Bias parameter in FFTLog
            beta (float): Power index of r to be multiplied to output
            prefactor (float, ndarray): Constant prefactor to be multiplied to output

        Returns:
            Ar (ndarray): Approximated Hannkel transformation of Ak on r=self.r, defined as

        """
        return prefactor * self.r**beta * hankel_transform(self.n, self.kr, q, mu, self.dlnk, Ak)

    def pk2xi(self, pk, idx_shift=0.0):
        """converting P(k) to xi(r)

        Converting 3-dimensional power spectrum into 3-dimensional correlation function, defined as

        .. math::
            \\xi(r) = \\int \\frac{{\\rm d}^{3}k}{(2\\pi)^3} P(k) e^{i k\cdot r}
                    = \\frac{1}{(2\\pi r)^{3/2+s}} \\int r{\\rm d}k [k^{3/2-s}P(k)] (kr)^{\\rm s} J_{1/2}(kr)

        where :math:`s` is shift index used for fftlog to converge. Note that the resultant :math:`\\xi` is independent from shift index.

        Args:
            pk (ndarray): Array of P(k=self.k)
            idx_shift (float): Power index, used for tune fftlog to converge.

        Returns:
            xi (ndarray): Array of correlation function, xi.

        """
        return self.iHT(self.k**(1.5-idx_shift)*pk, 0.5, idx_shift, beta=-1.5-idx_shift, prefactor=(2.0*np.pi)**(-1.5) )

    def xi2pk(self, xi, idx_shift=0.0):
        """converting xi(r) to P(k)

        Converting 3-dimensional correlation function into 3-dimensional power spectrum, defined as

        .. math::
            P(k) = \\int {\\rm d}^{3}r \\xi(r) e^{-i k\cdot r}
                 = \\frac{(2\\pi)^{3/2}}{k^{3/2+s}} \\int r{\\rm d}k [r^{3/2-s}\\xi(r)] (kr)^{s} J_{1/2}(kr)

        where :math:`s` is shift index used for fftlog to converge. Note that the resultant :math:`P` is independent from shift index.

        Args:
            xi (ndarray): Array of xi(r=self.r)
            idx_shift (float): Power index, used for tune fftlog to converge.

        Returns:
            pk (ndarray): Array of 3d power spectrum, pk

        """
        return self.HT(self.r**(1.5-idx_shift)*xi, 0.5, idx_shift, beta=-1.5-idx_shift, prefactor=(2.0*np.pi)**1.5 )

    def pk2wp(self, pk, idx_shift=0.0):
        """converting P(k) to wp(R)

        Converting 3-dimensional power spectrum into 2-dimensional projected correlation function, defined as

        .. math::
            w_{\\rm p}(r) = \\int_0^{\\inf} {\\rm d}\\Pi \\xi(\\sqrt{r^2+\\Pi})
                          = \\frac{1}{2\\pi r^{1+s}} \\int r{\\rm d}k [k^{1-s}P(k)] (kr)^{s} J_{0}(kr).

        where :math:`s` is shift index used for fftlog to converge. Note that the resultant :math:`w_{\\rm p}` is independent from shift index.

        Args:
            pk (ndarray): Array of P(k=self.k)
            idx_shift (float): Power index, used for tune fftlog to converge.

        Returns:
            wp (ndarray): Array of 2d projected correlation function, wp

        """
        return self.iHT(self.k**(1.0-idx_shift)*pk, 0.0, idx_shift, beta=-1-idx_shift, prefactor=1.0/(2.0*np.pi) )

    def pk2dwp(self, pk, idx_shift=0.0):
        """converting P(k) to dwp(R)

        Converting 3-dimensional power spectrum into 2-dimensional excess projected correlation function.

        .. math::
            \\Delta w_{\\rm p}(r) = \\bar{w_{\\rm p}}(r) - w_{\\rm p}(r)
                           = \\frac{1}{2\\pi r^{1+s}} \\int r{\\rm d}k [k^{1-s}P(k)] (kr)^{s} J_2(kr)

        where :math:`s` is shift index used for fftlog to converge. Note that the resultant :math:`\\Delta w_{\\rm p}` is independent from shift index.

        Parameters
        --------
        pk (ndarray): Array of P(k=self.k)
        idx_shift (float): Power index, used for tune fftlog to converge.

        Returns:
            wp (ndarray): Array of 2d excess projected correlation function, dwp.

        """
        return self.iHT(self.k**(1.0+idx_shift)*pk, 2.0, idx_shift, beta=-1-idx_shift, prefactor=1.0/(2.0*np.pi) )

