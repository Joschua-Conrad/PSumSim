"""Definition of more random processes to draw operands from."""

import scipy as sp
import numpy as np

class sinusoidal_gen(sp.stats.rv_continuous):
	r"""Define random process of variable drawing random sinusoidal values.
	
	This process describes randomly drawing uniform distributed values
	on :math:`[0:2\pi]` and calculating *sin* of that.
	
	Is implemented by subclassing `scipy.stats.rv_continuous` and then by creating
	one instance of that class as random generator. Still, passing
	*loc* and *scale* still works, even for creating a *frozen distribution*.
	
	Defined functions return same dtype as given with input.
	"""
	
	def _cdf(self, x):
		r"""Cumulative density function.
		
		`scipy.stats.rv_continuous.cdf` wraps this after applying *loc* and
		*scale*. This is the least we need to define.
		
		The *CDF* describes: with which probability do I draw a number smaller
		than *x* from the random process. For :math:`x < -1`, that is *0*. A
		sinusoid never yields values smaller than *-1*. For :math:`x > 1`, that
		is *1*, as all values of a sinusoid are below or eqal *1*.
		
		For values in-between, things are tricky. We have to look at regions
		where :math:`sin(k) < x` and check which length compared to
		:math:`2\pi` that region has. for :math:`x < 0`, we can map the
		desired range to the positive half wave and get
		:math:`\frac{1}{2} - \frac{arcsin(-x)}{\pi}`. For :math:`x >= 0`, we
		get :math:`\frac{1}{2} + \frac{arcsin(x)}{\pi}` by looking at the
		positive half wave and by adding :math:`+\frac{1}{2}` because half the
		values will already have  :math:`x >= 0`. By using point symmetry
		in *arcsin*, we get :math:`\frac{1}{2} + \frac{arcsin(x)}{\pi}`
		for all :math:`x \in [-1;1]`.

		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *CDF* for.

		Returns
		-------
		ret : `numpy.ndarray`
			Same shape and dtype as *x*, but with computed *CDF* values.

		"""
		
		#Which values lie out of the valid input range of arcsin
		toolow = (x < -1)
		toolarge = (x > 1)
		bad = (toolow | toolarge)
		
		#Will work in-place here
		ret = x.copy()
		#Set all digits where we won't need arcsin to 0, such that arcsin does
		#not yield NaN anywhere. Better debugging
		np.putmask(ret, mask=bad, values=0)
		#Compute cdf using arcsin where needed
		np.copyto(
				ret,
				(0.5 + (np.arcsin(ret) / np.pi)),
				mask=(~bad),
		)
		#Set other digits to fixed values
		np.putmask(ret, mask=toolow, values=0)
		np.putmask(ret, mask=toolarge, values=1)
		
		return ret
		
	def _pdf(self, x):
		r"""Probability density function.
		
		`scipy.stats.rv_continuous.pdf` wraps this after applying *loc* and
		*scale*. There exists a default implementation, but that uses
		numerif differentiation.
		
		The *PDF* is the derivative of *CDF*. The piece-wise definition in
		`_cdf` is derived piece-wise.
		
		:math:`\frac{\delta arcsin(x)}{\delta x} = \frac{1}{\sqrt{1 - x^{2}}}`
		as defined in `this scientific source <https://en.wikipedia.org/wiki/Inverse_trigonometric_functions#Derivatives_of_inverse_trigonometric_functions>`_.
		So the total *PDF* is :math:`\frac{1}{\pi \sqrt{1 - x^{2}}}`
		For :math:`x \in \{-1, 1\}` this yields *Inf*.
		
		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *PDF* for.

		Returns
		-------
		ret : `numpy.ndarray`
			Same shape and dtype as *x*, but with computed *PDF* values.

		"""
		
		
		#Which values have a constant 0 derivative
		constderiv = ((x < -1) | (x > 1))
		
		#Will work in-place here
		ret = x.copy()
		#Set all digits where we won't need derivative to 0, such that we do
		#not compute sqrt of negative numbers
		np.putmask(ret, mask=constderiv, values=0)
		#Compute derivative at other digits
		np.copyto(
				ret,
				(1. / (np.pi * np.sqrt(1. - np.power(ret, 2.)))),
				mask=(~constderiv),
		)
		#Set other digits to fixed values
		np.putmask(ret, mask=constderiv, values=0)

	def _ppf(self, x):
		r"""Percent point function.
		
		`scipy.stats.rv_continuous.ppf` wraps this after applying *loc* and
		*scale*. This is needed to draw numbers from this random process in
		`scipy.stats.rv_continuous.rv` with decent performance.
		
		The *PPF* is the inverse of *CDF*. It is given random numbers uniformly
		distributed on :math:`[0;1]` and should yield the sinusoidal amplitudes
		for that. So we should have some *sin* function as *PPF*. The definition
		is :math:`sin((x - \frac{1}{2})\pi)` for :math:`x \in [0;1]`. For
		other *x*, *NaN* is returned, as this is the inverse of `_cdf` and
		that never yields numbers outside :math:`[0;1]`.

		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *PPF* for.

		Returns
		-------
		ret : `numpy.ndarray`
			Same shape and dtype as *x*, but with computed *PPF* values.

		"""
		
		#Values on projection range, which is never reached by cdf
		bad = ((x < 0) | (x > 1))
		good = ~bad
		
		#Compute function on best dtype numpy determines and only for digits
		#where we need it.
		ret = np.subtract(x, 0.5 * np.pi, where=good)
		np.sin(ret, out=ret, where=good)
		np.putmask(ret, mask=bad, values=np.nan)
		
		#Cast dtype. If it is already the one of input value, keep without copy.
		ret = np.astype(ret, copy=False, dtype=x.dtype)
		
		return ret
