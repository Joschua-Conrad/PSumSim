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
	
	.. note::
		Do not set *a* or *b* (bounds of support of distribution). The methods
		defined here do not accept arguments for that. The standard sinusoidal
		distribution has a support on :math:`[-1;1]` and *scale* and *loc* can
		change that. Buf `_cdf` of very small numbers is just *0* and not
		*NaN* or such.
	"""
	
	def _cdf(self, x):
		r"""Cumulative density function.
		
		`scipy.stats.rv_continuous.cdf` wraps this after applying *loc* and
		*scale*. This is the least we need to define.
		
		The *CDF* describes: with which probability do I draw a number smaller
		than or equal to *x* from the random process. For :math:`x < -1`, that
		is *0*. A sinusoid never yields values smaller than *-1*. For
		:math:`x > 1`, that is *1*, as all values of a sinusoid are below or
		eqal *1*.
		
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
		good = ~bad
		
		#Compute function on best dtype numpy determines and only for digits
		#where we need it.
		ret = np.arcsin(x, where=good)
		np.divide(ret, np.pi, where=good, out=ret)
		np.add(ret, 0.5, where=good, out=ret)

		#Set other digits to fixed values
		np.putmask(ret, mask=toolow, values=0)
		np.putmask(ret, mask=toolarge, values=1)
		
		#Cast dtype. If it is already the one of input value, keep without copy.
		ret = ret.astype(copy=False, dtype=x.dtype)
		
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
		nonconstderiv = ~constderiv

		#Compute function on best dtype numpy determines and only for digits
		#where we need it.
		ret = np.power(x, 2, where=nonconstderiv)
		np.subtract(1, ret, where=nonconstderiv, out=ret)
		np.sqrt(ret, where=nonconstderiv, out=ret)
		np.multiply(ret, np.pi, where=nonconstderiv, out=ret)
		np.divide(1, ret, where=nonconstderiv, out=ret)
		
		#Set other digits to fixed values
		np.putmask(ret, mask=constderiv, values=0)
		
		#Cast dtype. If it is already the one of input value, keep without copy.
		ret = ret.astype(copy=False, dtype=x.dtype)
		
		return ret

	def _ppf(self, x):
		r"""Percent point function.
		
		`scipy.stats.rv_continuous.ppf` wraps this after applying *loc* and
		*scale*. This is needed to draw numbers from this random process in
		`scipy.stats.rv_continuous.rvs` with decent performance.
		
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
		ret = np.subtract(x, 0.5, where=good)
		np.multiply(ret, np.pi, out=ret, where=good)
		np.sin(ret, out=ret, where=good)
		
		#Other digits have fixed NaN value
		np.putmask(ret, mask=bad, values=np.nan)
		
		#Cast dtype. If it is already the one of input value, keep without copy.
		ret = ret.astype(copy=False, dtype=x.dtype)
		
		return ret


sinusoidal = sinusoidal_gen(name="sinusoidal")
"""`sinusoidal_gen` : Default instance of number generator.

`scipy.stats` does this as well. Use this as the default generator and call
this instance (not the class) to create frozen distributions."""


class fullscale_gen(sp.stats.rv_continuous):
	r"""Define random process of variable having a constant full scale value.
	
	Similar to `sinusoidal_gen`.
	
	Use *loc* to define the full-scale value.
	"""
	
	SAMPLE_DTYPE = "float"
	"""`str` : Datatype accepeted by `numpy` functions. Used in `_rvs`. Use
	platform's best floating point."""
	
	def _cdf(self, x):
		r"""Cumulative density function.
		
		Similar to `sinusoidal_gen._cdf`.
		
		We here only need to implement a step function with step at location
		zero. *loc* was already applied for us when this gets called. The *CDF*
		gives the probability to yield a value smaller *or equal* to *x*. So
		if *x* is *loc*, we already return a *1* probability.

		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *CDF* for.

		Returns
		-------
		ret : `numpy.ndarray`
			Same shape and dtype as *x*, but with computed *CDF* values.

		"""
		
		#Implement step function
		ret = np.where(x >= 0, 1, 0)
		#Cast dtype. If it is already the one of input value, keep without copy.
		ret = ret.astype(dtype=x.dtype, copy=False)
		
		return ret
		
	def _pdf(self, x):
		r"""Probability density function.
		
		Similar to `sinusoidal_gen._pdf`.
		
		Derivative of `_cdf`. Ideally, this would be a Dirac pulse, as that
		also has the correct area of *1*. But we here can only output *0*
		all the time and *Inf* at *0*.
		
		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *PDF* for.

		Returns
		-------
		ret : `numpy.ndarray`
			Same shape and dtype as *x*, but with computed *PDF* values.

		"""
		
		#Output with correct shape and dtype, but zero everywhere
		ret = np.zeros_like(x)
		#Where x was zero, the derivative is inf
		np.putmask(ret, mask=(x == 0), values=np.inf)
		
		return ret

	def _ppf(self, x):
		r"""Percent point function.
		
		Similar to `sinusoidal_gen._pdf`.
		
		This is not implemented. Should be the inverse of `_cdf`, but the *CDF*
		yields *0* for all :math:`x < loc` and *1* for all :math:`x \ge loc`.
		Which value to return then? And for all other values, the ivnerse would
		be *NaN*.

		Parameters
		----------
		x : `numpy.ndarray`
			Values to compute *PPF* for.

		Returns
		-------
		ret : `None`
			
		Raises
		------
		`NotImplementedError`
			Always.

		"""
		
		raise NotImplementedError(
				"The PPF for fullscale random process is not defined."
		)

	def _rvs(self, size=None, random_state=None):
		"""Define how random numbers are drawn from this process.
		
		Overwrites `scipy.stats.rv_continuous`.*_rvs* and is called by
		`scipy.stats.rv_continuous.rvs`.
		
		`_ppf` cannot be called, so the default definiton of this method does
		not work, so we create our own one.
		
		Functionality is easy: this random process draws the same number all
		the time. And *loc* is even applied for us. So we just return zeros
		all the time.
		

		Parameters
		----------
		size : (`tuple` of `int`), `None`, optional
			Size of array to generate. `None` generates a 0D array.
			The default is `None`.
		random_state : `object`, optional
			The random number generator. Not used and only added for
			interface compatibility with `scipy.stats.rv_continuous`.*_rvs*.
			The default is `None`.

		Returns
		-------
		ret : `numpy.ndarray`
			One array with dtype `SAMPLE_DTYPE` and *0* values, which will be
			moved to *loc*. Shape set by *size*.

		"""
		
		ret = np.zeros(shape=size, dtype=self.SAMPLE_DTYPE)

		return ret


fullscale = fullscale_gen(name="fullscale")
"""`fullscale_gen` : Default instance of number generator.

Like `sinusoidal`."""
