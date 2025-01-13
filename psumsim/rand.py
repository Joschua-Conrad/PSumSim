#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:40:07 2025

@author: jconrad
"""

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
		*scale*.
		
		The *CDF* describes: with which probability to I draw a number smaller
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
		for all :math:`-1 \ge x \le 1`.

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
		toolow = x < -1
		toolarge = x > 1
		bad = (toolow | toolarge)
		
		#Will work in-place here
		ret = x.copy()
		#Set all digits where we won't need arcsin to 0, such that arcsin does
		#not yield NaN anywhere. Better debussing
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
		
