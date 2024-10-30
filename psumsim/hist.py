"""Functions related to the underlying histogram format.

See `dataformat` for a description of the format and some terms."""

import numpy as np
from .array import normalizeAxes, padAxes

HIST_AXIS = -1
"""`int` : Default value for histogram axis.

The last axis is commonly used. The axis is initialized there and stays there.
Doe not matter what happens to the other axes, this index is always valid."""

MAC_AXIS = -4
"""`int` : Default value for axis over MAC operations.

In an MVM application, different multiplication operations are seen when
sweeping over this axis.

In a series of sum/quantize steps, this dimension is usually the first one to
reduce. AFterwards, the result shape has changed, this index is invalid, but
also no more needed.
"""

ACT_AXIS = -3
"""`int` : Default value for axis over different activation values."""

WEIGHT_AXIS = -2
"""`int` : Default value for axis over different weight values."""

STAT_AXIS = 0
"""`int` : Default value for axis over compuations in a statistic setup.

See `statstoc`."""

def checkStatisticsArgs(dostatistic, dostatisticdummy):
	"""Validate flags activating statistic simulation.
	
	See `statstoc` for an explaantion of simulation modes.
	
	Parameters
	----------
	dostatistic : `bool`
		If set, a *statistic* or *statisticdummy* simulation is run.
		See `statstoc`.
		
	dostatisticdummy : `bool`
		If set, a *statisticdummy* simulation is run.
		See `statstoc`.

	Raises
	------
	`ValueError`
		If *dostatisticdummy* is set, but *dostatistic* is not. 

	"""
	if (not dostatistic) and dostatisticdummy:
		raise ValueError(
				"When setting dostatisticdummy, dostatistic also "
				"needs to be set."
		)
		
def histlenToBincount(histlen):
	"""Turn a *histlen* to *bincount*.
	
	See `dataformat`.
	
	.. warning:
		A *bincount* of *1* or *2* is translated to *histlen* *1* by
		`bincountToHistlen`. But this function then returns *bincount*
		*3* in both cases. So try to prevent back-and-forth conversions.

	Parameters
	----------
	histlen : `int`
		A positive *histlen* as in `dataformat`.

	Returns
	-------
	bincount : `int`
		A positive *bincount* as in `dataformat`.

	"""
	
	returnpython = (not isinstance(histlen, np.ndarray))
	bincount = np.array(histlen, copy=True)
	np.multiply(bincount, 2, out=bincount)
	np.add(bincount, 1, out=bincount)
	if returnpython:
		bincount = int(round(bincount.item()))
	return bincount

def bincountToHistlen(bincount):
	"""Turn a *bincount*to *histlen*.
	
	See `dataformat`. *bincounts* *1* and *2* are valid and refer to value
	sets *[1]* and *[0; 1]* respectively. They return a *histlen* of *1*.
	

	Parameters
	----------
	bincount : `int`
		A positive *bincount* as in `dataformat`.

	Raises
	------
	`ValueError`
		If the *bincount* is not an odd number.

	Returns
	-------
	histlen : `int`
		A positive *histlen* as in `dataformat`.

	"""
	
	
	returnpython = (not isinstance(bincount, np.ndarray))
	histlen = np.array(bincount, copy=True)
	np.subtract(histlen, 1, out=histlen)
	np.floor_divide(histlen, 2, out=histlen)
	#If we now have histlen 0, we still turn it 1. In that case, histlen and
	#bincount can be the same for a first computation. They both represent
	#value 1.
	#If bincount was 2, we set (keep) histlen at 1. We then have only
	#values 0 and 1, which could be nice for an initial histogram
	forcehistlenone = (bincount <= 2)
	np.copyto(histlen, 1, where=forcehistlenone)
	#Otherwise check returned value
	newbincount = histlenToBincount(histlen=histlen)
	badhistlen = (newbincount != bincount) & (not forcehistlenone)
	if np.any(badhistlen):
		raise ValueError(
				f"Bincount {bincount} is not an odd number "
				f"representing positive and negative values on a histogram.",
				bincount,
		)
	if returnpython:
		badhistlen = int(round(badhistlen.item()))
	return histlen
		
def getHistValues(bincount, axis, ndim, padlower, padupper):
	"""Get a vector of values to which histogram bins refer.

	Parameters
	----------
	bincount : `int`
		Number of bins and length of histogram.
		
	axis : `int`
		Forwarded to `padAxes`. Often, `HIST_AXIS` is used.
		
	ndim : `int`
		Forwarded to `padAxes`.
		
	padlower : `bool`
		Forwarded to `padAxes`.
		
	padupper : `bool`
		Forwarded to `padAxes`.

	Returns
	-------
	histindex : `numpy.ndarray` of `int`
		The values the bins of a histogram refer to. By default, this is
		1D array. Lower and upper padding is then applied by `padAxes`.
	"""
	
	#Get broadcastable matrix which notes numbers which notes which numbers
	#make it where in the histogram.
	#THe bincount is the length of the full histogram with positive and
	#negative numbers.
	histlen = bincountToHistlen(bincount=bincount)
	#If the bincount was already just 1, we get histlen 1, but we really
	#meant to give digits a value 1
	if bincount <= 1:
		rangestart = histlen
	#If it was 2, we also get histlen of 1, but the value range starts
	#at 0.
	elif bincount == 2:
		rangestart = 0
	#otherwise, we return a true symmetric range
	else:
		rangestart = -histlen
	histindex = np.arange(start=rangestart, stop=histlen+1, dtype="int")
	
	#Expand dimensions to make this broadcastable with tounpack
	histindex = padAxes(
			value=histindex,
			innertoaxis=axis,
			referencendim=ndim,
			padlower=padlower,
			padupper=padupper,
	)
	
	return histindex

def unpackHist(tounpack, bincount, axis):
	"""Unpack a *dostatistic* to a *dostatisticdummy* representation.
	
	See `dataformat`. This turns an array of `int` values into an array
	of one-hot histograms of `bool`.

	Parameters
	----------
	tounpack : `numpy.ndarray`
		The array to unpack. Is not allowed to be of type `numpy.floating`.
		The new histogram axis must already exist and have length 1.
		
	bincount : `int`
		Desired length of new histogram.
		
	axis : `int`
		New histogram axis index. Often, `HIST_AXIS` is used.

	Raises
	------
	`TypeError`
		If *tounpack* has some floating-point dtype.
		
	`IndexError`
		If the given histogram axis has not length 1.
		
	`ValueError`
		If during conversion, there are are values found which are too large
		to be represented by *bincount*. 

	Returns
	-------
	unpacked : `numpy.ndarray` of `bool`
		New array, with same shape like *tounpack* except that the *axis* has
		now length *bincount*.

	"""
	
	#Axis should be a single value
	axis, = normalizeAxes(axes=axis, referencendim=tounpack.ndim)
	
	if np.issubdtype(tounpack.dtype, np.floating):
		raise TypeError(
				f"Expected some int type for histogram unpacking, but got "
				f"{tounpack.dtype}.",
				tounpack.dtype,
		)
	if tounpack.shape[axis] > 1:
		raise IndexError(
				f"Can only unpack histogram along a length 1 dimension, but "
				f"got asked to unpack dimension {axis} in shape "
				f"{tounpack.shape}.",
				axis,
				tounpack.shape,
		)
	histlen = bincountToHistlen(bincount=bincount)
	maxval = np.max(np.absolute(tounpack), axis=None, keepdims=False)
	if maxval > histlen:
		raise ValueError(
				f"Unpacking a histogram with bincount {bincount}, and "
				f"histlen {histlen}, but "
				f"the maximum value {maxval} is too large.",
				bincount,
				histlen,
				maxval,
		)
	
	histindex = getHistValues(
			bincount=bincount,
			axis=axis,
			ndim=tounpack.ndim,
			padlower=False,
			padupper=True,
	)
	
	#Do the unpacking
	unpacked = (tounpack == histindex)
	
	return unpacked

def packHist(topack, axis, keepdims=True, strict=True):
	"""Pack a *dostatisticdummy* to a *dostatistic* representation.
	
	See `dataformat`. This turns an array of one-hot `bool` values into an
	array of `int`.

	Parameters
	----------
	topack : `numpy.ndarray`
		The array to pack. Is not allowed to be of type `numpy.floating`.
		The histograms must be one-hot encoded.

	axis : `int`
		Old histogram axis index. Often, `HIST_AXIS` is used.
		
	keepdims : `bool`, optional
		Whether to keep the obsolete histogram axis after conversion instead of
		using `numpy.squeeze` on it.
		The default is to keep it.
		
	strict : `bool`, optional
		If set, *tounpack* is checked on being a one-hot histogram.
		The default is to do the check.

	Raises
	------
	`TypeError`
		If *topack* has some floating-point dtype.
		
	`ValueError`
		If during conversion, there are histograms found which have not exactly
		one non-zero value. Skipped if *strict* is not set.

	Returns
	-------
	packed : `numpy.ndarray` of `int`
		The packed array. Dimension *axis* has length 1 or is completely removed,
		depending in *keepdims*. The datatype is deduced from `getHistValues`,
		as that knows best about values of histograms.

	"""
	#Axis should be a single value
	axis, = normalizeAxes(axes=axis, referencendim=topack.ndim)
	
	if np.issubdtype(topack.dtype, np.floating):
		raise TypeError(
				f"Expected some int type for histogram unpacking, but got "
				f"{topack.dtype}.",
				topack.dtype,
		)
		
	#Get number of histogram bins from histogram axis.
	bincount = topack.shape[axis]
	
	histindex = getHistValues(
			bincount=bincount,
			axis=axis,
			ndim=topack.ndim,
			padlower=False,
			padupper=True,
	)
	
	#Replace the ones in the historgam by the value they represent.
	#Actually even multiply, otherwise unpacking two hist axes in series
	#does not work. Multiplication only works with int dtype
	#packed = np.where(topack, histindex, np.zeros_like(histindex))
	packed = topack.astype(histindex.dtype)
	np.multiply(packed, histindex, out=packed)
	#And remove all the zeros. This works, because we solely expect
	#one-hot vectors. Before doing so, check that we actually have one-hot.
	ishot = np.count_nonzero(topack, axis=axis, keepdims=True)
	ishotmax = int(ishot.max(axis=None, keepdims=False))
	ishotmin = int(ishot.min(axis=None, keepdims=False))
	if strict and ((ishotmax != 1) or (ishotmin != 1)):
		raise ValueError(
				f"Packing a histogram to uint values only works, if only "
				f"up to 1 value is set along histogram axis, but from "
				f"{ishotmin} up to {ishotmax} were found.",
				ishotmin,
				ishotmax,
		)
	
	packed = np.sum(packed, axis=axis, keepdims=keepdims, dtype=packed.dtype)
	
	return packed

def packStatistic(topack, axis, keepdims=True):
	"""Pack a *dostatisticdummy* to a *dostochastic* representation.
	
	See `dataformat`. Use this, if you have a ton of computations which have
	been done in parallel and want to know the probability of each result
	value.

	Parameters
	----------
	topack : `numpy.ndarray` of `bool`
		The array to pack. Any dtype allowed.

	axis : `int`
		The statistic axis. Often, `STAT_AXIS` is used.
		
	keepdims : `bool`, optional
		Whether to keep the obsolete statistic axis after conversion instead of
		using `numpy.squeeze` on it.
		The default is to keep it.

	Returns
	-------
	packed : `numpy.ndarray` of `float`
		The packed array. Dimension *axis* has length 1 or is completely removed,
		depending in *keepdims*. The datatype is determined by `numpy.average`
		and probably is some `float` probability.

	"""
	
	if topack.dtype != "bool":
		raise ValueError(
				f"Can only pack histograms with bool dtype to some probability, "
				f"but got {topack.dtype}.",
				topack.dtype,
				
		)
	
	#Axis should be a single value
	axis, = normalizeAxes(axes=axis, referencendim=topack.ndim)
	
	#Average bits along statistic dim
	packed = np.average(
			topack,
			axis=axis,
			keepdims=keepdims,
			#dtype="float",
	)
	
	return packed

def getHistLenFromMaxValues(
		target,
		maxhistvalue,
		dostatistic,
		dostatisticdummy,
		histaxis,
	):
	"""Derive *histlen* and *histvalues* from a known maximum of computations.
	
	See `maxhistvalue`.

	Parameters
	----------
	target : `numpy.ndarray`
		Asked for *ndim* and *shape*. Returned *histvalues* will be broadcastable
		to this array.
		
	maxhistvalue : `numpy.ndarray`, `None`
		The maximum magnitude of each histogram. As the values we return are not
		histogram-wise, this is max'ed to a single value. So one could also
		ask :code:`target.shape[histaxis]`, but that does not exist in
		*dostatisticdummy* (see `statstoc`).
	
	dostatistic : `bool`
		If set, a *statistic* or *statisticdummy* simulation is run.
		See `statstoc`.
		
	dostatisticdummy : `bool`
		If set, a *statisticdummy* simulation is run.
		See `statstoc`.
		
	histaxis : `int`
		The axis along one finds histogram values. Often, `HIST_AXIS` is used.

	Returns
	-------
	oldhistlen : `int`, `None`
		The single *histlen*, in which all values of all histograms would
		fit. Derived from *maxhistvalue* and if that is `None`, this is
		`None`, too.
		
	oldhistvalues : `numpy.ndarray`, `None`
		Returned by `getHistValues`. The values represented by bins.
		Broadcastable to *target*. Derived from *target.shape* and NOT
		*maxhistvalues*. `None`, if the histogram axis is a length-1 dummy
		due to statistic simulation. The idea is simply that if one
		has some `int` result and no histogram, one should not need the
		*histvalues*.
	"""
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=target.ndim)
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	#Get length of histogram. Ask maxhistvalue here instead of
	#target.shape[histaxis], because the hist length could be 1 in an int
	#computation.
	#Use max over all dimensions, because that is what a stochastic
	#computation also does. This also helps with applying the
	#same scale on all digits, otherwise we would get into trouble
	#latest when combining values along differently scaled axes.
	#If maxhistvalue was not given, also set no value here. The caller
	#should supervise, that the oldhistlen is then not used.
	if maxhistvalue is not None:
		oldhistlen = int(round(np.max(
				maxhistvalue,
				axis=None,
				keepdims=False,
		)))
	else:
		oldhistlen = None
		
	#Whether we do not do uint computations and have a hist dim which
	#length we can use
	canusehistax = (not dostatistic) or dostatisticdummy
		
	#Values of histogram enteies from central def. These should be
	#broadcastable to the hist axis, so use its length instead of the
	#bincount retrieved from maxhistvalue, because that might be shorter.
	if canusehistax:
		#Also pad lower axes, otherwise axis indices do not work.
		oldhistvalues = getHistValues(
				bincount=target.shape[histaxis],
				axis=histaxis,
				ndim=target.ndim,
				padlower=True,
				padupper=True,
		)
	#If we do not have a histogram because we work on uint, the caller
	#should have a simple fallback not needing oldhistvalues.
	else:
		oldhistvalues = None
		
	return oldhistlen, oldhistvalues
