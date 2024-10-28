import numpy as np
from .array import normalizeAxes, padAxes

#The histogram axis we in the end expect
HIST_AXIS = -1
ACT_AXIS = -3
WEIGHT_AXIS = -2
MAC_AXIS = -4
STAT_AXIS = 0

def checkStatisticsArgs(dostatistic, dostatisticdummy):
	if (not dostatistic) and dostatisticdummy:
		raise ValueError(
				"When setting dostatisticdummy, dostatistic also "
				"needs to be set."
		)
		
def histlenToBincount(histlen):
	returnpython = (not isinstance(histlen, np.ndarray))
	bincount = np.array(histlen, copy=True)
	np.multiply(bincount, 2, out=bincount)
	np.add(bincount, 1, out=bincount)
	if returnpython:
		bincount = int(round(bincount.item()))
	return bincount

def bincountToHistlen(bincount):
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
		
def packUnpackHistCommon(bincount, axis, ndim, padlower, padupper):
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
	
	histindex = packUnpackHistCommon(
			bincount=bincount,
			axis=axis,
			ndim=tounpack.ndim,
			padlower=False,
			padupper=True,
	)
	
	#Do the unpacking
	unpacked = (tounpack == histindex)
	
	return unpacked

def packHist(topack, axis, keepdims, strict):
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
	
	histindex = packUnpackHistCommon(
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

def packStatistic(topack, axis, keepdims):
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
		oldhistvalues = packUnpackHistCommon(
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
