"""Functions to simulate MVM applications and for using PSumSim as a package."""

import numpy as np
import scipy
import math
import copy
from .array import normalizeAxes,padAxes, getValueAlongAxis, padToEqualShape, getValueFromBroadcastableIndex
from .hist import getHistValues, bincountToHistlen, histlenToBincount, checkStatisticsArgs, packHist, unpackHist, packStatistic, getHistLenFromMaxValues

def probabilisticAdder(
		tosum,
		reduceaxis,
		histaxis,
		positionweights,
		positionweightsonehot,
		disablereducecarry,
		chunkoffsets,
		overchunkaxis,
		dostatisticdummy,
		allowoptim,
):
	"""Sum power of one axis into the histogram axis.
	
	Parameters
	----------
	tosum : `numpy.ndarray`
		Sum these value. Simulation modes *dostochastic* and *dostatisticdummy*
		(see `statstoc`) are supported.`
		
	reduceaxis : `int`
		One axis to reduce/sum into histogram axis. It is assumed that values
		along this axis are stochastically independent.
		
	histaxis : `int`
		Where the already existing histogram axis is found. Its values will
		be interpreted as described by `getHistValues`. The sum along this
		axis MUST yield *1.0*.
		
	positionweights : `numpy.ndarray`
		Weights to use along *reduceaxis* to weight summands. Must be a 1D
		array with same length as *histaxis* in *tosum*. Dtype must be `int`.
		While summing along `MAC_AXIS`, one probably uses only value *1* here.
		But when summing e.g. along ativation values, this is some
		`numpy.arange`.
	
	positionweightsonehot : `bool`
		If set, describes that along *reduceaxis*, the number of occurences
		of values happening in an actual computation is limited. For example,
		if first `MAC_AXIS` is reduced, the histogram describes for each
		activation value and for its possible occurence counts the
		probabilities. If there are 128 MAC operations, it is not possible
		that activation value 1 is set 128 times and value 2 also 128 times.
		So when reducing `ACT_AXIS`, set this parameter. The expected maximum
		value of the resulting histogram is then reduced to 128 activations
		having value 2, resulting in a maximum outcome
		:math:`128 \times 2 = 256`
		
		However, the computation :math:`128 \times 1 + 64 \times 2 = 256` is not
		outside the maximum range. The simulated probability of this impossible
		case is non-zero. This leads to one of the `limitations`.
		
		As a rule of thumb: set this for `ACT_AXIS` and `WEIGHT_AXIS` or if
		you look at the output-histogram length and think it is too long.
		
	disablereducecarry : `bool`
		If set, the occurences along *reduceaxis* are actually exclusive. This
		is a more extreme *positionweightsonehot*: not only the expected
		maximum value is limited, but along all histograms reduced, only one can
		have a non-zero value. So values along *reduceaxis* are not added to
		a running result one-by-one, but probabilities leading to one result
		bin can just be added up, because they are exclusive.
		
		To stay with the example above, after having `MAC_AXIS` reduced and while
		reducing `ACT_AXIS`, there still could be a totally valid summation
		:math:`64 \times 1 + 64 \times 2 = 192`. So along *reduceaxis*
		(which here holds value *1* and *2*) it is possible that activation *1*
		occured some times along MACs and value *2* also occured some times.
		They are not exclusive and *disablereducecarry* should not be set.
		
		BUt if you first reduce `ACT_AXIS` and `WEIGHT_AXIS` before reducing
		`MAC_AXIS`, you can set this. Because in reality,
		`ACT_AXIS` and `WEIGHT_AXIS` describe a one-hot 2D matrix, where in
		reality one digit will occur saying "this MAC multiplied this weight
		and this activation value". *disablereducecarry* should be set then.
		
		As a rule of thumb, set this when first reducing `ACT_AXIS` and
		`WEIGHT_AXIS` or when expecting that of all combined histograms,
		only one value in reality will be set.
	
	chunkoffsets : `numpy.ndarray`
		Must be as long as the *overchunkaxis* dimension.
		See `chunks`. In principle, all axes besides *histaxis* and *reduceaxis*
		are treated in parallel and are kept like `probabilisticAdder` would
		have been called independently along them. But along one axis, one
		can add a rising offset to *positionweights*. In the end, we iterate
		over chunks and their offsets and over *reduceaxis* and over old
		histogram values and if the bin is set, we add
		:math:`\text{histvalue} \times (\text{positionweight} + \text{chunkoffset})`.
		The *chunkoffsets* are thereby usefule when having an axis like
		`ACT_AXIS` and chunking it. If we originally had 5 levels with
		*positionweights* *[-2, -1, 0, +1, +2]* and we chunk that into two
		chunks, we by default would get two chunks of length 3 with
		*positionweights* *[-1, 0, +1]*. Reducing the chunks afterwards simply
		gives wrong results. But we can use chunkoffsets *[-1, +2]* to get
		a sum of *positionweights* and *chunkoffsets* resulting into
		*[-2, -1, 0]* for first chunk and *[+1, +2, +3]* for second chunk.
		Now, each element in each chunk gets a correct weight. The uppermost
		weight *+3* is unsued, because it sits in a *residual chunk*
		(see `chunks`).
		
		Just use some `numpy.zeros` here, except when having used chunking on
		an axis, which needs non-uniform *positionweights*.
	
	overchunkaxis : `int`
		The axis over which chunks are given. Add a length-1 dim if needed.
		
	dostatisticdummy : `bool`
		See `statstoc`. Note that this function cannot run *dostatistic*
		simulations, as it always works on histograms.
		
	allowoptim : `bool`
		If set, runtime of this function can be accelerated by drawing the
		results from a binomial distribution and by directly summing-up
		probabilities and not histograms if *disablereducecarry* is set.
		Thereby, no error is made. Unset this only for debugging. In that case,
		the result is always created by iterating over chunks, *reduceaxis*
		and over *histaxis*, which simply takes time.

	Raises
	------
	`IndexError`
		If *reduceaxis*, *histaxis* and *overchunkaxis* are not three different
		values.
		
	`TypeError`
		If *positionweights* or *chunkoffsets* are not of some `int` dtype.
		
	`ValueError`
		If *chunkoffsets* of *positionweights* have a bad shape.

	Returns
	-------
	target : `numpy.ndarray`
		Result with same dtype as *tosum*. *histaxis* is longer here, but still
		a sum over it gives *1.0*. Length of *reduceaxis* is 1.

	"""
	
	#Historgrams do not include a 0 bin, this is implciit because the
	#sum over probabilities must be 1.
	#We always keep axes here to not get confused with axis indices.
	
	#Make axes indices positive and expect a single value
	reduceaxis, = normalizeAxes(axes=reduceaxis, referencendim=tosum.ndim)
	histaxis, = normalizeAxes(axes=histaxis, referencendim=tosum.ndim)
	overchunkaxis, = normalizeAxes(axes=overchunkaxis, referencendim=tosum.ndim)
	
	if reduceaxis == histaxis:
		raise IndexError(
				f"Reduce and histogram axis need to be different, but are "
				f"{reduceaxis} and {histaxis}.",
				reduceaxis,
				histaxis,
		)
		
	if reduceaxis == overchunkaxis:
		raise IndexError(
				f"Reduce and overchunk axis need to be different, but are "
				f"{reduceaxis} and {overchunkaxis}.",
				reduceaxis,
				overchunkaxis,
		)
		
	if overchunkaxis == histaxis:
		raise IndexError(
				f"Overchunk and histogram axis need to be different, but are "
				f"{overchunkaxis} and {histaxis}.",
				overchunkaxis,
				histaxis,
		)
		
	if np.issubdtype(positionweights.dtype, np.floating):
		raise TypeError(
				f"positionweights can only be of integer dtypes, but found "
				f"{positionweights.dtype}.",
				positionweights.dtype,
		)
		
	if np.issubdtype(chunkoffsets.dtype, np.floating):
		raise TypeError(
				f"chunkoffsets can only be of integer dtypes, but found "
				f"{chunkoffsets.dtype}.",
				chunkoffsets.dtype,
		)
		
	expectedshape = (tosum.shape[reduceaxis],)
	if positionweights.shape != expectedshape:
		raise ValueError(
				f"positionweights should be 1D array with shape "
				f"{expectedshape}, but shape "
				f"{positionweights.shape} was found.",
				expectedshape,
				positionweights.shape
		)
	expectedshape = (tosum.shape[overchunkaxis],)
	if chunkoffsets.shape != expectedshape:
		raise ValueError(
				f"chunkoffsets should be 1D array with shape "
				f"{expectedshape}, but shape "
				f"{chunkoffsets.shape} was found.",
				expectedshape,
				chunkoffsets.shape
		)
	if disablereducecarry and (not positionweightsonehot):
		raise ValueError(
				"Setting disablereducecarry only makes sense if also "
				"positionweightsonehot is also set."
		)
	
	#Compute expected length of new value axis. We maximum add all numbers
	#along reduce axis up, where the max value is described by the
	#valueaxis.
	#Histograms do not include a bin for value 0. If you have N bins, you
	#use them to mark bin values 1 to N.
	#We expect given histograms and posweights to be symmetric and
	#use absolut().
	#Similar computation for chunks, but there we only need a maximum.
	#Because that weight is as long as the overchunkax, but we do not
	#care about that axis.
	positionweightsabs = np.absolute(positionweights, dtype=positionweights.dtype)
	posweightssum = positionweightsabs.sum(
			axis=None,
			keepdims=False,
	)
	chunkoffsetmax = chunkoffsets.max(
			axis=None,
			keepdims=False,
	)
	#Histlen is supposed to represent the maximum number magnitude.
	#THe histogram axis contains zero and negative numbers, too.
	oldbincount = tosum.shape[histaxis]
	oldhistlen = bincountToHistlen(bincount=oldbincount)
	
	#In one-hot, we know that the hist axis represents how often a value
	#along reduceaxis occurs. And the total number of occurences is limited.
	#We get the maxmum reslt, if all occurences happen at the maximum posweight.
	#The chunk weight is simply an offset applied on top of positionweights.
	if positionweightsonehot:
		posweightssummax = positionweightsabs.max(
				axis=None,
				keepdims=False,
		)
	else:
		posweightssummax = posweightssum

	posweightssum = int(round(posweightssum))
	chunkoffsetmax = int(round(chunkoffsetmax))
	posweightssummax = int(round(posweightssummax))
	newhistlen = oldhistlen * (posweightssummax + chunkoffsetmax)
	newhistlen = int(round(newhistlen))
	reducelen = tosum.shape[reduceaxis]
	chunkcount = tosum.shape[overchunkaxis]
	
	#Number of bins in new hist axis
	newbincount = histlenToBincount(histlen=newhistlen)
	
	#Which values the old histogram represented.
	oldhistvalues = getHistValues(
			bincount=oldbincount,
			axis=histaxis,
			ndim=tosum.ndim,
			padlower=False,
			padupper=True,
	)
	
	#Also detect whether all chunkoffsets are the same
	chunkoffsetmin = chunkoffsets.min(axis=None, keepdims=False)
	allchunkssame = np.all(chunkoffsetmax == chunkoffsetmin)
	
	#Prepare result shape, where the reduce axis has only lenth 1 and the
	#length of the target axis was updated.
	#We so far cmómputed with magnitudes only, but the new hist axis needs
	#space for 0 and negative, too.
	targetshape = np.array(tosum.shape)
	targetshape[reduceaxis] = 1
	targetshape[histaxis] = newbincount
	targetdtype=tosum.dtype
	
	#If all values along the reduce axis are the same, we can use a
	#binomial distribution to right away read our result. But that does
	#work only, if the old histogram had just a single value. Because the
	#binomial distribution tells us for all combinations of booleans set
	#along reduce axis, but if these values are not just booleans but rather
	#integers from old hist axis, we also gotta regard combinations between
	#different uint values along reduce axis. Even though they migh have
	#same probabilities, one might sill draw different values along
	#reduce axis. But the binomial distribution only regards "set or not
	#set".
	#Also works only, if values and weigths along reduceaxis are all the same,
	#otherwise it is not known which probability p to use in binomial.
	#If the axis we add from is onehot, we should not use binomial, because
	#it will reagrd all combinatiosn of multiple bits along reduceaxis being
	#set.
	#Also works only, if the chunksweights are all the same, because
	#we use accessors to shift the binomial distribution and that does
	#not broadcast a chunk ax.
	#We try to test the more computational expensive crieteria in the
	#end and only if previous criteria passed.
	
	#Only do, if we can have ultiple values along reduceax set and if
	#ther is a primitive old histogram with only values 0 and 1 and a sum
	#of 1 over all values and if such optimizations are allowed
	dobinomial = bool((not positionweightsonehot) and (oldbincount <= 2) and allchunkssame and allowoptim)
	
	#Check if posweights are all the same. We do not care about chunkoffsets,
	#as each chunk gets a separate histogram.
	if dobinomial:
		posweightmin = positionweights.min(axis=None, keepdims=False)
		posweightmax = positionweights.max(axis=None, keepdims=False)
		dobinomial = np.all(posweightmax == posweightmin)
	if dobinomial:
		tosummin = tosum.min(axis=reduceaxis, keepdims=True)
		tosummax = tosum.max(axis=reduceaxis, keepdims=True)
		dobinomial = np.all(tosummax == tosummin)
		
	#Now do simplified computation
	if dobinomial:
		#Generate target array keep dtype. Create the target once and do not use a
		#growing one, because the growing one would cause many memory allocations.
		target = np.zeros(targetshape, dtype=targetdtype)
		#This is where we read the binomial distribution. We want to know
		#probabilitiy to get k set bits along reduce axis.
		sumreads = np.arange(start=0, stop=reducelen+1, step=1, dtype="uint")
		#Will need this to pad axes to coordinates where binomial dist
		#is read.
		sumreads = padAxes(
				value=sumreads,
				innertoaxis=histaxis,
				referencendim=tosum.ndim,
				padlower=False,
				padupper=True,
		)
		
		#THe probability that a single occurence is set can be read from
		#histogram. If bincount is 1, that is the bin we need. If bincount
		#is 2, we have probs of 0 and 1 values and need the upper one
		binomprob = getValueAlongAxis(
				value=tosummin,
				start=-1,
				stop=None,
				step=None,
				axis=histaxis,
		)
		
		#We sample the full area under the pdf. The sum along sumreads is 1.
		sumvals = scipy.stats.binom.pmf(n=reducelen, k=sumreads, p=binomprob)
		sumvals = sumvals.astype(dtype=targetdtype)
		#We now have a value with length 1 reduce axis and good count of
		#hist values.
		#Write these values now to the target axis.
		#We now have to describe where the repeating occurences
		#along reduceaxis make it into the new histogram.
		#If we currently look at posweight value 2 and that it
		#occurs along reduce axis 1, 2, 3, 4, .. times, that generates
		#values in new histogram with values 2, 4, 6, 8, ...
		#Here, we use stepped, basics numpy indexing instead of
		#advanced indexing to utilize numpy views instead of copies.
		
		#Compute the value if a first set value in histogram.
		#We only know hof often the values occur, but posweight can
		#change that to a different value in histogram.
		thisposweightval = int(posweightmin + chunkoffsetmin)
		#Index of the first digit we write: the 0 value.
		#It sits at the center of the hist.
		firstwrittenidx = int(newhistlen)
		targetaccessor = getValueAlongAxis(
				value=None,
				#The smallest hist index we propagate into: one bool
				#along histaxis was set. That adds exactly the old hist
				#val.
				start=firstwrittenidx,
				#We in the end add up reducelen+1 values and our accessor
				#shall yield that number of elements.
				stop=firstwrittenidx+(thisposweightval*(reducelen+1)),
				#THe value is also our stepwidth
				step=thisposweightval,
				axis=histaxis,
		)
		#Combine the signal energy into target hist. THe initial zero
		#we wrote above is overwritten here.
		target[targetaccessor] = sumvals
			
	#In one-hot systems, one could expect that you can simplify addition.
	#BUt imagine having a one-hot axis to reduce and an already existing
	#target hist axis. So you can go to one position along the reduce axis
	#and use the hist axis to read how often that value occured. Now imagine
	#value 1 occured 4 times and value 2 only 3 times. THe first one adds
	#signal 1*4 = 4 to the hist axis, the second one 2*3 = 6. The code
	#below would then set the hist axis entries 4 and 6. And with two set
	#digits in the histogram, that's no more a histogram. Because only
	#values 4+6=10 should be set. So: as soon as the source histogram is
	#longer than 1, we again need to regard a ton of combinations leading
	#to a histogram bin. ANd we then can right away go to default computation.
	#This works only, if we know that each entry in the output hist can
	#only be set by a single digit of the ones we add over.
	#(oldbincount == 1) effectively causes this optimization to never be
	#active.
	elif positionweightsonehot and disablereducecarry and (oldbincount == 1) and allowoptim:
		#Generate target array keep dtype. Create the target once and do not use a
		#growing one, because the growing one would cause many memory allocations.
		target = np.zeros(targetshape, dtype=targetdtype)
		
		reducevalues = positionweights
		reducevalues = padAxes(
				value=reducevalues,
				innertoaxis=reduceaxis,
				referencendim=tosum.ndim,
				padlower=False,
				padupper=True,
		)
		
		chunkvalues = chunkoffsets
		chunkvalues = padAxes(
				value=chunkvalues,
				innertoaxis=overchunkaxis,
				referencendim=tosum.ndim,
				padlower=False,
				padupper=True,
		)
		
		#positionweights and chunkoffsets are added when bein interpreted.
		#Stick to some dtype noting value of digits.
		totalvalues = np.add(reducevalues, chunkvalues, dtype=oldhistvalues.dtype)
		#effectivehistvalue could add even more dims to totalvalues
		#depending on whether histaxis sits before or after chunks.
		totalvalues = np.multiply(totalvalues, oldhistvalues, dtype=oldhistvalues.dtype)
		
		newhistvalues = getHistValues(
				bincount=newbincount,
				axis=histaxis,
				ndim=tosum.ndim,
				padlower=False,
				padupper=True,
		)
		
		#Shape to reshape values looked up from tosum. Along histaxis,
		#multiple values could be set.
		tosumredshape = list(targetshape)
		tosumredshape[histaxis] = -1
		
		#Now set values in target. Use for loop, otherwise we would in
		#memory iterate over new hist, old hist and reduce hist.
		for thishistidx in range(newbincount):
			targetaccessor = getValueAlongAxis(
					value=None,
					start=thishistidx,
					stop=thishistidx+1,
					step=1,
					axis=histaxis,
			)
			#In newhistvalues, the histaxis is the one with idx 1 to be
			#broadcastable.
			thishistval = newhistvalues[thishistidx:(thishistidx+1)]
			tosumaccessor = (totalvalues == thishistval)
			tosumaccessor = np.broadcast_to(tosumaccessor, shape=tosum.shape)
			tosumred = tosum[tosumaccessor]
			tosumred = np.reshape(tosumred, tosumredshape)
			tosumred = tosumred.sum(axis=histaxis, keepdims=True, dtype=targetdtype)
			target[targetaccessor] = tosumred
			
	#Otherwise iterate over all combinations of old histogram value and
	#added value along reduceax. We add the values one by one.
	else:
		#Whether we can optimize reading target: in disablereducecarry
		#the running result hist we move is alwas the one with only the
		#0 value set.
		simplifytarget = disablereducecarry and allowoptim
		
		#Skip preparation of target, if that simplification is made
		if not simplifytarget:
			#Generate target array keep dtype. Create the target once and do not use a
			#growing one, because the growing one would cause many memory allocations.
			target = np.zeros(targetshape, dtype=targetdtype)
			#Our target must contain a hot value along its histogram axis.
			#Otherwise we lack a 0-probability to shift up and down.
			targetsetter = getHistValues(
					bincount=newbincount,
					axis=histaxis,
					ndim=target.ndim,
					padlower=False,
					padupper=True,
			)
			targetsetter = (targetsetter == 0)
			#Set digits in target while broadcasting the mask. ALlow unsafe
			#casting, as the target could be bool, too.
			np.copyto(target, 1, casting="unsafe", where=targetsetter)
		#If the simplification is made, we have to know the index
		#where the 1 value sits
		else:
			#Create newhistvalues as 1D array
			newhistvalues = getHistValues(
					bincount=newbincount,
					axis=0,
					ndim=1,
					padlower=False,
					padupper=True,
			)
			#And find idx where the 0 sits
			targetzeroidx = (newhistvalues == 0)
			targetzeroidx = np.argmax(
					targetzeroidx,
					axis=None,
					keepdims=True,
			)
			targetzeroidx = int(round(targetzeroidx.item()))
		
		#If that is the case, we still have a probability map with one
		#histogram per value along reduceax. And that hist has a sum of values
		#of 1 itself. We cannot sum all of them up. The probability that
		#a value in old histogram is set, becomes the probability that it is
		#set and that all other values are unset.
		#We assume that all histograms add equal powers. If values are
		#weaker in a histogram along readuceaxis, zero probability is
		#supposed to be higher.
		if disablereducecarry and (not dostatisticdummy):
			#Divide by number of histograms in float
			tosumfiltered = np.divide(tosum, reducelen, dtype=tosum.dtype)
		elif disablereducecarry and dostatisticdummy:
			#Otherwise, unset the zero-val probabilities, because we have
			#them maybe set in every hist. Then, still ensure using that
			#zero value that at least one entry over reducelen histograms
			#is set.
			histzerovalidx = (oldhistvalues == 0)
			tosumfiltered = np.where(histzerovalidx, False, tosum)
			nothingset = np.any(tosumfiltered, axis=(histaxis, reduceaxis), keepdims=True)
			np.logical_not(nothingset, out=nothingset)
			firstreduce = getValueAlongAxis(
					value=tosumfiltered,
					start=0,
					stop=1,
					step=None,
					axis=reduceaxis,
			)
			np.copyto(firstreduce, True, where=(histzerovalidx & nothingset))
		else:
			tosumfiltered = tosum
			
		
		#Iterate over all values to sum up and their histogram bins.
		#Do this with a dummy for loop to reduce memory requirement.
		#Using np operations anyhow is not possible, because the shifted access
		#does not support to roll one axis along multiple different values.
		#Furthermore, we cannot parallelize this loop, because each loop
		#iteration reads and writes target. This is an inter-loop dependency.
		#We have that, because we can always only add two numbers up. Otherwise
		#we have to go thru all combinatorial cases of which combinations of
		#operand values generate the same result. Meaning 0+1 is the same as 1+0.
		#added numbers can create the same result.
		for redidx in range(reducelen):
			tosumred = getValueAlongAxis(
					value=tosumfiltered,
					start=redidx,
					stop=redidx+1,
					step=1,
					axis=reduceaxis,
			)
			
			#We now first iterate along the old histogram with a constant
			#position in reduceaxis. The nice thing along this axis: the
			#histogram axis is a one-hot axis and the sum of probabilities
			#along that ax is 1. So we can simply OR all of them without
			#watching cross-combinations.
			#We do so by combining shifted versions of the target histogram
			#in an innertarget adn at the end of iterating over the old
			#histogram, we apply that inner target as the new one.
			#If writing to target is disabled, we read from dummy target
			#always and keep this running result once created
			if (not disablereducecarry) or (redidx == 0):
				innertargetshape = targetshape
				innertargetshape[histaxis] = newbincount
				innertarget = np.zeros(shape=innertargetshape, dtype=targetdtype)
			
			#Now iterate over all histogram values we add and hence over
			#shift widths we have to apply on existing target. This cannot
			#be parallelized, because we then cannot work with
			#slice() accessors and hence with numpy views. Numpy advanced
			#indexing creates copies!
			for histidx in range(oldbincount):
				tosumval = getValueAlongAxis(
						value=tosumred,
						start=histidx,
						stop=histidx+1,
						step=1,
						axis=histaxis,
				)
				
				#On the innermost side, we now even need to iterate over
				#chunks, which usually can just be treated in parallel by np,
				#but that cannot be made, if chunkoffsets are used.
				#Because different chunkoffsets yield different shifts
				#and shifts are implemented as accessors and these cannot
				#be made different over chunkax.
				#Depending on the case, we generate single chunkoffsets and
				#accessors or we don't care and access all chunks in parallel.
				if allchunkssame:
					chunkaccessors = (
							dict(
									start=None,
									stop=None,
									step=None,
							),
					)
					chunkoffsetiter = (chunkoffsetmin,)
				else:
					chunkaccessors = range(chunkcount)
					chunkaccessors = (dict(start=i, stop=i+1, step=1) for i in chunkaccessors)
					chunkoffsetiter = iter(chunkoffsets)
					
				for chunkaccessor, thischunkoffset in zip(chunkaccessors, chunkoffsetiter):
					chunkaccessor = getValueAlongAxis(
							value=None,
							axis=overchunkaxis,
							**chunkaccessor,
					)
				
					#Get a rolled version of the target. Adding +1 means that the
					#existing output histogram is shifted
					#by 1 towards upper values.
					#The shift concurs to how much we add, if a bit is set.
					#Regard that we ignore the histidx 0 in our histograms. That
					#value would not shift and would assign target=target.
					#We check that in the end.
					#Positionweights start from 1 and their value can increase
					#the added value and hence the shift.
					#The chunkoffsets are simply applied on top.
					shift = (oldhistvalues[histidx] * (positionweights[redidx] + thischunkoffset))
					#oldhistvalues is an nparray with padded upper dims,
					#where the lowest dim is the hist axis. So we might
					#have some dummy dims dn remove them.
					shift = np.squeeze(shift)
					shift = shift.item()
					shift = int(round(shift))
					#Instead of shifting the target in memory, we simply use accessors
					#during computation.
					#This is where the shifted values will be assigned to.
					#Because we will shift
					#them up, we have to omit some upper vlaues.
					#We do not write target histogram values, where we know that
					#there is no accumulatd signal energy, yet.
					
					#We need accessors for accessing full histograms if
					#we do not simplify target to a single digits along
					#histax
					if not simplifytarget:
						#Prevent stop=-0 becoming stop=0 returning no elems.
						#Everything is designed such that both accessors
						#access the same number of elems.
						innertargetaccessorlower = getValueAlongAxis(
								value=None,
								#Ignore lower elems, if shift is negative
								start=max(-shift, 0),
								#Ignore upper elems, if shift is positive
								stop=(-max(shift, 0) or None),
								step=None,
								axis=histaxis,
						)
						#These are the positions we write the shifted values to.
						#This must access as many elements in innertarget as
						#innertargetaccessorlower in target. Regard that
						#innertarget is already targetignoredhistvalues elements
						#shorter than target, because innertarget is re-created
						#for each iteration over reduceaxis.
						innertargetaccesorupper = getValueAlongAxis(
								value=None,
								#Ignore lower elems, if shift is positive
								start=max(shift, 0),
								#Ignore upper elems, if shift is negative
								stop=(-max(-shift, 0) or None),
								step=None,
								axis=histaxis,
						)
					#Otherwise, we only need an accessor for a single digit
					#in where we write to
					else:
						innertargetaccesorupper = getValueAlongAxis(
								value=None,
								start=(targetzeroidx+shift),
								stop=(targetzeroidx+shift+1),
								step=None,
								axis=histaxis,
						)
					
					#There could be entries which are set and which we
					#shift out and then forget. This can happen, when
					#one has e.g. 16 MAC Operations and assumes that the maximum
					#result is having the maximum weight/act position
					#with value 3 set 16 times. positionweightsonehot is set then and
					#that makes sense. But if before when adding along MAC
					#axis one use scale (which invokes clip), the maximum
					#weight/act creates clip(16*3*scale, 16) = 16 as a value.
					#But if not the maximum value was set 16 times, but 
					#values 2 and 3 only 8 times, you get
					#clip(8*3*scale, 16) + clip(8*2*scale, 16), which can
					#be higher than the former result. This means: you spend
					#weights/act with less value and get a bigger result
					#due to combination of sclaed/clipped sub-results.
					#We detect that and raise an exception or write
					#the sum of ignored digits to a single traget bin.
					#All this does not apply, if the histogram we move
					#anyhow is only the zero bin with all the probability.
					if (shift != 0) and (not simplifytarget):
						
						#Whether we raise an exception on shifted-out
						#digits of value.
						#Cases where to check for digits being shifted
						#out of hist: In stochatic computation, we even
						#need the shifting out of values in one-hot
						#posweights, as that regards that one cannot add
						#e.g. 16 weight-value 1 and 16 weight-value 2, if
						#there are only 16 MACs.
						#In statistic computation, where bool values
						#never create impossible correlations causing too
						#large values, we check the condition.
						checkexception = ((not positionweightsonehot) or dostatisticdummy)
						
						#Whether we are in s ome stochastic computation, where
						#the shifting out is allowed to happen and we
						#compute a correction factor.
						computecorrection = (positionweightsonehot and (not dostatisticdummy))
						
						#Whether we put all the missing power into a
						#single last digits instead of scaling the other ones.
						placeinlast = False
						
						#Put correctie hist power in last bin instead of
						#scaling all bins.
						#placeinlast = computecorrection
						#computecorrection = False
						
						#The follwoing computations are expensive, do them
						#only if results are later needed.
						if checkexception or computecorrection or placeinlast:
							#Accessor for all elems, which are not read due
							#to shift
							targetaccessorignored = getValueAlongAxis(
									value=None,
									#If shift is positive, choose some upper elems
									start=(min(-shift, 0) or None),
									#Otherwise choose lower ones
									stop=(max(-shift, 0) or None),
									step=1,
									axis=histaxis,
							)
							#The actually ignored digits
							ignoreddigits = target[targetaccessorignored][chunkaccessor]
							#Combination of all the digits, which would be 
							#shifted out resulting in their total value.
							sumofignoreddigits = np.sum(
									ignoreddigits,
									axis=histaxis,
									keepdims=True,
									dtype=target.dtype,
							)
							
						#Combined with whether they really would be
						#shifted. This is the actual multiplication, that
						#the shifted hist only makes it into traget, if
						#the corresponding source value is actually set.
						#Note that tosumval has length 1 along reduce and
						#old hist axes.
						if checkexception or placeinlast:
							if dostatisticdummy:
								weightedsumofignoreddigits = sumofignoreddigits & tosumval[chunkaccessor]
							else:
								weightedsumofignoreddigits = sumofignoreddigits * tosumval[chunkaccessor]
						else:
							weightedsumofignoreddigits = None

						
						#Check whether digits of value are shifted out, but
						#it should not hvae happened.
						if checkexception:
							outofboundcount = np.count_nonzero(
									weightedsumofignoreddigits,
									axis=None,
									keepdims=False,
							)
							if outofboundcount > 0:
								raise ValueError(
										f"{outofboundcount} digits are shifted "
										f"outside the known value range.",
										outofboundcount,
								)
						
						#And acessor to last digit which is in innertarget.
						#This is easier. If the shift is positive, we shifted
						#histogram up and can write the last digit in
						#innertarget.
						#This is only needed, if you want to compute write
						#the value of the shifted-out digits into a last
						#digits which is actually inside target.
						if placeinlast:
							clipdigitidx = (max(shift, 0) and -1) or (min(shift, 0) and 0)
							innertargetaccessorclipdigit = getValueAlongAxis(
									value=None,
									#Read single elem, but if last elem is read,
									#-1+1=0 and we cannot use 0 as stop, we 
									#want None them to read the last elem only.
									start=clipdigitidx,
									stop=((clipdigitidx + 1) or None),
									step=1,
									axis=histaxis,
							)
						else:
							innertargetaccessorclipdigit = None
						
						#But if we are in stochastic computation and actually
						#see that there is some probability shifted out, we
						#correct the other digits by a factor to stil have
						#a sum-1 histogram. In practice, these digits should
						#hold extremely low probability, because they
						#describe having a large number of multiple weight/act
						#values set.
						if computecorrection:
							sumcorrection = sumofignoreddigits
							sumcorrection = np.subtract(
									1,
									sumcorrection,
									dtype=target.dtype,
							)
						else:
							sumcorrection = None
					
					#If there is no shift, we have no ingored digits and
					#all the stuff above does not apply.
					else:
						weightedsumofignoreddigits = None
						innertargetaccessorclipdigit = None
						sumcorrection=None
					
					#Now write shifted versions of target into innertarget.
					#Always work only on chunks we gathered the shift for.
					if dostatisticdummy:
						#First the more complex case, where the histogram
						#which is moved by shift is more than just a set
						#0 digit
						if not simplifytarget:
							#If the upper accessors are non-empty, apply (OR)
							#shifted target in innertarget. But the sfhifted
							#version, which represents an added value, is only
							#regarded if the value we add up right now is actually
							#set.
							if ((newbincount - abs(shift)) > 0):
								innertarget[innertargetaccesorupper][chunkaccessor] |= \
										target[innertargetaccessorlower][chunkaccessor] & tosumval[chunkaccessor]
							
							#Now also write the digits which have been shifted
							#out into one clip digit. In practice, sumofignoreddigits
							#is always None, but this is what one would need to
							#write shifted-out digits inot final ones.
							if (weightedsumofignoreddigits is not None) and (innertargetaccessorclipdigit is not None):
								innertarget[innertargetaccessorclipdigit][chunkaccessor] |= \
										weightedsumofignoreddigits
										
						#In the simple case, the histogram which is shifted has
						#only the zero digit set. Nothing is shifted out
						#and if there is only a single source bin set,
						#there is also only single target bin set.
						#The innertargetaccessor was already updated to
						#access a single bin only.
						else:
							innertarget[innertargetaccesorupper][chunkaccessor] |= \
									tosumval[chunkaccessor]
						
					#Compute with probabilities just like with bits.
					else:
						if not simplifytarget:
							#There is only one small difference compared to
							#boolean computations: there could be cases, where
							#we shift out hist power and it does not make
							#it to innertarget. If there is a correction factor,
							#we apply it to still have a sum-1 histogram.
							targetsumcorrected = target[innertargetaccessorlower][chunkaccessor]
							if sumcorrection is not None:
								#If you divide by 0 here, it means that all
								#histogram power is in the shifted-out digits.
								#But that should never happen. Getting a NaN
								#then is ok.
								targetsumcorrected = np.divide(
										targetsumcorrected,
										sumcorrection,
										dtype=target.dtype,
								)
							if ((newbincount - abs(shift)) > 0):
								innertarget[innertargetaccesorupper][chunkaccessor] += \
										targetsumcorrected * tosumval[chunkaccessor]
										
							if (weightedsumofignoreddigits is not None) and (innertargetaccessorclipdigit is not None):
								innertarget[innertargetaccessorclipdigit][chunkaccessor] += \
										weightedsumofignoreddigits
										
						else:
							innertarget[innertargetaccesorupper][chunkaccessor] += \
									tosumval[chunkaccessor]
			
			#All ist values of the current reduceax have been combined into
			#innertarget. This is basically our new target.
			#nnertarget includes all chunks, so we can again work on all
			#chunks in parallel.
			#Only do this, if we don't disable updating the target to
			#read updated version in the next iteration. Otherwise, we
			#keep the result within inn ertarget and let it read the
			#dummy version of target and let it shift a single set
			#value according to okne-hot setup.
			#Explicitly delete the target before overwriting the reference,
			#otherwise we might accumulate many huge arrays.
			#If reducecarry is enabled, simplifytarget is anyhow never
			#set, so there is no need for an additional check here.
			if not disablereducecarry:
				del target
				target = innertarget
			
		#If target was never updated, update it now. Onl delete the
		#old one if it existed
		if disablereducecarry:
			if not simplifytarget:
				del target
			target = innertarget
						
	return target


def getHistStddev(
		target,
		maxhistvalue,
		dostatistic,
		dostatisticdummy,
		histaxis,
		stataxis,
		stddevdtype,
	):
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=target.ndim)
	stataxis, = normalizeAxes(axes=stataxis, referencendim=target.ndim)
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	#Length of stat axis for variance computation
	statlen = target.shape[stataxis]
	
	#First find old standard deviation. The mean is given by us,
	#because we treat only positive values.
	#We force a float dtype, because uint or bool wont suffice.
	#A standard deviation shall be found over statistics axis.
	#Also the histogram axis should vanish in the process.
	#That way, both statistic and stochastic operations yield a result.
	stddevshape = list(target.shape)
	stddevshape[stataxis] = 1
	stddevshape[histaxis] = 1
	stddevmean = np.zeros(shape=stddevshape, dtype=stddevdtype)
	
	#First check cases, where we have statistic data
	if dostatistic:
		targetstddev = target
		#In dummy statistic, turn the hist of bools into uint
		if dostatisticdummy:
			targetstddev = packHist(
					topack=targetstddev,
					axis=histaxis,
					keepdims=True,
					strict=True,
			)
		#Now get the standard deviation over uints. The statistic and
		#hist axes have length 1 after this.
		#This call only works for new np versions, which know the
		#mean= parameter
		#stddev = np.var(
		#		targetstddev,
		#		axis=stataxis,
		#		dtype=stddevdtype,
		#		keepdims=True,
		#		mean=stddevmean,
		#)
		#So instead compute by hand. SUbtract mean, square differences,
		#sum them up und divide by number. All with known dtype.
		stddev = np.subtract(targetstddev, stddevmean, dtype=stddevdtype)
		np.square(stddev, out=stddev)
		stddev = np.sum(
				stddev,
				axis=stataxis,
				dtype=stddevdtype,
				keepdims=True,
		)
		np.divide(stddev, statlen, out=stddev)
		
		#Still, if we e.g. quantize histograms along different activation
		#value positions, they all need a same scale. So we have to also
		#reduce these axes. And we do so using a mean.
		#This mean is only suitable on variances, not standard
		#deviations and that is why we did not compute standard
		#deviation in the first place.
		stddev = np.mean(
				stddev,
				axis=None,
				keepdims=True,
				dtype=stddevdtype,
		)
		#Now place sqrt after having done the mean
		np.sqrt(stddev, out=stddev)
		
	#Otherwise, get standard deviation from histogram
	else:
		
		#This is here only called in stochastic computation and we
		#then are protected against this method returning None anywhere.
		oldhistlen, oldhistvalues = getHistLenFromMaxValues(
				target=target,
				maxhistvalue=maxhistvalue,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				histaxis=histaxis,
		)
		
		#Distance of values to mean. Use already float, because 
		#in future, the mean could be a non-zero float, too.
		histvaluetomean = np.subtract(
				oldhistvalues,
				stddevmean,
				dtype=stddevdtype,
		)
		#THe variance is the square of the distance to mean and a sum
		#over these weighted by their porbability. But the distance
		#could be a huge nuumber, the probability a small one. So we
		#multiply them mixed:
		stddev = np.multiply(
				histvaluetomean,
				target,
				dtype=stddevdtype,
		)
		np.multiply(
				stddev,
				histvaluetomean,
				out=stddev,
		)
		#Sum the weighted squares and set histaxis len to 1
		stddev = np.sum(
				stddev,
				axis=histaxis,
				keepdims=True,
				dtype=stddevdtype,
		)
		#Same problem as above: we need to get a single value for
		#all histograms, if we later want to combine them in another
		#add operation with correct weights.
		stddev = np.mean(
				stddev,
				axis=None,
				keepdims=True,
				dtype=stddevdtype,
		)
		#We so far only have a variance.
		np.sqrt(stddev, out=stddev)
		
	return stddev

def quantizeClipScaleValues(
		toprocess,
		maxhistvalue,
		mergevalues,
		dostatistic,
		dostatisticdummy,
		cliplimitfixed,
		valuescale,
		histaxis,
		stataxis,
		scaledtype,
	):
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=toprocess.ndim)
	stataxis, = normalizeAxes(axes=stataxis, referencendim=toprocess.ndim)
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	oldhistlen, oldhistvalues = getHistLenFromMaxValues(
			target=toprocess,
			maxhistvalue=maxhistvalue,
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
			histaxis=histaxis,
	)
	
	#Whether we have real histograms and not just uint numbers we can
	#process by scale/round.
	processhist = (not dostatistic) or dostatisticdummy
	
	if (oldhistlen is None) and (mergevalues is not None):
		raise ValueError(
				f"mergevalues is applied with value {mergevalues}, but "
				f"there was no way of finding oldhistlen. Is "
				f"maxhistvalue {maxhistvalue} None?",
				mergevalues,
				maxhistvalue,
		)
		
	if (cliplimitfixed is not None) and (valuescale is not None):
		raise ValueError(
				f"cliplimitfixed is {cliplimitfixed} and valuescale is "
				f"{valuescale}, but only one of the two can be given.",
				cliplimitfixed,
				valuescale,
		)
	
	#Mergevalues needs to be outside the (-1;1) range and we cannot merge
	#all bins intoa  single one, as there need to be at least two
	#resulting bins. We also have to block mergevalues=-1, because
	#that also wants to keep only a single bin and is like merging
	#all old bins into a single one.
	#Mergevalues is a divider to given values, so it works on
	#magnitudes, so we refer it to histlen, not bincount
	#No need to check this, if mergevalues was not given.
	if (mergevalues is not None) and (((mergevalues < 1) and (mergevalues > -1)) or (mergevalues >= oldhistlen)):
		raise ValueError(
				f"mergevalues needs to be larger than or equal to 1 "
				f"or smaller than or equal to -1 "
				f"and smaller than oldhistlen {oldhistlen}, but "
				f"{mergevalues} was given.",
				oldhistlen,
				mergevalues,
		)
		
	#If given, mergevalues as well as cliplimitfixed and valuescale are
	#float. Extract that from np value
	if mergevalues is not None:
		mergevalues = float(np.squeeze(mergevalues).item())
	if cliplimitfixed is not None:
		cliplimitfixed = float(np.squeeze(cliplimitfixed).item())
	if valuescale is not None:
		valuescale = float(np.squeeze(valuescale).item())
		
	#Turn negative values into positive ones
	if (mergevalues is not None) and (mergevalues <= -1):
		keepvalues = -mergevalues
		#Very negative values are clipped to simply keep the whole
		#histogram.
		keepvalues = min(keepvalues, oldhistlen)
		#Luckily, mergevalues can be a float number. We choose one,
		#which will reach teh good number of bins from known
		#number of old bins.
		posmergevalues = float(oldhistlen) / float(keepvalues)
		mergevalues = posmergevalues
		
	#If values will be gained by factor 2 and then always two values
	#shall be merged together, nothing happens. So the mergevalues
	#are corrected by cliplimitfixed. But mergevalues will never be
	#below 1. And also regard, that a cliplimitfixed below 1.
	#changes nothing.
	if (mergevalues is not None) and (cliplimitfixed is not None):
		mergevalues = float(mergevalues) / max(float(cliplimitfixed), 1.)
		mergevalues = max(mergevalues, 1.)
	
	#We will now scale and round target and maxhistvalue. We implement
	#a way of scale/round histograms and int numbers. They are actually
	#very similar, as the histogram computation first derives a value
	#vector, then processes it like the int numbers and then uses it as
	#index.
	#A case can also re-derive mergevalues from rounding and we store that
	#for return in a special result
	mergevaluesret = mergevalues
	scalemergecases = (
		#For the given input value, we maybe have to do hist computations
		(toprocess, processhist, False),
		#maxhistvalue is always int. This is also a good, reproducable
		#computation to re-derive mergevalues.
		(maxhistvalue, False, True),
	)
	scaledmergedvalues = list()
	
	for thistarget, thisprocesshist, updatemergevalues in scalemergecases:
		
		#If the given value to scale is None, keep None. Needed for
		#maxhistvalue being None
		if thistarget is None:
			scaledmergedvalues.append(None)
			continue
		
		#Only updat mergevalues, if mergevalues are given
		updatemergevalues = updatemergevalues and (mergevalues is not None)
		
		#If there is nothing to do, just return a copy.
		#This always returns a copy.
		if (mergevalues is None) and (cliplimitfixed is None) and (valuescale is None):
			scaledmergedvalues.append(thistarget.copy())
			continue
		
		#For int numbers without hist axis, take the value as-is to scale and merge
		if not thisprocesshist:
			toprocess = thistarget
		#For histogram computations, take the value vector and hence do the
		#same computations on the values represneted by histogram
		else:
			toprocess = oldhistvalues
			
		#Remember dtype to go possibly back to after rounding
		olddtype = toprocess.dtype

		#First maybe apply scale and clip to get less quant error.
		#DO not clip with the single oldhistlen, but rather with the
		#maxhistvalue, which can regard limited histograms along chunks.
		#Still, the same cliplimitfixed is used for all chunks to enable
		#to combine them properly later.
		#Do not actually multiply values, but rather only do the corresponding
		#clip resulting from a cliplimitfixed, because the the spacing between
		#values is then still 1 and there is no need to call round(), which would
		#move bins.
		#If we would do the actual scaling, we would have bins being spaced
		#by some larger distance, but we anyhow never carry that information.
		#A cliplimit of 2 clips histlen at half the magnitude.
		#A cliplimit of 0.5 changes nothing.
		if cliplimitfixed is not None:
			#Compute to which value too large values are clipped.
			largevalidhistval = np.divide(maxhistvalue, cliplimitfixed, dtype=scaledtype)
			#This needs to be rounded, because we will only support int values,
			#to stay in int value spacing
			np.round(largevalidhistval, out=largevalidhistval)
			largevalidhistval = largevalidhistval.astype(toprocess.dtype)
			#NOw clip while returning a copy of same dtype. toprocess
			#has not been copied yet
			toprocess = np.clip(
					toprocess,
					a_min=-largevalidhistval,
					a_max=largevalidhistval,
					dtype=toprocess.dtype,
			)
			
		#A value scale is an alternative to cliplimit. It does not clip
		#and can even increase the bincount, when it scales up. It also
		#really scales up/down bin positions and then ensures that
		#there is a round operation on bin indices
		if valuescale is not None:
			#Copy from some int to float dtype
			toprocess = np.multiply(
					toprocess,
					valuescale,
					dtype=scaledtype,
			)
			#mergevalues is untouched. Simply by definition.
			
		#If we later re-derive mergevalues, remember toprocess here.
		#Or more exact: the maximum value.
		if updatemergevalues:
			updatemergevaluesref = np.max(
					toprocess,
					axis=None,
					keepdims=False,
			)
			
		#Then do the down division, which will casue different
		#values to get closer ogether to be rounded to the
		#same int.
		#Rounding is made, because teh division brings us to float dtype,
		#but we need integer values of bins. This then also merges power
		#of multiple bins into a single one, which is what quantization does.
		if mergevalues is not None:
			if valuescale is None:
				#Introduce some float dtype in division when coming from original
				#int or the int type kept in cliplimit
				toprocess = np.divide(
						toprocess,
						mergevalues,
						dtype=scaledtype,
				)
			else:
				#Or already have scaledtype and a copy of what was given in
				#the beginning fo the loop iteration and reuse that memory
				np.divide(
						toprocess,
						mergevalues,
						out=toprocess,
				)
				
		#Round and go back to int dtype, if it was introduced. If we
		#have a scaleddtype, that is a copy of waht we started with in
		#this loop iteration.
		if (valuescale is not None) or (mergevalues is not None):
			np.round(toprocess, out=toprocess)
			toprocess = toprocess.astype(dtype=olddtype)
			
		#If we shall update mergevalues, we now re-derive mergevalues
		#from rounded result. mergevalues is a factor applied on
		#updatemergevaluesref and then we round. We now calculate which
		#mergevalues would generate the same result, but even before
		#rounding. mergevalues is a single digit, so we only work on
		#max values. A maxvalue should give the most precision.
		#If mergevalues is only updated for computed maxhistvalue, the
		#result is reliable in statistic vs. stochastic.
		if updatemergevalues:
			mergevaluesret = np.max(toprocess, axis=None, keepdims=False)
			mergevaluesret = np.divide(
					updatemergevaluesref,
					mergevaluesret,
					dtype=scaledtype,
			)
			mergevaluesret = float(np.squeeze(mergevaluesret).item())
				
			
		#We now have some copied ndarray with same dtype as toprocess
		#had in the beginning.
		
		#For uint computation, we are done now. We even kept the dtype.
		if not thisprocesshist:
			scaledmergedvalues.append(toprocess)
			
		#Otherwise, we use the re-computed histogram representation as
		#index array.
		else:
		
			#The new histogram length is its maximum value which is
			#the maximum value to which an old value makes it.
			newhistlen = int(round(np.max(toprocess, axis=None, keepdims=False)))
			#Also derive bincount
			newbincount = histlenToBincount(histlen=newhistlen)
		
			#Design new target value
			newtargetshape = list(thistarget.shape)
			newtargetshape[histaxis] = newbincount
			newtarget = np.zeros(shape=newtargetshape, dtype=thistarget.dtype)
		
			#toprocess is now a map showing for each old hist value
			#where it propagates to into the new, (shorter) hist.
			#We now also need to know the integer values of these new
			#bins.
			#Add lower dims to achieve compatibility with getValueALongAxis
			newhistvalues = getHistValues(
					bincount=newbincount,
					axis=histaxis,
					ndim=thistarget.ndim,
					padlower=True,
					padupper=True,
			)
		
			#Go along all new values in histogram and sum up all values
			#from old histogram, which make it inot this new bin
			for newhistidx in range(newbincount):
				#At some new hist index, we want all values with
				#corresponding integer and we look them up from values
				#related to all new bins.
				newhistaccessor = getValueAlongAxis(
						value=None,
						start=newhistidx,
						stop=newhistidx+1,
						step=None,
						axis=histaxis,
				)
				newvalue = np.sum(
						thistarget,
						axis=histaxis,
						keepdims=True,
						dtype=thistarget.dtype,
						where=(newhistvalues[newhistaccessor]==toprocess),
				)
				if dostatistic:
					newtarget[newhistaccessor] |= newvalue
				else:
					newtarget[newhistaccessor] += newvalue
					
			#And this is now our scaled and merged results
			scaledmergedvalues.append(newtarget)
		
	#Read scaled/merged given value and its maxhistvalue
	target, maxhistvalue = scaledmergedvalues
		
	return target, maxhistvalue, cliplimitfixed, mergevaluesret

def reduceSum(
		tosum,
		dostatistic,
		dostatisticdummy,
		selfcheckindummy,
		reduceaxes,
		chunksizes,
		mergevalues,
		cliplimitstddev,
		cliplimitfixed,
		positionweights,
		positionweightsonehot,
		disablereducecarries,
		chunkoffsetsteps,
		histaxis,
		maxhistvalue,
		stataxis,
		#If histaxis is created, all passed axis indices shall match
		#axes AFTER adding histaxis.
		docreatehistaxis,
		mergeeffortmodel,
		allowoptim,
):
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	#We can create the histogram axis. Keep maxhistvalue synchronous in
	#shape.
	if docreatehistaxis:
		tosum = np.expand_dims(tosum, axis=histaxis)
		#Also derive 0 probability, such that the sum over values along
		#hist axis is 1.
		tosumzeroprob = None
		#In stochastic computing
		if not dostatistic:
			tosumzeroprob = 1 - tosum
		#In statistic computing, compute with booleans. But only if the
		#hist length i really used instead of int computations.
		elif dostatisticdummy:
			tosumzeroprob = ~tosum
		#If we have the zero prob, use it.
		if tosumzeroprob is not None:
			tosum = np.concatenate(
					(tosumzeroprob, tosum),
					axis=histaxis,
					dtype=tosum.dtype,
			)
		#Keep maxhistvalue synchronous
		if maxhistvalue is not None:
			maxhistvalue = np.expand_dims(maxhistvalue, axis=histaxis)
		
	#Normalize axis indices to positive ones.
	#reduceaxes will be a 1D array, but we expect a single elem for
	#histaxis.
	reduceaxes = normalizeAxes(axes=reduceaxes, referencendim=tosum.ndim)
	histaxis, = normalizeAxes(axes=histaxis, referencendim=tosum.ndim)
	stataxis, = normalizeAxes(axes=stataxis, referencendim=tosum.ndim)
	
	#Normalize input arguments, which should be given as often as we have
	#axes to reduce
	normalizecases = (
			(positionweights, "positionweights",),
			(positionweightsonehot, "positionweightsonehot",),
			(disablereducecarries, "disablereducecarries",),
			(chunkoffsetsteps, "chunkoffsetsteps",),
			(chunksizes, "chunksizes",),
	)
	normalizevalues = list()
	for normalizevalue, normalizename in normalizecases:
		if normalizevalue is None:
			normalizevalue = (None,) * reduceaxes.size
		if len(normalizevalue) != reduceaxes.size:
			raise IndexError(
					f"Expected number of {normalizename} and number of axes to "
					f"reduce to be equal, but got {len(normalizevalue)} and "
					f"{reduceaxes.size}.",
					normalizename,
					normalizevalue,
					reduceaxes,
			)
		normalizevalues.append(normalizevalue)
		
	#Apply values, which have been checked for length and where None was
	#repalced by repeated value.
	positionweights, positionweightsonehot, disablereducecarries, chunkoffsetsteps, chunksizes = \
			normalizevalues
		
	if histaxis == stataxis:
		raise ValueError(
				f"Statistics and histogram axis are not allowed to be the "
				f"same, but are {stataxis} and {histaxis}.",
				stataxis,
				histaxis,
		)
		
	#It is not allowed to pass cliplimitstddev and cliplimitfixed, as cliplimitfixed
	#is derived from cliplimitstddev
	if (cliplimitstddev is not None) and (cliplimitfixed is not None):
		raise ValueError(
				f"It is not allowed to make cliplimitstddev {cliplimitstddev} and "
				f"cliplimitfixed {cliplimitfixed} both non-None. Cliplimitfixed "
				f"is derived from cliplimitstddev.",
				cliplimitstddev,
				cliplimitfixed,
		)
			
	#Replace default ADC models by callables, which get called with the
	#number of levels they right now convert and which return float.
	#The analog model raises effort linearly with the number of levels,
	#digital raises linear with the number of bits.
	#Float conversion happens later, but we have to sum over multiple ADC
	#calls.
	if mergeeffortmodel == "analog":
		mergeeffortmodel = lambda x: np.sum(x)
	elif mergeeffortmodel == "digital":
		mergeeffortmodel = lambda x: np.sum(np.log2(x, dtype="float"))
		
	#Whether we later call probabilistic adder. We don't if we don't work
	#on histograms and just do a simple uint computation in here.
	calledadder = (not dostatistic) or dostatisticdummy
		
	#Will now sum up axis by axis here. Between iterations, keep number
	#of axes and remember what to squeeze for the end.
	target = tosum
	squeezedims = list()
	for thisreduceax, thisposweights, thisposweightonehot, thisdisablereducecarry, thischunkoffsetstep, thischunksize in zip(reduceaxes, positionweights, positionweightsonehot, disablereducecarries, chunkoffsetsteps, chunksizes):
		#Load histogram axis. Will possibly be updated, if we add other
		#axes.
		thishistax = histaxis
			
		if thishistax == thisreduceax:
			raise ValueError(
					f"Histogram and reduce axis are not allowed to be the "
					f"same, but are {thishistax} and {thisreduceax}.",
					thishistax,
					thisreduceax,
			)
			
		if stataxis == thisreduceax:
			raise ValueError(
					f"Statistics and reduce axis are not allowed to be the "
					f"same, but are {stataxis} and {thisreduceax}.",
					stataxis,
					thisreduceax,
			)
		
		#Positionweights can be an argument and a str. Unpack if given.
		#same and None support setting the same value as weight
		#on all.
		#Hist supports a stepsize with which positionweights arange rises.
		#This is only a future-proof feature and is not tested.
		thispositionweightsarg = None
		knownposweightswithargs = ("same", None, "hist",)
		if isinstance(thisposweights, tuple) and \
				(len(thisposweights) == 2) and \
				(thisposweights[0] in knownposweightswithargs):
			thisposweights, thispositionweightsarg = thisposweights
		
		#The axis over which we sum is now replaced by an axis over chunks
		#and one inside chunks. We remember the added axis idx and whether
		#we used only a dummy chunk. Dummy chunks will be removed towards
		#the very end.
		overchunkax = thisreduceax
		insidechunkax = thisreduceax + 1
		maxchunksize = tosum.shape[thisreduceax]
		
		#The index of the histogram axis possibly changed, because we
		#added another axis.
		if thishistax >= insidechunkax:
			thishistax = thishistax + 1
			
		#Now actually introduce the new axis
		if thischunksize is None:
			#One chunk size for each reduced dimension.
			thischunksize = maxchunksize
			#And there is only a single chunk
			thischunkcount = 1
			#The chunk dim will be removed towards the end.
			dopurgeoverchunkaxis = True
			#Add the overchunkax by expanding dims, which costs no memory.
			#So all items to reduce are then in the innerchunkaxis following
			#the overchunkaxis having length 1.
			target = np.expand_dims(target, axis=overchunkax)
			#Keep maxvalue synchronous if given
			if maxhistvalue is not None:
				maxhistvalue = np.expand_dims(maxhistvalue, axis=overchunkax)
		else:
			#Some actual chunksize we will use. Enforce type and shape.
			thischunksize = int(thischunksize)
			if thischunksize < 1:
				raise ValueError(
						f"Got chunksize {thischunksize}, but only "
						f"values of at least 1 are allowed.",
						thischunksize,
				)
			#Limit chunksize to have all data in a single chunk
			thischunksize = min(thischunksize, maxchunksize)
				
			#THe overchunkaxis is then kept in the end.
			dopurgeoverchunkaxis = False
			
			#Reshape underlying array to inlcude inside chunk ax.
			#The reduce ax is removed and replaced by automatically found
			#overchunk dim and the used-set inside-chunk dim.
			#Skipped, because this one needs all elements to split exactly
			#into the chunks.
			#target = np.reshape(target, (
			#		*target.shape[:thisreduceax],
			#		-1,
			#		thischunksize,
			#		*target.shape[thisreduceax+1:],
			#))
			
			#Instead, we derive indices to draw elems along reduce
			#ax from on our own. The inner index is the one which
			#updates faster.
			inneridx = np.arange(
					start=0,
					stop=thischunksize,
					step=1,
					dtype="uint",
			)
			inneridx = np.expand_dims(inneridx, axis=0)
			overidx = np.arange(
					start=0,
					stop=maxchunksize,
					step=thischunksize,
					dtype="uint",
			)
			overidx = np.expand_dims(overidx, axis=-1)
			#Combine the inner and over indices, which already have
			#been made compatible in ndim
			totalidx = inneridx + overidx
			
			#So far, everything is exactly as with np.reshape. But if
			#we do not have all chunks full, we get elems in totalidx,
			#which would do out-of-bound index accesses. And these
			#are prevented now by masking them.
			totalidxbad = totalidx >= maxchunksize
			
			#Set bad indices to something which works.
			np.putmask(totalidx, mask=totalidxbad, values=0)
			
			#The indices access from reduce axis and replace reudce ax
			#by outer and inn er chunk ax
			totalaccessor = (slice(None, None, None),) * thisreduceax
			totalaccessor = (*totalaccessor, totalidx,)
			
			#Do the access
			target = target[totalaccessor]
			#Also keep max hist value synchronous. The accessor keeps the
			#statistics dim as-is and that is ok.
			if maxhistvalue is not None:
				maxhistvalue = maxhistvalue[totalaccessor]
			
			#Create a padded version of the mask being index compatible
			#with target
			#the current mask already includes over and inner chunk axis
			totalidxbadpadded = padAxes(
					value=totalidxbad,
					innertoaxis=insidechunkax,
					referencendim=target.ndim,
					padlower=False,
					padupper=True,
			)
			
			#Unset the enttries, which were out of bound and which
			#lookup from index 0 just to not crash. We cannot set digits to
			#0, we have to set histograms to zero meaning to set the
			#0 bin if actual hist adder is later called.
			#Refer to thishistax, which already has been updated,
			#because target already had a new axis introduced.
			if calledadder:
				zerohistforpadding = getHistValues(
						bincount=target.shape[thishistax],
						axis=thishistax,
						ndim=target.ndim,
						padlower=False,
						padupper=True,
				)
				zerohistforpadding = (zerohistforpadding == 0)
			#Otherwise, we set the corresponding digits simply to int val 0.
			else:
				zerohistforpadding = 0
			#Use copyto, as that supports broadcasting.
			np.copyto(
					target,
					zerohistforpadding,
					where=totalidxbadpadded,
			)
			#AGain keep maxhistvalue synchronous and keep statistics axis as is.
			#maxhistvalue is treated as ifthe adder was never called, because
			#it works on uint.
			if maxhistvalue is not None:
				np.copyto(
						maxhistvalue,
						0,
						where=totalidxbadpadded,
				)
			
			#Find chunkcount from what the target can hold.
			thischunkcount = target.shape[overchunkax]
		
		#Preprocess the position weights. None or "same" needs to create an
		#all-1 vector, "bits" needs to create a power of two vector.
		#This will later be used as an index to roll matrices, so we need
		#some index dtype.
		#We do not weight along histaxis, as this is a historgam and basta,
		#but we rather weight along the axis we reduce.
		positionweightslen = thischunksize
		positionweightsdtype = "int"
		positionweightsarehist = False
		#None or same weights all entries equally. One can tune which same
		#weight is used for all.
		if thisposweights in ("same", None,):
			thisposweights = np.ones(
					shape=(positionweightslen,),
					dtype=positionweightsdtype,
			)
			if thispositionweightsarg is not None:
				np.multiply(
						thisposweights,
						thispositionweightsarg,
						out=thisposweights,
				)
		#hist means: also the axis we read from is a histogram axis. This is
		#needed to merge multiple histogram axes.
		#Histogram values are well defined
		elif thisposweights == "hist":
			#Posweights are histogram. Prevent padding.
			thisposweights = getHistValues(
					bincount=positionweightslen,
					axis=0,
					ndim=1,
					padlower=False,
					padupper=True,
			)
			#Use explicit datatype from above
			thisposweights = thisposweights.astype(dtype=positionweightsdtype)
			#If an argument to hist wsa given, it is the stepsize and we
			#use that by multiplying.
			if thispositionweightsarg is not None:
				np.multiply(thisposweights, thispositionweightsarg, out=thisposweights)
			#We later need to know that this was a historgram posweight
			positionweightsarehist = True
		#Everything else is kept as vector
		else:
			thisposweights = np.array(thisposweights, dtype=positionweightsdtype)
			if thisposweights.shape != (positionweightslen,):
				raise ValueError(
						f"thisposweights should be 1D array with length "
						f"{positionweightslen}, but shape "
						f"{thisposweights.shape} was found.",
						positionweightslen,
						thisposweights.shape
				)
				
		#Positionweights encode one-hot, if a histogram was given.
		#We never overwrite a decision a user gave.
		if thisposweightonehot is None:
			thisposweightonehot = positionweightsarehist
		else:
			thisposweightonehot = bool(thisposweightonehot)
			
		#positionoffsets can also be given and a stepsize is transalted
		#into an arange synchronous to chunksize. The sum of chunkoffset and
		#chunked posweights then yields unchunked posweights.
		#If thischunkcount is 1, we essentially also need no offset steps.
		#There is only a single chunk and we weight it with posweights only
		#then.
		if (thischunkoffsetstep is not None) and (thischunkcount > 1):
			thischunkoffsets = getHistValues(
					bincount=thischunkcount,
					axis=0,
					ndim=1,
					padlower=False,
					padupper=True,
			)
			np.multiply(thischunkoffsets, thischunkoffsetstep, out=thischunkoffsets)
		else:
			thischunkoffsets = np.zeros(
					shape=(thischunkcount,),
					dtype=thisposweights.dtype,
			)
			
		#Set to also compute stochastic results using uint computation.
		#Use this, if dostatistic works, but dostatisticdummy does not.
		selfcheck = selfcheckindummy and dostatistic and dostatisticdummy
		#Remember target before possibly overwriting it
		targetuint = target
		
		#Expand posweights and offsets to be broadcastable with target
		thisposweightspadded = padAxes(
				value=thisposweights,
				innertoaxis=insidechunkax,
				referencendim=targetuint.ndim,
				padlower=False,
				padupper=True,
		)
		thischunkoffsetspadded = padAxes(
				value=thischunkoffsets,
				innertoaxis=overchunkax,
				referencendim=targetuint.ndim,
				padlower=False,
				padupper=True,
		)
		
		#Compute the combined posweights on our own. Will need that in
		#uint computation or for maxhistval
		combinedweights = thisposweightspadded + thischunkoffsetspadded
		
		#Update the running
		#fullscale result. Stat axis should be 1 here.
		if maxhistvalue is not None:
			
			#If we add another, older histogram axis, it already has
			#length 1 along reduce axis. There is then  no point in
			#using positionweights. In that case, we use only chunkoffset.
			
			#Give maxhistvalue and combinedweights same ndim
			combinedweightsmaxhistvalue = combinedweights
			combinedweightsmaxhistvalue = padAxes(
					value=combinedweightsmaxhistvalue,
					innertoaxis=-1,
					referencendim=maxhistvalue.ndim,
					padlower=True,
					padupper=False,
			)
				
			#If maxhistlen has an axis, which is the reduced inside chunk
			#axis, which has length 1 (because it used to be a hist axis)
			#and if it is weighted with hist (as one should do with an
			#old hist axis), we assume that all positionweights are 1.
			#There is no weighting along the old hist axis, because it is
			#uint already.
			combinedweightsaxlen = combinedweightsmaxhistvalue.shape[insidechunkax]
			maxhistvalueaxlen = maxhistvalue.shape[insidechunkax]
			if (combinedweightsaxlen != maxhistvalueaxlen) and (maxhistvalueaxlen == 1) and positionweightsarehist:
				combinedweightsmaxhistvalue = np.add(
						thischunkoffsetspadded,
						1,
						dtype=combinedweightsmaxhistvalue.dtype,
				)
			
			#Double-check, that broadcasting with new combinedweights would
			#not increase array size.
			combinedweightsmaxhistvalue = padAxes(
					value=combinedweightsmaxhistvalue,
					innertoaxis=-1,
					referencendim=maxhistvalue.ndim,
					padlower=True,
					padupper=False,
			)
			combinedweightsaxlen = (
					combinedweightsmaxhistvalue.shape[overchunkax],
					combinedweightsmaxhistvalue.shape[insidechunkax],
			)
			maxhistvalueaxlen = (
					maxhistvalue.shape[overchunkax],
					maxhistvalue.shape[insidechunkax],
			)
			if combinedweightsaxlen != maxhistvalueaxlen:
				raise RuntimeError(
						f"Broadcasted positionweights and chunkoffsets "
						f"have lengths {combinedweightsaxlen} over and "
						f"inside chunks, but maxhistvalue has "
						f"{maxhistvalueaxlen}."
						f"They have to be equal.",
						combinedweightsaxlen,
						maxhistvalueaxlen,
				)
			
			#Weight along over/inside chunk ax with correct posweights.
			maxsubresults = maxhistvalue * combinedweightsmaxhistvalue
			#The histograms are always symmetric around a 0 value and need
			#to be capable of holding both negative and positive magnitudes.
			#The larger MAGNITUDE describes the maxhistval.
			np.absolute(maxsubresults, out=maxsubresults)
			#In int computattion, we always could use np.sum(). But that is
			#because if the reduce axis was a one-hot, we would
			#just see a ingle set value long reduce axis. This
			#is different in full-scale computation. Here, we see
			#a full scale for all positions along reduce axis and
			#have to use the hottest one ourselves.
			if thisposweightonehot:
				maxhistvalue = np.max(
						maxsubresults,
						axis=insidechunkax,
						keepdims=True,
				)
			else:
				maxhistvalue = np.sum(
						maxsubresults,
						axis=insidechunkax,
						keepdims=True,
						dtype=maxhistvalue.dtype,
						)
			
		#Everything prepared, do the actual summation.
		if calledadder:
			target = probabilisticAdder(
					tosum=target,
					reduceaxis=insidechunkax,
					histaxis=thishistax,
					positionweights=thisposweights,
					positionweightsonehot=thisposweightonehot,
					disablereducecarry=thisdisablereducecarry,
					chunkoffsets=thischunkoffsets,
					overchunkaxis=overchunkax,
					dostatisticdummy=dostatisticdummy,
					allowoptim=allowoptim,
			)
		#The uint stochastic computation can be made only or for comparing
		#against probabilistic adder
		if (not calledadder) or selfcheck:
			#This is the statistic implementation.
			#No histograms, but only integer values.
			#Histogram axes have length 1 and will remain like that.
			
			#For debugging: turn during stochastic dummy given historgrams
			#into values and compute with them. If this is done, we first
			#have to turn a historgram into uint.
			if selfcheck:
				targetuint = packHist(
						topack=targetuint,
						axis=thishistax,
						keepdims=True,
						strict=True,
				)
			
			#Sum along weighted reduceax.
			#Can ignore hisogram axis, because it will have length 1 and in
			#return we use uint values and not value probabilities.
			subresults = targetuint * combinedweights
			targetuint = np.sum(
					subresults,
					axis=insidechunkax,
					keepdims=True,
					dtype=targetuint.dtype,
			)
			
			#The probabilistic finds values, which are too large
			#and raises an exception. We also do that here.
			outofbound = (targetuint > maxhistvalue) | (targetuint < -maxhistvalue)
			outofboundcount = np.count_nonzero(outofbound, axis=None, keepdims=False)
			if outofboundcount > 0:
				raise ValueError(
						f"{outofboundcount} digits are outside the expected "
						f"value range.",
						outofboundcount,
				)
			
			#Now we either have a new running result which we keep in the
			#uint domain because we compute everything there, or we 
			#pack the actual result also into uint, so we have something
			#to compare during debug.
			if selfcheck:
				targetpacked = packHist(
						topack=target,
						axis=thishistax,
						keepdims=True,
						strict=True,
				)
				badcount = np.count_nonzero(targetpacked != targetuint)
				#Fix linter warning
				badcount = int(badcount)
				targetpacked = targetpacked
				#Raise Error if self check failed
				if badcount > 0:
					raise RuntimeError(
							f"In self check, {badcount} digits differend "
							f"between stochasticdummy and stoachstic uint "
							f"computation.",
							badcount,
					)
			else:
				target = targetuint
			
		#Remove the insidechunkax, which now should be 1. Again keep
		#maxhistvalue in same shape
		target = np.squeeze(target, axis=insidechunkax)
		if maxhistvalue is not None:
			maxhistvalue = np.squeeze(maxhistvalue, axis=insidechunkax)
		#The overchunkax is kept and removed at the very end of all loop
		#iterations, otherwise the target shape changes between iterations.
		if dopurgeoverchunkaxis:
			squeezedims.append(overchunkax)
			
	#Check if our hist axis is now maybe too large, because we added some
	#residual chunks with hist entries which were never used, but which were
	#present to place e.g. two full and one residual hist in one array.
	#Do this now and not at the very end, because e.g. mergevalues already
	#wants to know some final bincount to e.g. compute the number of bins
	#to keep for.
	newbincount = target.shape[histaxis]
	newhistlen = bincountToHistlen(bincount=newbincount)
	#We will only remove, if all the different histograms want the same
	#and if we actually would remove smth.
	#We can only do that, if we have knowledge about the maximum
	#possible value.
	if maxhistvalue is not None:
		maxhistvaluemin = np.min(maxhistvalue, axis=None, keepdims=False)
		maxhistvaluemax = np.max(maxhistvalue, axis=None, keepdims=False)
		maxhistvaluemin = int(round(maxhistvaluemin))
		maxhistvaluemax = int(round(maxhistvaluemax))
		if (maxhistvaluemin == maxhistvaluemax) and (maxhistvaluemin < newhistlen):
			#This would be our new histlen: what max value says
			truncatedhistlen = maxhistvaluemin
			#Turn to bincount
			truncatedbincount = histlenToBincount(histlen=truncatedhistlen)
			
			#These values are represented by current hist
			newhistvalues = getHistValues(
					bincount=newbincount,
					axis=0,
					ndim=1,
					padlower=False,
					padupper=True,
			)
			#And these by the shortened version
			truncatedhistvalues = getHistValues(
					bincount=truncatedbincount,
					axis=0,
					ndim=1,
					padlower=False,
					padupper=True,
			)
			
			#By checking difference in value range, check how mcuh to remove
			#on lower and upper end.
			removelowerelems = np.min(
					truncatedhistvalues,
					axis=None,
					keepdims=False,
			) - \
			np.min(
					newhistvalues,
					axis=None,
					keepdims=False,
			)
			removeupperelems = np.max(
					newhistvalues,
					axis=None,
					keepdims=False,
			) - \
			np.max(
					truncatedhistvalues,
					axis=None,
					keepdims=False,
			)
			removelowerelems = int(round(removelowerelems))
			removeupperelems = int(round(removeupperelems))
			#We cannot add any elems. We will only remove.
			removelowerelems = max(removelowerelems, 0)
			removeupperelems = max(removeupperelems, 0)
			
			#Use accessor to remove elems
			target = getValueAlongAxis(
					value=target,
					start=removelowerelems,
					#Prevent a -0 access
					stop=(-removeupperelems or None),
					step=None,
					axis=histaxis,
			)
			
	#For later dype for a scale or standard deviaiton. We will now reduce
	#over statistics axis, so even when calculating with bool, the values
	#become float.
	scaledtype = "float"
	
	#We need this to limit scales to a valid range if they are given
	if (cliplimitstddev is not None) or (cliplimitfixed is not None):
		oldhistlen, _ = getHistLenFromMaxValues(
				target=target,
				maxhistvalue=maxhistvalue,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				histaxis=histaxis,
		)
		
		#Raise exception if we had no way of finding oldhistlen.
		if oldhistlen is None:
			raise ValueError(
					f"Tried finding oldhistlen to limit cliplimitstddev "
					f"{cliplimitstddev} and cliplimitfixed {cliplimitfixed} to a valid "
					f"range. But no oldhistlen was found from maxhistvalue"
					f"{maxhistvalue}. Is it None?",
					cliplimitstddev,
					cliplimitfixed,
					maxhistvalue,
			)
	
	#Before emulating quantization, derive a factor to clip to multiple of
	#standard deviation now. This derives cliplimitfixed, which can also
	#be given directly as argument to apply a known scale
	if cliplimitstddev is not None:
		
		stddev = getHistStddev(
				target=target,
				maxhistvalue=maxhistvalue,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				histaxis=histaxis,
				stataxis=stataxis,
				stddevdtype=scaledtype,
			)

		#Check the new histogram limit the user requested in terms of
		#multiple stddev
		#This is s single number being the same for all histograms
		#possibly existing to scale all histograms which later are maybe
		#merged by same factor.
		newlimit = np.multiply(cliplimitstddev, stddev, dtype=scaledtype)
		
		#If the user would scale the histogram up, we prevent that.
		#In practice, one would not shrinkt the histogram and pad
		#with zeros.
		#We also force to use at least the lowest bin.
		#In uint computation, we do not have a histlen.
		#Use oldhistlen and not maxhistvalue, becaue we need the same
		#value for all scaled elements, not a different scale for each
		#chunk possibly.
		np.clip(newlimit, a_min=1, a_max=oldhistlen, out=newlimit)
		
		#Compute factor by which values are scaled up. Everything refers
		#to magnitudes so far.
		#Again use oldhistlen to get a single scale for all values.
		cliplimitfixed = np.divide(oldhistlen, newlimit, dtype=scaledtype,)

	#If we do not derive cliplimitfixed, still check if it already given and
	#then limit to values scaling up and not leaving less than a single bin.
	#If given as argument, this is a python arg
	elif cliplimitfixed is not None:
		cliplimitfixed = np.array(cliplimitfixed, dtype=scaledtype, copy=True)
		cliplimitfixed = np.reshape(cliplimitfixed, tuple())
		np.clip(cliplimitfixed, a_min=1, a_max=oldhistlen, out=cliplimitfixed)
		
			
	#Actually apply scale and mergevalues now. This also returns the
	#effectively applied cliplimit and mergevalues. These have normalized
	#dtype and mergevalues is definately positive and includes that
	#cliplimitfixed can be so strong that no more quantization is needed.
	if (mergevalues is not None) or (cliplimitfixed is not None):
		target, maxhistvalue, cliplimitfixed, mergevalues = quantizeClipScaleValues(
				toprocess=target,
				maxhistvalue=maxhistvalue,
				mergevalues=mergevalues,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				cliplimitfixed=cliplimitfixed,
				valuescale=None,
				histaxis=histaxis,
				stataxis=stataxis,
				scaledtype=scaledtype,
		)
			
			
	#Now estimate the effort on generating the outputs. If the method
	#is given we use the model. We refer to the number of steps the
	#generated output has and do not care about positive/negative.
	mergeeffort = None
	if mergeeffortmodel is not None:
		#First get a number of Levels our output supports. Get it from
		#maxhistvalue, as that one regards that for residual chunks, an
		#ADC is allowed to support less levels.
		#maxhistvalue has a dummy stat and hist axis, which is exactly
		#what we need.
		histlen = maxhistvalue
		bincount = histlenToBincount(histlen=histlen)
		#Use the merge model, which will compute for each bincount how
		#much power the ADC needs and then sums the ADC runs up.
		mergeeffort = float(mergeeffortmodel(bincount).item())
		
	#Now possibly remove overchunkax, which were added as dummies.
	#Keep maxhistvalue synchronous
	target = np.squeeze(target, axis=tuple(squeezedims))
	if maxhistvalue is not None:
		maxhistvalue = np.squeeze(maxhistvalue, axis=tuple(squeezedims))
	
	return target, mergeeffort, maxhistvalue, cliplimitfixed, mergevalues


def equalizeQuantizedUnquantized(
			retunquantized,
			retquantized,
			runidx,
			histaxis,
			stataxis,
			dostatistic,
			dostatisticdummy,
	):
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	unquantizedresult = retunquantized["results"][runidx]
	quantizedresult = retquantized["results"][runidx]
	unquantizedmaxhistvalue = retunquantized["maxhistvalues"][runidx]
	quantizedmaxhistvalue = retquantized["maxhistvalues"][runidx]
	unquantizedmergevaluess = retunquantized["mergevaluess"]
	quantizedmergevaluess = retquantized["mergevaluess"]
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=quantizedresult.ndim)
	stataxis, = normalizeAxes(axes=stataxis, referencendim=quantizedresult.ndim)
	
	#We do tolerate inequal result lengths, where one probably selects
	#simply the last elem in both, but we need to have same number of
	#mergevalues in both.
	if len(unquantizedmergevaluess) != len(quantizedmergevaluess):
		raise IndexError(
				f"There are len(unquantizedmergevaluess) mergevaluess "
				f"in unquantized reuslt and {len(quantizedmergevaluess)} "
				f"in quantized. But the same count needs to be provided "
				f"to find how much was introduced in quantized version.",
				unquantizedmergevaluess,
				quantizedmergevaluess,
		)
		
	if histaxis == stataxis:
		raise IndexError(
				f"histaxis {histaxis} and stataxis {stataxis} cannot be the "
				f"same.",
				histaxis,
				stataxis,
		)
	
	processed = quantizedresult
	
	#Find which mergevalues were added on top when adding quantization.
	#Do so by walking along all runs and by finding the difference.
	introducedmergevalues = None
	for unquantizedmergevalues, quantizedmergevalues in zip(unquantizedmergevaluess, quantizedmergevaluess):
		if unquantizedmergevalues is None:
			thisintroducedmergevalues =  quantizedmergevalues
		elif quantizedmergevalues is None:
			raise ValueError(
					f"Unquantized mergevalues were {unquantizedmergevalues}, "
					f"but quantized mergevalues are None. So the quantized "
					f"experiment removes quantization.",
					unquantizedmergevalues,
			)
		else:
			#This check is skipped, because adding quantization in an earlier stage
			#can easily trigger this exception unreasoned
			#if quantizedmergevalues < unquantizedmergevalues:
			#	raise ValueError(
			#			f"Unquantized experiment merged with mergevalues "
			#			f"{unquantizedmergevalues}, quantized with "
			#			f"{quantizedmergevalues}. So the unquantized experiment "
			#			f"has stronger quantization.",
			#			unquantizedmergevalues,
			#			quantizedmergevalues,
			#	)
			thisintroducedmergevalues = quantizedmergevalues / unquantizedmergevalues
			
		#Now apply what was added in this result on total result.
		if introducedmergevalues is None:
			introducedmergevalues = thisintroducedmergevalues
		elif thisintroducedmergevalues is None:
			introducedmergevalues = introducedmergevalues
		else:
			introducedmergevalues *= thisintroducedmergevalues
			
	#Now check: if we take the maximum histvalue of the quantized and scale
	#it up and then round it and get a value larger than the unquantiezd
	#one, we limit the upscaling. Otherwise the value which has been
	#scaled up to make quant/unquant comparable even after quantization is
	#longer than the unquantized one.
	if (introducedmergevalues is not None):
		#The scale is a single float digit. So we need to reduce maxhistvalues
		#also to single digits.
		unquantizedmaxhistvalue = np.max(
				unquantizedmaxhistvalue,
				axis=None,
				keepdims=False,
		)
		quantizedmaxhistvalue = np.max(
				quantizedmaxhistvalue,
				axis=None,
				keepdims=False,
		)
		quantizedmaxhistvalafterscale = np.multiply(
				quantizedmaxhistvalue,
				introducedmergevalues,
				dtype="float",
		)
		quantizedmaxhistvalafterscale = np.round(
				quantizedmaxhistvalafterscale,
		)
		if np.any(quantizedmaxhistvalafterscale > unquantizedmaxhistvalue):
			introducedmergevalues = np.divide(
					unquantizedmaxhistvalue,
					quantizedmaxhistvalue,
					dtype="float"
			)
			introducedmergevalues = np.squeeze(introducedmergevalues).item()
			introducedmergevalues = float(introducedmergevalues)
		
	#Mergevalues scale the entire value range down. We now scale that back
	#up and in return get a sparse histogram.
	processed, _, _, _ = quantizeClipScaleValues(
			toprocess=processed,
			maxhistvalue=None,
			mergevalues=None,
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
			cliplimitfixed=None,
			valuescale=introducedmergevalues,
			histaxis=histaxis,
			stataxis=stataxis,
			scaledtype="float",
	)
	
	#But added cliplimitfixed could also cause different value scales
	#But that only clips the bins to a subset and merges some outer
	#bins into min/max bins. That is reverted by padding.
	#The unquantized value is passed as shape, so it is not altered.
	#If we have a length 1 hist axis, we skip this padding and keep it.
	if dostatistic and (not dostatisticdummy):
		excludeaxes = (stataxis, histaxis,)
		padsymmetricaxes = None
	else:
		excludeaxes = stataxis
		padsymmetricaxes = histaxis
	_, processed = padToEqualShape(
			a=unquantizedresult.shape,
			b=processed,
			excludeaxes=excludeaxes,
			padsymmetricaxes=padsymmetricaxes,
	)
	
	if unquantizedresult.shape != processed.shape:
		raise RuntimeError(
				f"Tried equalizing histograms, but the shapes in the end "
				f"are {unquantizedresult.shape} and {processed.shape}, "
				f"so that failed.",
				unquantizedresult.shape,
				processed.shape,
		)
	
	#return the two histograms which are ready to be plot in one
	return unquantizedresult, processed

def applyCliplimitStddevAsFixedFrom(groups, fromreturn, onlyatindices):
	
	cliplimitfixeds = tuple((i for i in fromreturn["cliplimitfixeds"]))
	
	if len(groups) != len(cliplimitfixeds):
		raise IndexError(
				f"Got {len(groups)} groups to write cliplimitfixeds to, but "
				f"{len(cliplimitfixeds)} cliplimitfixeds were given.",
				len(groups),
				len(cliplimitfixeds),
		)
		
	#Onlyidx can be a list of indices where to apply this. Or a single idx
	#can be given. Only these indices are then processed.
	if not isinstance(onlyatindices, (tuple, list, type(None))):
		onlyatindices = (onlyatindices,)
	
	#COpy groups and where cliplimitstddev is set, unset and use cliplimitfixed
	#from stochastic experiment. Because the fullscale will not be able
	#to deduct a correct stddev
	newgroups = copy.deepcopy(groups)
	for groupidx, groupcliplimitfixed in enumerate(zip(newgroups, cliplimitfixeds)):
		if (onlyatindices is not None) and (groupidx not in onlyatindices):
			continue
		group, cliplimitfixed = groupcliplimitfixed
		if group["cliplimitstddev"] is not None:
			group["cliplimitstddev"] = None
			group["cliplimitfixed"] = cliplimitfixed
			
	return newgroups

def computeSqnr(
			unquantized,
			quantized,
			histaxis,
			stataxis,
			bincount,
			dostatistic,
			dostatisticdummy,
			errordtype,
	):
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	if unquantized.shape != quantized.shape:
		raise ValueError(
				f"Got unquantized and quantized shapes {unquantized.shape} "
				f"and {quantized.shape}. But they have to be equal.",
				unquantized.shape,
				quantized.shape,
		)
		
	if histaxis == stataxis:
		raise IndexError(
				f"histaxis {histaxis} and stataxis {stataxis} cannot be the "
				f"same.",
				histaxis,
				stataxis,
		)
	
	histaxis, = normalizeAxes(
			axes=histaxis,
			referencendim=unquantized.ndim,
	)
	stataxis, = normalizeAxes(
			axes=stataxis,
			referencendim=unquantized.ndim,
	)
	
	histlen = bincountToHistlen(bincount=bincount)
	histvalues = getHistValues(
			bincount=bincount,
			axis=histaxis,
			ndim=unquantized.ndim,
			padlower=True,
			padupper=True,
	)
	
	#Stochastic solution walks along bins and tries to refer the error
	#power in terms of probability between bins to find probability with
	#which power in a discrete quantized bin originated from some
	#unquantized bin.
	if not dostatistic:
		
		#Compute probability error. This has too much probability in the discrete,
		#quantizeroutputs and we spread these into negative probabilities
		#which we should have reached.
		#Depending on where we spread that into, a different quantizererror is
		#reached.
		error = np.subtract(quantized, unquantized, dtype=errordtype)
		
		#We will alter this error, so we use a copy to return as error in
		#probability.
		errorprob = error.copy()
		
		#Treat positive and negative side symmetrically. Both access the 0 val
		posaccessor = getValueAlongAxis(
				value=None,
				start=histlen,
				stop=None,
				step=None,
				axis=histaxis,
		)
		negaccessor = getValueAlongAxis(
				value=None,
				start=None,
				stop=histlen+1,
				step=None,
				axis=histaxis,
		)
		#Also access exactly the 0 value
		zeroaccessor = getValueAlongAxis(
				value=None,
				start=histlen,
				stop=histlen+1,
				step=None,
				axis=histaxis,
		)
		#Halven the error at 0, because it is regarded twice
		np.divide(error[zeroaccessor], 2, out=error[zeroaccessor])
		
		#This is the computed error power.
		#We gather errors with linear error magnitude for plotting and with
		#squared for SNR computation
		errorpowerlinear = np.zeros(shape=error.shape, dtype=errordtype)
		errorpowersquared = np.zeros(shape=error.shape, dtype=errordtype)
		
		#Remember accessors we need to iterate from zero and whether some
		#arrays need to be flipped to always treat the 0 bin first.
		#This ensures symmetry.
		cases = (
				(posaccessor, False,),
				(negaccessor, True,),
		)
		
		for posnegaccessor, doflip in cases:	
			posnegerror = error[posnegaccessor]
			posneghistvalues = histvalues[posnegaccessor]
			#Flip the valus we read if needed. If flipping is made, the index
			#zero reads a last value, which is the 0 bin in negative half.
			if doflip:
				posnegerror = np.flip(posnegerror, axis=histaxis)
				posneghistvalues = np.flip(posneghistvalues, axis=histaxis)
			#Iterate over all values including 0
			iterlen = histlen+1
			#Remember non-eaten up probabilities from previously read bins
			shape = list(posnegerror.shape)
			shape[histaxis] = iterlen
			remainingpower = np.zeros(shape=shape, dtype=errordtype)
			#It does not make sense to always walk over all bins to collect
			#their power. The first ones maybe already have been read.
			ignorefirstbins = 0
			
			for histidx in range(iterlen):
				thishistaccessor = getValueAlongAxis(
						value=None,
						start=histidx,
						stop=histidx+1,
						step=None,
						axis=histaxis,
				)
				untilbeforethishistaccessor = getValueAlongAxis(
						value=None,
						start=ignorefirstbins,
						stop=histidx,
						step=None,
						axis=histaxis,
				)
				thisbinpower = posnegerror[thishistaccessor]
				thishistvalue = posneghistvalues[thishistaccessor]
				thisremainingpower = remainingpower[thishistaccessor]
				
				#In first iteration, take the all the error power of first bin
				#to be read by other bins. THese other bins will then add error.
				if histidx == 0:
					np.copyto(thisremainingpower, thisbinpower)
					continue
				
				#Now check how much power of previous bins we could use now to
				#compensate the current bin meaning searching which probability of
				#earlier bins belongs into this one.
				#First, we only regard bins until this one.
				usepower = remainingpower[untilbeforethishistaccessor]
				#Copy, as we will step by step change values
				usepower = usepower.copy()
				#We will only munch up bins, which have a different sign
				#than the current value. THe ones with equal sign are set to
				#0 to not use their power.
				#A bin being 0 is automatically a good one. The computation
				#is not made wrong thereby, but it is then later marked for
				#having been consumed and the bin is ignored in upcoming iters.
				usepowerbadsign = ((np.sign(usepower) == np.sign(thisbinpower)) & (usepower != 0))
				np.putmask(
						usepower,
						mask=usepowerbadsign,
						values=0,
				)
				#Build cumulative sum to find how much of the current bin is
				#compensated by applied all the bins step by step
				usepowercum = np.cumsum(
						usepower,
						axis=histaxis,
						dtype=usepower.dtype,
				)
				#Find the position starting from which the merged bins are
				#definately too large. Ignore appended bins, which are just
				#0 all the time and do not continue to increase cumsum.
				#If no bin makes it to be too large, set index to maximum
				toolargemask = np.absolute(usepowercum) > np.absolute(thisbinpower)
				np.logical_and(toolargemask, (usepower != 0), out=toolargemask)
				toolargeidx = np.argmax(toolargemask, axis=histaxis, keepdims=True,)
				np.copyto(
						toolargeidx,
						(histidx-1-ignorefirstbins),
						where=(~np.any(toolargemask, axis=histaxis, keepdims=True)),
				)
				#Get a reference index with broadcastable shape
				histindices = np.arange(usepowercum.shape[histaxis])
				histindices = padAxes(
						value=histindices,
						innertoaxis=histaxis,
						referencendim=usepower.ndim,
						padlower=False,
						padupper=True,
				)
				#Build mask masking bins which cumulatively add too
				#much power exlcuding a last bin which maybe adds too much
				usepoweroverflow = (histindices > toolargeidx)
				#And mask masking only the bin which is too much
				usepowerborder = (histindices == toolargeidx)
				#Now use copyto, which is fine with broadcasting mask shape,
				#to set all bins which provide too much accumulated power to 0.
				np.copyto(usepower, 0, where=usepoweroverflow)
				#Check which power we would now apply by including a last bin
				#which adds too much
				maybeoverflowpower = getValueFromBroadcastableIndex(
						value=usepowercum,
						index=toolargeidx,
						axis=histaxis,
				)
				#Really compute how much too much we would add. Remember that
				#we only took usepower, where it had an opposing sign compared
				#to this bin. So np.add is the right call.
				toomuchpower = np.add(
						maybeoverflowpower,
						thisbinpower,
						dtype=usepower.dtype,
				)
				#There is not too much power if not too much was subtracted
				nottoomuchpower = (np.absolute(maybeoverflowpower) < np.absolute(thisbinpower))
				np.putmask(
						toomuchpower,
						mask=nottoomuchpower,
						values=0,
				)
				#At the regarding bin, subtract the power which would be subtracted
				#too much. Fix only the border bin.
				np.subtract(
						usepower,
						toomuchpower,
						where=usepowerborder,
						out=usepower,
				)
				#Now re-compute the power we will actually apply: the maybe too
				#large value reduced by what we reduce
				totalusepower = np.subtract(
						maybeoverflowpower,
						toomuchpower,
						dtype=usepower.dtype,
				)
				#And how large the current error is after compensating a part
				#of it
				thisupdatedbinpower = np.add(
						thisbinpower,
						totalusepower,
						dtype=errordtype,
				)
				#Find the error which would be applied when taking power from
				#a different bin into this bin meaning a value was at a
				#wrong position. Take only values from bins, where usepowers
				#really could be set from previous runs.
				posdependenterrorlinear = np.subtract(
						posneghistvalues[untilbeforethishistaccessor],
						thishistvalue,
						dtype=posneghistvalues.dtype,
				)
				#Also get squared version for computing powers
				posdependenterrorsquared = np.square(
						posdependenterrorlinear,
						dtype=posdependenterrorlinear.dtype)
				#And find sign of the error in current bin. If it is positive,
				#we have one of the big needles at a discrete quantization level
				#and we write the transferred error power to where we
				#read it from.
				writeerrortoprevious = (thisbinpower > 0)
				writeerrortothis = (~writeerrortoprevious)
				#Compute error made here as powers depending on position and
				#with weighting by probability.
				thiserrorlinear = np.multiply(
						posdependenterrorlinear,
						np.absolute(usepower),
						dtype=errordtype,
				)
				#For all the error we have to write into this bin,
				#sum over errors made from different bins
				thiserrorlinearthis = np.sum(
						thiserrorlinear,
						axis=histaxis,
						dtype=errordtype,
						keepdims=True,
						where=writeerrortothis,
				)
				#Same for squared error
				thiserrorsquared = np.multiply(
						posdependenterrorsquared,
						np.absolute(usepower),
						dtype=errordtype,
				)
				thiserrorsquaredthis = np.sum(
						thiserrorsquared,
						axis=histaxis,
						dtype=errordtype,
						keepdims=True,
						where=writeerrortothis,
				)
				#Add to global error lists. All accessors used here return only
				#views, so the original, full arrays will be updated.
				#THe linear error still has a sign and we maybe have to flip
				#that.
				#We provide the power to write into the single current bin and
				#the rest to write into previous bins. Only one case applies for
				#each histogram.
				writetocases = (
						(thiserrorlinearthis, thiserrorlinear, errorpowerlinear, True),
						(thiserrorsquaredthis, thiserrorsquared, errorpowersquared, False),
				)
				for srcthis, src, dst, allowsignflip in writetocases:
					#Access only one half (pos or neg) from the target
					dst = dst[posnegaccessor]
					#In negative half, we flipped our inputs before starting
					#to iterate over hist and now have to do the same for the
					#target.
					if doflip:
						dst = np.flip(dst, axis=histaxis)
					#If we flip, we also flip the sign, because the
					#histvalues also have been read in reversed order. But we
					#also flip all results, as otherwise the linear error magnitude
					#is negative. So if doflip and allowsignflip, we would
					#flip twice and keep the sign.
					if ((not doflip) and allowsignflip):
						np.negative(srcthis, out=srcthis)
						np.negative(src, out=src)
					#No need to use add operation, as we are the first iteration
					#to tuch this bin.
					dst[thishistaccessor] = srcthis
					#For adding power to previous bins, we have to really add
					#on top.
					np.add(
							dst[untilbeforethishistaccessor],
							src,
							where=writeerrortoprevious,
							out=dst[untilbeforethishistaccessor],
					)
				
				#We now took power from remainingpower to put it into our
				#current bin. Regard in reaminingpower that its components have
				#been in use
				np.subtract(
						remainingpower[untilbeforethishistaccessor],
						usepower,
						out=remainingpower[untilbeforethishistaccessor],
				)
				#And add uncompensated power to remainingpower
				np.copyto(thisremainingpower, thisupdatedbinpower)
				
				#Now check which bins have been consumed totally in all histograms,
				#such that we can later ignore these bins in all computations
				#to make it faster.
				#Compute how many bins of usepower have been consumed totally
				totalconsumedbins = np.min(toolargeidx, axis=None, keepdims=False)
				totalconsumedbins = totalconsumedbins.item()
				#But we ignored bins with wrong sign for compensating
				#current value bin.
				totalwrongsignbins = np.argmax(usepowerbadsign, axis=histaxis, keepdims=True)
				totalwrongsignbins = np.min(totalwrongsignbins, axis=None, keepdims=False)
				totalwrongsignbins = totalwrongsignbins.item()
				#The index we have for the consumed bin is the last one which
				#was consumed. But this last one was maybe partially consumed.
				#So if the index is 0, we can throw away 0 bins, because
				#we maybe later need to work with the partially consumed one.
				#Use that rule to update the number of lower bins which will
				#be ignored for all histograms now. Do not ignore bins
				#which have not been read now due to wrong sign, they will
				#be read later.
				ignorefirstbins += min(totalconsumedbins, totalwrongsignbins)
				
				
		#Summarize sqaured erros to a single number. Sum squares, as that
		#is how proper error computation works when having errors weighted with
		#probability.
		errorpowertotal = np.sum(
				errorpowersquared,
				axis=histaxis,
				keepdims=True,
				dtype=errordtype,
		)
				
		#Also compute signalpower
		histpowers = np.square(histvalues, dtype=histvalues.dtype)
		signalpower = np.multiply(
				histpowers,
				unquantized,
				dtype=errordtype,
		)
		signalpowertotal = np.sum(
				signalpower,
				axis=histaxis,
				keepdims=True,
				dtype=errordtype,
		)
		
	#In statistic, we have simple numbers and not proabilities
	else:
		#We always need an integer version for error computation and
		#some booleans for probabilistic computations 
		if dostatisticdummy:
			unquantizedint = packHist(
					topack=unquantized,
					axis=histaxis,
					keepdims=True,
					strict=True,
			)
			quantizedint = packHist(
					topack=quantized,
					axis=histaxis,
					keepdims=True,
					strict=True,
			)
			unquantizedbool = unquantized
			quantizedbool = quantized
		else:
			unquantizedbool = unpackHist(
					tounpack=unquantized,
					bincount=bincount,
					axis=histaxis,
			)
			quantizedbool = unpackHist(
					tounpack=quantized,
					bincount=bincount,
					axis=histaxis,
			)
			unquantizedint = unquantized
			quantizedint = quantized
		
		#Turn the booleans into actual probabilities
		unquantizedprob = packStatistic(
				topack=unquantizedbool,
				axis=stataxis,
				keepdims=True,
		)
		
		quantizedprob = packStatistic(
				topack=quantizedbool,
				axis=stataxis,
				keepdims=True,
		)
		
		#The error is the difference between the numbers
		errorint = np.subtract(quantizedint, unquantizedint, dtype=unquantizedint.dtype)
		
		#Also return an error in value-occurence probability.
		errorprob = np.subtract(quantizedprob, unquantizedprob, dtype=errordtype)
		
		#We also want the error power contributed by each bin.
		#This is the linear error contributed by each statistic computation
		errorabs = np.absolute(errorint, dtype=errorint.dtype)
		#We have to sort them into a histogram, where each hist value is
		#related to an unquantized computation result and we have the
		#linear error contributed by that computation. We so far
		#keep the statistic axis.
		#This is the output with correct shape
		errorhistlinear = np.zeros(shape=unquantizedbool.shape, dtype=errorabs.dtype)
		#Copy the error we already know to hist locations where the boolean
		#unquantized result has a hot digit
		np.copyto(errorhistlinear, errorabs, where=unquantizedbool)
		#We so far did not combine error magnitude with probabilities. So
		#we still can square to have a squared error
		errorhistsquared = np.square(errorhistlinear, dtype=errorhistlinear.dtype)
		#Now average over statistics axis which is the same like multiplying
		#error magnitude and its probbility. Do not use packStatistic, becuase
		#we here have no bool dtype, because we have for each computation at its
		#desired result value the error it made. And that error is not bool.
		errorpowerlinear = np.average(
				errorhistlinear,
				axis=stataxis,
				keepdims=True,
				#dtype="float",
		)
		errorpowersquared = np.average(
				errorhistsquared,
				axis=stataxis,
				keepdims=True,
				#dtype="float",
		)
		
		#Error and signal power for SNR could be computed by using
		#already computed value probabilities and squared errorpower per
		#bin. But we instead use simply the int numbers. Because that is
		#the way where I know that it is definately correct.
		#Errorpower is the squared error
		errorpower = np.square(errorint, dtype=errorint.dtype)
		#Also derive signalpower from unquantized value
		signalpower = np.square(unquantizedint, dtype=unquantizedint.dtype)
		#Sum values up and do so with error dtype, as we maybe need float
		#to generate these huge numbers
		errorpowertotal = np.sum(
				errorpower,
				axis=stataxis,
				keepdims=True,
				dtype=errordtype,
		)
		signalpowertotal = np.sum(
				signalpower,
				axis=stataxis,
				keepdims=True,
				dtype=errordtype,
		)
	
	baderrorpower = (errorpowertotal == 0)
	np.putmask(errorpowertotal, mask=baderrorpower, values=1)
	snrlinear = np.divide(signalpowertotal, errorpowertotal, dtype=errordtype)
	snrlog = np.log10(snrlinear, dtype=errordtype)
	np.multiply(snrlog, 10, out=snrlog)
	np.putmask(snrlog, mask=baderrorpower, values=np.inf)
			
	return snrlog, errorprob, errorpowerlinear, errorpowersquared
	
	
def generateSimulationOperands(
		statisticdim,
		dostatisticdummy,
		activationlevels,
		weightlevels,
		nummacs,
		randombehave,
		randomclips,
):
	
	#Length of statistics dimension, which we always keep
	dostatistic = statisticdim is not None
	if not dostatistic:
		statisticdim = 1
		
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	if len(randomclips) != 2:
		raise IndexError(
				f"randomclips is {randomclips}, but there must "
				f"be 2 elements",
				randomclips,
		)

	#Random generator with reliable seed
	randomgen = np.random.default_rng(seed=12345)
	
	#Generate weights and activations and turn them into their histograms.
	activationsweights = list()
	#The original numbers beore splitting into bits
	activationsweightsint = list()
	#The used bincounts
	bincounts = list()
	for operandlevels, randomclip in zip((activationlevels, weightlevels), randomclips):
		randomclip = float(randomclip)
		#This is the defined random behavior. The user can pass a class or use
		#a pre-define one.
		#We only draw positive numbers.
		RANDOM_GEN_CLASSES = dict(
				#All values with same probability
				uniform=scipy.stats.uniform,
				#Normal distribution, where large vales are clipped to the
				#exact maximum
				norm=scipy.stats.norm,
				#Normal distribution, which is truncated and which does not
				#clip values.
				truncnorm=scipy.stats.truncnorm,
				#A uniform distribution yielding just full scale values
				fullscale=scipy.stats.uniform,
		)
		RANDOM_GEN_ARGS = dict(
				uniform=dict(
						#loc is the lower bound
						#The uniform distribution uses the randomclip as
						#max/min value. There are no sigmas and the scale is
						#not normalized to any sigma.
						#loc is left edge
						loc=-randomclip,
						#scale is distance to right edge
						scale=2.*randomclip,
				),
				norm=dict(
						loc=0.,
						#We create standard norma values and the clipping value
						#is then in multiple of standard deviations by
						#setting sigma to 1.
						#This makes it easy to apply e.g.
						#optimum clipping criterion, as it is a number in
						#multiple of standard deviation.
						scale=1.,
				),
				truncnorm=dict(
						loc=0.,
						scale=1.,
						a=-randomclip,
						#Here, we truncate even when generating the numbers.
						#Values larger than this are not dawn.
						b=randomclip,
				),
				fullscale=dict(
						loc=randomclip,
						#Scale zero causes NaN cdf
						scale=0.,
				),
		)
		#Custom definitions for cds, as fullscale does not have one
		EXPLICIT_CDFS = dict(
				uniform=None,
				norm=None,
				truncnorm=None,
				#For fullscale, cdf jumps from 0 to 1 when crossing the fullscale
				#value.
				fullscale=lambda a: (a >= randomclip).astype(dtype=a.dtype)
		)
		
		#Instantiate a default generator, if the given generator is a valid
		#generator key.
		if randombehave in set(RANDOM_GEN_CLASSES.keys()):
			explicitcdf = EXPLICIT_CDFS[randombehave]
			randombehavelookup = RANDOM_GEN_CLASSES[randombehave](**RANDOM_GEN_ARGS[randombehave])
		else:
			explicitcdf = None
			randombehavelookup = randombehave
		
		#The number of histogram entries we will get. Histlen regard
		#amplitudes only and this is what we get from user.
		histlen = int(round(operandlevels))
		#And the number of bins in the full histogram.
		bincount = histlenToBincount(histlen=histlen)
		
		#Remember histlen for later use
		bincounts.append(bincount)
		
		#If we do statistics, draw random numbers
		if dostatistic:
			#Raw gaussian numbers. Already include a dimension for bits.
			operand = randombehavelookup.rvs(
					size=(statisticdim, nummacs, 1),
					random_state=randomgen,
			)
			#Clip random number at fullscale
			np.clip(
					operand,
					a_min=-randomclip,
					a_max=randomclip,
					out=operand,
			)
		
		#Or draw probabilities for values
		else:
			#Values between -clip and +clip value. These are the rounded values,
			#which sit at the center of pdf bins.
			operand = np.linspace(
					start=-randomclip,
					stop=randomclip,
					num=bincount,
			)
			#Add dimensions for hist and dummy statistics.
			#We let the operator rise along mac axis.
			operand = np.expand_dims(operand, axis=(0, -1))
			
			#These are the bins we now integrate to get probability of the
			#integer values computed above.
			#In these bins, we integrate the pdf via cdf.
			#The values within a bin round() to the respective value we
			#later turn into bins.
			#We have to ragard that max/min values
			#would be included, but the pdf is clipped and never draws
			#them. So these bins are in there only half.
			
			#We lose half teh first and half the last bin, so one bin in total.
			#But the given range extends along positive and negative.
			binwidth = float(2 * randomclip) / float(bincount - 1)
			#Draw the lower and upper bounds of bins, first and last bin bounds
			#reach over allowed range.
			lowerbinbounds = np.linspace(
					start=-randomclip-(binwidth/2.),
					stop=randomclip-(binwidth/2.),
					num=bincount,
					dtype="float",
			)
			upperbinbounds = np.linspace(
					start=-randomclip+(binwidth/2.),
					stop=randomclip+(binwidth/2.),
					num=bincount,
					dtype="float",
			)
			#Combine bin bounds
			binbounds = np.stack((lowerbinbounds, upperbinbounds), axis=0)
			#Clip disallowed values
			np.clip(
					binbounds,
					a_min=-randomclip,
					a_max=randomclip,
					out=binbounds,
			)
			#Use cdf to integrate pdf in bounds. Possibly use explicit cdf.
			if explicitcdf is None:
				thiscdf = randombehavelookup.cdf
				
			else:
				thiscdf = explicitcdf
			probabilities = thiscdf(binbounds)
			probabilities = probabilities[1] - probabilities[0]
			#Add power below lowest bin bound and above highest to border bins.
			#This mimics clipping.
			probabilities[0] += thiscdf(binbounds[0, 0])
			probabilities[-1] += (1. - thiscdf(binbounds[1, -1]))
			
			#Add dimension for mac and statistics. So probabilities
			#are swept along hist axis, which is where we need them later.
			probabilities = np.expand_dims(probabilities, axis=(0, 1))
		
		#We have numbers with a value up to randomclip. Turn that into
		#integers up to histlen (out targeted magnitude).
		np.multiply(
				operand,
				float(histlen) / float(randomclip),
				out=operand,
		)
		operand.round(out=operand)
		#We should have reached integer values now.
		operand = operand.astype("int")
		#Remember integer operand
		activationsweightsint.append(operand)
		
		#We now have in stochastic only a vector of rising values in hist
		#format and their probability of appearance. But the probabilitiy is
		#actually the value we need.
		if not dostatistic:
			
			#The histogram probabilities are the actual operand we need.
			#Right now, operand only holds the value represetned by the
			#hist value.
			operand = probabilities
			#Use the mac dim, which is only present in stochastic and
			#which has been summed to 1, now as dim along macs.
			#Tiling along the nummacs dim does not make sense, as that would
			#only repeat probabilities we keep the same. We still have
			#to do it, as later operations might chunk this axis and then
			#things become different.
			operand = np.tile(operand, reps=(1, nummacs, 1))
			
		#In statistics, we need to turn random uint values into hists.
		else:
			operand = unpackHist(
					tounpack=operand,
					bincount=bincount,
					axis=-1,
			)
			#The unpack function does that already.
			#operand = operand.astype(bool)
			
			#If we go for the stochastic implementation, we have to make the bool
			#axis a int axis, because reduceSum will always keep the histogram
			#axis a dummy and will use int dtype instead.
			if not dostatisticdummy:
				operand = operand.astype(dtype="int")
			
			#In dostatisticdummy, we keep the bool dtype.
			
		#We currently have shape (statistics, nummacs, hist).
		#If we're doing statistic, the statistics dim is >1 and we have
		#a bool dtype.
		#Otherwise, the statistics dim is =1, but we have some float dtype.
			
		#Remember the found operand.
		activationsweights.append(operand)
		
	activations, weights = activationsweights
	activationsint, weightsint = activationsweightsint
	activationbincount, weightbincount = bincounts
	
	#Now do the multiplication. We only have positive numbers, so we can
	#do an AND to get that, or we multiply probabilities. But we have to
	#do all combinations of activation and weight hist positions.
	#STart by creating an index matrix. Activation index
	activationhistidx = np.arange(activationbincount, dtype="uint")
	activationhistidx = np.reshape(
			activationhistidx,
			(activationbincount, 1),
	)
	#Weight index
	weighthistidx = np.arange(weightbincount, dtype="uint")
	weighthistidx = np.reshape(
			weighthistidx,
			(1, weightbincount),
	)
	activationhistidx = np.tile(activationhistidx, reps=(1, weightbincount))
	weighthistidx = np.tile(weighthistidx, reps=(activationbincount, 1))
	#Use the index matrices to lookup activation and weight bins.
	#Keep the statistics and nummacs dim, but replace the hist position
	#by looked-up bins. We get two matrices with shape
	#(statistics, nummacs, activationbincount, weightbincount)
	#And from that we can lookup all bit combinations for the MAC op
	activationformult = activations[:, :, activationhistidx]
	weightformult = weights[:, :, weighthistidx]
	
	#Do the multiply operations. AND the bits or multiply probabilities.
	#AND'ing even works for default statistics with uint numbers, as these
	#uint numbers are currently 0 or 1.
	if dostatistic:
		multiplied = np.bitwise_and(activationformult, weightformult)
	else:
		multiplied = np.multiply(activationformult, weightformult)
		
	#Now set a first maxhistvalue. This value does not have a statistics dim, as
	#it represetns a full scale operation. The fullscale is all booleans
	#being 1. All full scale numbers are positive, as they describe full
	#scale magnitude.
	#Full scale is always an int number. Even if we compute with one-hot
	#hists or their probabilities.
	maxhistvalueshape = list(multiplied.shape)
	maxhistvalueshape[0] = 1
	maxhistvalue = np.ones(
			shape=maxhistvalueshape,
			dtype="int",
	)
		
	return dict(
			activations=activations,
			activationsint=activationsint,
			weights=weights,
			weightsint=weightsint,
			activationhistidx=activationhistidx,
			weighthistidx=weighthistidx,
			activationformult=activationformult,
			weightformult=weightformult,
			multiplied=multiplied,
			firstmaxhistvalue=maxhistvalue,
	)

def simulateMvm(
		statisticdim,
		dostatisticdummy,
		selfcheckindummy,
		activationlevels,
		weightlevels,
		nummacs,
		randombehave,
		randomclips,
		groups,
):
	
	#We do statistic simulation, if dimension for their results is given
	dostatistic = statisticdim is not None
	
	checkStatisticsArgs(
			dostatistic=dostatistic,
			dostatisticdummy=dostatisticdummy,
	)
	
	operands = generateSimulationOperands(
		statisticdim=statisticdim,
		dostatisticdummy=dostatisticdummy,
		activationlevels=activationlevels,
		weightlevels=weightlevels,
		nummacs=nummacs,
		randombehave=randombehave,
		randomclips=randomclips,
	)
	
	results = list()
	mergeefforts = list()
	maxhistvalues = list()
	cliplimitfixeds = list()
	mergevaluess = list()
	
	runningresult = operands["multiplied"]
	
	#Suggestion for first maxhistvalue is among operands
	maxhistvalue = operands["firstmaxhistvalue"]
	
	#Debug only a single statistic test, where something goes wrong
	runsinglestat = None
	#runsinglestat = 23
	if (runsinglestat is not None) and dostatistic:
		if runningresult.shape[0] > runsinglestat:
			runningresult = runningresult[runsinglestat:runsinglestat+1]
	
	for group in groups:
		runningresult, mergeeffort, maxhistvalue, cliplimitfixed, mergevalues = reduceSum(
				tosum=runningresult,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				selfcheckindummy=selfcheckindummy,
				#The stataxis needs to be synchronous to generated
				#operands, but is not parameterizable there.
				stataxis=0,
				maxhistvalue=maxhistvalue,
				**group,
		)
		results.append(runningresult)
		mergeefforts.append(mergeeffort)
		maxhistvalues.append(maxhistvalue)
		cliplimitfixeds.append(cliplimitfixed)
		mergevaluess.append(mergevalues)
		
	#Return all results
	return dict(
			**operands,
			results=results,
			mergeefforts=mergeefforts,
			maxhistvalues=maxhistvalues,
			cliplimitfixeds=cliplimitfixeds,
			mergevaluess=mergevaluess,
	)

def optimumClippingCriterion(levels, abstol=1e-6, maxiter=100):
	"""Compute optimum clipping criterion.
	
	See `clipping` for why one maybe needs clipping in extend to quantization.
	
	This evaluates (17) from [OCC]_ to compute the optimum cliplimit. The found
	value needs to be multiplied by the standard-deviation of the value which
	is to be clipped. It also is dependent on the number of levels of the
	quantizer following the clipping.
	
	The function needs multiple iterations to converge to a final value.
	
	Parameters
	----------
	levels : `int`
		The number of levels in the quantizer following clipping. [OCC]_ uses
		a bitwidth *B*.
		
	abstol : `float`, optional
		The absolute tolerance t be achieved in the iterative computation.
		The default is 1e-6.
	maxiter : `int`, optional
		Maximum number of iterations after which computation is interrupted.
		The default is 100.

	Raises
	------
	`RuntimeError`
		If convergence within *abstol* and *maxiter* was not achieved.

	Returns
	-------
	result : `float`
		The *cliplimitstddev* (see `clipping`).

	"""

	#Value to start with. Clipping at two standard deviations is not bad.
	lastresult = 2.
	bitwidth = math.log2(float(levels))
	#Scipy calls the cumulative cdf the survival function
	ccdfnumpy = scipy.stats.norm(loc=0., scale=1.).sf
	ccdf = lambda x: ccdfnumpy(x).item()
	
	for iteridx in range(maxiter):
		
		numerator = math.sqrt(2. / math.pi) * math.exp(-(lastresult**2.) / (2.))
		ccdfval = ccdf(x=lastresult)
		denominator = ((4.**(-bitwidth)) / 3.) + (2. * ccdfval)
		result = numerator / denominator
		if abs(result - lastresult) < abstol:
			break
		lastresult = result
		
	else:
		raise RuntimeError(
				f"Computing optimum clipping criterion with levels={levels}, "
				f"abstol={abstol} and maxiter={maxiter}. No convergence."
		)
	
	return result

