import numpy as np

def normalizeAxes(axes, referencendim):
	axes = np.array(axes, dtype=int)
	axes = axes.reshape((-1,))
	#Make negative indices positive
	axes = np.where(axes < 0, referencendim + axes, axes)
	return axes

def getValueAlongAxis(value, start, stop, step, axis):
	if value is not None:
		axis, = normalizeAxes(axes=axis, referencendim=value.ndim)
	#Access single element from axis
	#by accessing range, that does not remove the now-length-1 axis
	accessor = (*((slice(None, None),)*axis), slice(start, stop, step))
	
	#Return accessor, if value is not given. Is nice for getting views on
	#arrays outside of this function.
	if value is None:
		return accessor

	else:
		return value[accessor]

def getValueFromBroadcastableIndex(value, index, axis):
	
	if isinstance(value, np.ndarray):
		valueshape = value.shape
	else:
		valueshape = value
		
	axis, = normalizeAxes(axes=axis, referencendim=len(valueshape))
	
	#Build an accessor. Imagine the index selects all values alogn an
	#axis, but hase some additional length-1 dims. If the index would be
	#used on the axis, that axis is replaced by multiple axes including the
	#length-1 ones. So we instead access all eleemnts on other axes to
	#prevent that. The many axis indices are then broadcasted in the
	#end to one having the same ndim as the value.
	accessor = list()
	for axidx in range(len(valueshape)):
		#Append real int acessor from user
		if axidx == axis:
			accessor.append(index)
		#Append broadcastable dummy accessor
		else:
			allaccessor = np.arange(valueshape[axidx])
			allaccessor = padAxes(
					value=allaccessor,
					innertoaxis=axidx,
					referencendim=len(valueshape),
					padlower=False,
					padupper=True,
			)
			accessor.append(allaccessor)
			
	accessor = tuple(accessor)
	
	if valueshape is value:
		return accessor
	else:
		return value[accessor]

def padAxes(value, innertoaxis, referencendim, padlower, padupper):
		#innertoaxis must be a single one
		innertoaxis, = normalizeAxes(
				axes=innertoaxis,
				referencendim=referencendim,
		)
		
		#Pad, such that inner dimension sits at dimension innertoaxis in
		#an array with length referencedim. Needed to make arrays
		#broadcastable.
		if padupper:
			padupperdims = np.arange(
					start=1,
					stop=(referencendim-innertoaxis),
					step=1,
					#We will need negative numbers in the following, so do not
					#use uint.
					dtype="int",
			)
			#Only pad at the end.
			np.negative(padupperdims, out=padupperdims)
		else:
			padupperdims = tuple()
			
		#Pad lower axes to e.g. make a value usable as index array
		if padlower:
			#To move the inner axis to given index, we have to pad lower
			#axes, but not the ones which already exist
			#If we have no value shape, assume that given value is 1D
			if value is None:
				existingdims = 1
			else:
				existingdims = value.ndim
			padlowercount = innertoaxis - (existingdims - 1)
			if padlowercount < 0:
				raise ValueError(
						f"Having {existingdims} dimensions and being asked "
						f"to move the upper dimension to index {innertoaxis} "
						f"by upper padding. Cannot remove dimensions by "
						f"padding.",
						existingdims,
						innertoaxis,
				)
			padlowerdims = range(padlowercount)
		else:
			padlowerdims = tuple()
		
		#Padrules must be tuples
		padrule = (*padlowerdims, *padupperdims,)
		
		#Returns a view of a value or the padrule if no value was given.
		if value is None:
			return padrule
		else:
			return np.expand_dims(value, axis=padrule)
		
def padToEqualShape(a, b, excludeaxes, padsymmetricaxes):
	
	#Extract shaped from a and b and even allow that just shapes
	#are given.
	if isinstance(a, np.ndarray):
		ashape = a.shape
	else:
		ashape = a
	if isinstance(b, np.ndarray):
		bshape = b.shape
	else:
		bshape = b
	
	#If ndim is different, there is nothing we can do
	if len(ashape) != len(bshape):
		raise ValueError(
				f"Can only pad axis length of arrays with equal axes count, "
				f"but got shapes {ashape} and {bshape}.",
				ashape,
				bshape,
		)
	
	#Return copies, if a and b are the same.
	if ashape == bshape:
		#If i is also its shape, a was given as shape, not ndarray.
		#A shape is a tuple and has no copy function.
		if a is ashape:
			acopy = (*ashape,)
		else:
			acopy = a.copy()
		if b is bshape:
			bcopy = (*bshape,)
		else:
			bcopy = b.copy()
		return acopy, bcopy
	
	if excludeaxes is not None:
		excludeaxes = normalizeAxes(
				axes=excludeaxes,
				referencendim=len(ashape),
		)
		
	if padsymmetricaxes is not None:
		padsymmetricaxes = normalizeAxes(
				axes=padsymmetricaxes,
				referencendim=len(ashape),
		)

	#First find shape difference
	singlepadrule = np.array(ashape) - np.array(bshape)
	#Set difference along ignored axes to 0. Excludeaxes is an index access.
	if excludeaxes is not None:
		singlepadrule[excludeaxes] = 0
	#Add axis for lower and upper padding
	singlepadrule = np.expand_dims(singlepadrule, axis=-1)
	#Split padrule to evenly pad bottom and top. In the end,
	#number of values along ax shall compensate what was
	#missing.
	#We first add all padding to end of axes and then split between beginning
	#and end if needed.
	fullpadrule = np.concatenate((np.zeros_like(singlepadrule), singlepadrule), axis=-1)
	if padsymmetricaxes is not None:
		#Where we need the splitting, copy full padding to beginning, then
		#halven both.
		fullpadrule[padsymmetricaxes, 0] = fullpadrule[padsymmetricaxes, 1]
		#// rounds towards -inf. But we want to round towards 0, as padding
		#a or b shall be the same.
		fullpadrule[padsymmetricaxes,] = (np.sign(fullpadrule[padsymmetricaxes,]) * \
				(np.absolute(fullpadrule[padsymmetricaxes,]) // 2))
		#Then, check sum of padded values for all, this
		#will just be easy for padding only in the end.
		#If we here padded e.g. 2 up to 3, we want to pad in the beginning.
		#Because in that case, we have aligned values 0 and 1 and padded
		#for -1.
		fullpadrulesum = np.sum(fullpadrule, axis=-1, keepdims=True)
		np.add(
				fullpadrule[:, :1],
				#Keep the sign of padrule, as it describes to pad in a or b.
				#This add operation increases the magnitude.
				np.sign(singlepadrule),
				out=fullpadrule[:, :1],
				where=(np.absolute(fullpadrulesum) < np.absolute(singlepadrule)),
		)
	#Use positive numbers to pad b
	if b is not bshape:
		b = np.pad(
				b,
				pad_width=np.absolute(np.maximum(fullpadrule, 0)),
				mode="constant",
				constant_values=0,
		)
	#And vice versa
	if a is not ashape:
		a = np.pad(
				a,
				pad_width=np.absolute(np.minimum(fullpadrule, 0)),
				mode="constant",
				constant_values=0,
		)
	
	return a, b
