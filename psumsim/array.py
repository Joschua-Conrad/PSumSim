"""Functions in conjunction with `numpy.ndarray`.

This module does not have anything to do with partial sums and histograms."""

#This module is imported virtually everywhere. Prevent circular imports by
#importing external modules only.
import numpy as np

def normalizeAxes(axes, referencendim):
	"""Make axes indices positive.

	Parameters
	----------
	axes : `int`, `iterable` of `int`, `numpy.ndarray`
		The one or many axes to normalize. If this has more than one dimension,
		the array is flattened.
		
	referencendim : `int`
		Needed to turn negative indices to positive ones. If this is 5, the
		*axes* entry with value *-1* is converted to *4*, such that it refers
		to the last axis of an array with *5* dimensions.

	Returns
	-------
	axes : 1D `numpy.ndarray` of type `int`
		The normalized axes.

	"""
	
	axes = np.array(axes, dtype=int)
	axes = axes.reshape((-1,))
	#Make negative indices positive
	axes = np.where(axes < 0, referencendim + axes, axes)
	return axes

def getValueAlongAxis(value, start, stop, step, axis):
	"""Get a subset of values along axis of `numpy.ndarray`.
	
	Parameters
	----------
	value : `numpy.ndarray`, `None`
		The array to get a subset of values from.
		
	start : `int`, `None`
		At which index the set starts. Use `None` for the first element.
	stop : `int`, `None`
		At which index the set ends. Use `None` for the last element.
	step : `int`, `None`
		The step for indexing elements. Use `None` to use step-width *1* and to
		access each element in the start-stop range.
	axis : `int`
		The axis along which *start*, *stop* and *step* indexing is made.
		All other axes are kept as-is.

	Returns
	-------
	`numpy.ndarray`, (`tuple` of `slice`)
		If *value* was an array, this is a `numpy.ndarray.view` on the specified subset
		of elements. See `views`. The dtype is kept. All dimensions except
		*axis* have the same length. Only dimension axis is as long as
		specified by *start*, *stop* and *step*.
		
		If *value* was `None`, this returns a `tuple`. This *accessor* can be
		used as in :code:`somearray[accessor]` to acces elements in some other
		array.

	"""
	
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
	"""Use an `int` array to access elements.
	
	If you have an `numpy.ndarray` with indices of elements you want from another
	`numpy.ndarray`, use this function. `numpy` does provide similar functionality
	by default, but they do not support to keep the array shape the same except
	for one dimenion made shorter by selecting elements. This function
	supports this.
	

	Parameters
	----------
	value : `numpy.ndarray`, ((`tuple`, `list`) of `int`)
		An array to index or the shape of such an array.
		
	index : `numpy.ndarray` of `int`
		The index to use. The shape must be broadcastable to *value* except at
		*axis*. The numbers are interpreted as index along *axis* from which to
		take elements.
		
	axis : `int`
		The axis index along which *index* is interpreted.

	Returns
	-------
	`numpy.ndarray`, (`tuple` of `numpy.ndarray`)
		If *value* was an array, this is a `numpy.ndarray`. BUt this is NOT
		something like in `views`. This is a copy. Because under the hood,
		everything is wrapped into *advanced indexing* and these return copies.
		
		If *value* was `None`, an accessor is returned. This *accessor* can be
		used as in :code:`somearray[accessor]` to acces elements in some other
		array.
	"""
	
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
	"""Append axes to make an `numpy.ndarray` broadcastable to another one.
	
	Parameters
	----------
	value : `numpy.ndarray`, `None`
		The array to add new axes to.
		
	innertoaxis : `int`
		The padding will move the last (inner-most) dimension of *value* to
		this new axis.
	referencendim : `int`
		Negative *innertoaxis* refer to an array with this many dimensions.
	padlower : `bool`
		If set, add lower dimensions to fulfill what *innertoaxis* wants.
		Note that this needs to know how many dimensions exist in *value*.
		If *value* is not given, a 1D array is assumed.
	padupper : `bool`
		If set, add upper dimensions to fulfill what *innertoaxis* wants.

	Raises
	------
	`ValueError`
		If the shape adjustment would not need padding, but rather axes
		squeezing.

	Returns
	-------
	`numpy.ndarray`, (`tuple` of `int`)
		The array with adjusted axes if *value* was not `None`. Otherwise, a
		`tuple` ready for being passed to `numpy.expand_dims` is returned.
		That can be used to pad an actual array.

	"""
	
	#innertoaxis must be a single one
	innertoaxis, = normalizeAxes(
			axes=innertoaxis,
			referencendim=referencendim,
	)
	
	#Pad, such that inner dimension sits at dimension innertoaxis in
	#an array with length referencedim. Needed to make arrays
	#broadcastable.
	if padupper:
		paduppercount = (referencendim - 1 - innertoaxis)
		if paduppercount < 0:
			raise ValueError(
					f"Being asked to an inner axis to index {innertoaxis} in "
					f"array with {referencendim} axes. That does not work by "
					f"padding.",
					innertoaxis,
					referencendim,
			)
		padupperdims = np.arange(
				start=1,
				stop=(paduppercount + 1),
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
	"""Pad elements to give to arrays the same shape.

	Parameters
	----------
	a : `numpy.ndarray`, ((`tuple`, `list`) of `int`)
		An array to pad or a shape to pad the other array to. If a shape
		is given, that shape is not padded.
		
	b : `numpy.ndarray`, ((`tuple`, `list`) of `int`)
		Just like *a*. For each dimension, the array with the shorter one is
		found and pad.
		
	excludeaxes : `None`, ((`tuple`, `list`) of `int`)
		A set of axes not to touch and to keep as-is.
		
	padsymmetricaxes : `None`, ((`tuple`, `list`) of `int`)
		Indices of axes which are not to be padded in the end, but rather
		equally in the beginning and end. If an odd number of elemnts is
		to be padded, one element more is padded in the beginning. Set this
		for equalizing histogram axes (see `dataformat`), where positive and
		negative value bins are padded then.

	Raises
	------
	`IndexError`
		If *a* and *b* have different number of dimensions.

	Returns
	-------
	`numpy.ndarray`, ((`tuple`, `list`) of `int`)
		Padded version of *a*. If a `numpy.array` was passed, a copy with same
		dtype, but added dimensions is returned. `numpy.pad` returns copies,
		so this does not support `views`. If a shape was passed, the same shape
		object (without being copied) is returned.
		
	`numpy.ndarray`, ((`tuple`, `list`) of `int`)
		Padded version of *b*.

	"""
	
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
		raise IndexError(
				f"Can only pad axis length of arrays with equal axes count, "
				f"but got shapes {ashape} and {bshape}.",
				ashape,
				bshape,
		)
	
	#Return copies, if a and b are the same.
	if ashape == bshape:
		#If i is also its shape, a was given as shape, not ndarray.
		#But we return shapes as-is and not as copies below, so we do not
		#copy then here.
		if a is ashape:
			acopy = a
		else:
			acopy = a.copy()
		if b is bshape:
			bcopy = b
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
