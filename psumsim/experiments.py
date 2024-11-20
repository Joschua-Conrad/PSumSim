"""Running a huge number of `simulateMvm` and `computeSqnr` calls.

Use functions defined here for exploring an MVM desing space.

The application first reduces along `ACT_AXIS`, `WEIGHT_AXIS` and
lastly along `MAC_AXIS`, but this one is chunked into tiles fitting on MVM
hardware. The results are clipped and *intermediate quantization* is applied
(like by an ADC in an in-memory compute MVM [IMC]_). After
combining the results of the chunks, the MVM algorithm applies a
*final quantization*.

When wanting to run an own design-space exploration, one probably needs to
change code in here. Check `readmeinstall` and use an *editable* installation
in that case."""

import collections
import re
import itertools
import numpy as np
import progressbar
import os
import json
import argparse
import pathlib
import multiprocessing
from .hist import histlenToBincount, HIST_AXIS, ACT_AXIS, WEIGHT_AXIS, MAC_AXIS, STAT_AXIS
from .simulate import simulateMvm, computeSqnr, optimumClippingCriterion, equalizeQuantizedUnquantized
from psumsim import __version__

NUMMACS = [64, 128,]
"""`list` of `int`
*nummacs* to explore."""

CHUNKSIZES = [None, 10, 32, 64, 100, 128,]
"""`list` of (`int`, `None`)
Chunksizes to explore. `None` uses no chunking and only *final quantization*
is applied."""

LEVELS = [None, 1, 3, 7, 15, 31,]
"""`list` of (`int`, `None`)
Number of levels to use to draw activations and weights and to quantize.
"""

RANDOM_BEHAVES = ["uniform", "norm", "truncnorm",]
"""`list` of `str`
Random models to draw actvations and weights from.
See `generateSimulationOperands`."""

#Again, first list None to first generate references
CLIP_LIMITS = [None, "occ", 3]
"""`list` of (`str`, `None`, `int`, `float`)
Cliplimits applied when quantizing and drawing operands.
See `clipping` and `generateSimulationOperands`.
"""

STATISTICS_DIM = 100
"""`int` : Number of computations to run in parallel for each experiment to
conduct selfcheck."""

RUN_DESCRIPTION_CORE_CLS = collections.namedtuple(
		typename="RUN_DESCRIPTION_CORE_CLS",
		field_names=(
				#Arguments needed to call simulateMvm
				"nummacs",
				"chunksize",
				"activationlevels",
				"weightlevels",
				"intermediatelevels",
				"finallevels",
				"initialcliplimit",
				"intermediatecliplimit",
				"finalcliplimit",
				"randombehave",
				#Another run description, which is used to compare
				#quantization error to. This is a run without any
				#quantization, so one without any cliplimit, so there is
				#no need to set cliplimit from stddev to fixed to make
				#them comparable.
				"sqnrreference",
				#Is set, if a description is a reference
				"issqnrreference",
				#If set, this run is invalid and will be skipped
				"skipthisrun",
		),
)
"""`collections.namedtuple` : All named fields to describe a single experiment.

This defines which information is needed to define a single experiment from
all the many ones exploring design space.

nummacs
	Drawn from `NUMMACS` and forwarded to `simulateMvm`.
	
chunksize
	Drawn from `CHUNKSIZES`. If `None`, there is no chunking of `MAC_AXIS` and
	only final quantization.

activationlevels
	Drawn from `LEVELS` and forwarded to `generateSimulationOperands`.
	
weightlevels
	Drawn from `LEVELS` and forwarded to `generateSimulationOperands`.
	
intermediatelevels
	Drawn from `LEVELS` and forwarded to `simulateMvm` for parameterizing
	intermediate quantization of chunked results. Note that this is a
	*histlen* (see `dataformat`). Is forwarded with a negative sign.
	
finallevels
	Drawn from `LEVELS` and forwarded to `simulateMvm` for parameterizing
	final quantization of the overall result. Note that this is a
	*histlen* (see `dataformat`). Is forwarded with a negative sign.
	
intermediatecliplimit
	Drawn from `CLIP_LIMITS`. Forwarded to `simulateMvm` for parameterizing
	intermediate quantization of chunked results.
	
finalcliplimit
	Drawn from `CLIP_LIMITS` and forwarded to `simulateMvm` for parameterizing
	final quantization of the overall result.
	
randombehave
	Drawn from `RANDOM_BEHAVES` and forwarded to `generateSimulationOperands`.
	
sqnrreference
	`None` or an instance of `RunDescription`. This run generates the reference
	for `computeSqnr`. If intermediate and final quantization is used, the reference
	has only final quantization but is similar otherwise. If only final
	quantization is used, the reference has no quantization. If no quantization
	is used, there is no reference and there will be no SQNR.
	
issqnrreference
	`bool`. Is set for experiments, which are needed as SQNR reference somewhere.
	
skipthisrun
	`bool`. Is set, if the specified experiment shall be skipped. There are
	several reasons coded in `RunDescription.__new__`. It e.g. does not
	make sense to have a *chunksize*, if there is no intermediate quantization.
	The number of weight levels cannot be higher than number of activation levels
	to prevent redundant cases. Final or intermediate clipping can only work,
	if the corresponding quantization is activated, too.
"""

class RunDescription(RUN_DESCRIPTION_CORE_CLS):
	"""Wrapper for `RUN_DESCRIPTION_CORE_CLS`.
	
	This adds more conversion methods to the experiment description."""
	
	#RE to turn a run description identifier back into an object
	STR_TO_RUN_DESCRIPTION_RE = re.compile((
			r"rb_(?P<randombehave>[^_]+)_"
			r"sc_(?P<initialcliplimit>[^_]+)_"
			r"nm_(?P<nummacs>[^_]+)_"
			r"al_(?P<activationlevels>[^_]+)_"
			r"wl_(?P<weightlevels>[^_]+)_"
			r"ic_(?P<intermediatecliplimit>[^_]+)_"
			r"fc_(?P<finalcliplimit>[^_]+)_"
			r"cs_(?P<chunksize>[^_]+)_"
			r"fl_(?P<finallevels>[^_]+)_"
			r"il_(?P<intermediatelevels>[^_]+)"
	))
	"""`re.Pattern` : Compiled regular expression to turn `str` experiment
	descriptions back to objects."""
	
	@classmethod
	def cliplimitFilter(cls, val):
		"""Normalize a `CLIP_LIMITS` type.
		
		`None` is kept, even if given as `str`. `float` is kept. `int` is
		turned to `float`. If any other value can be converted to `float`,
		`float` is kept. Any value still not matching is converted to
		`str`.
		
		Parameters
		----------
		val : `None`, `str`, `float`, `int`, `object`
			Value to normalize in type.

		Returns
		-------
		val : `None`, `str`, `float`, `object`
			Type-normalized value.

		"""
		
		if val in (None, str(None)):
			val = None
		elif isinstance(val, (float, int)):
			val = float(val)
		else:
			try:
				val = float(val)
			except ValueError:
				val = str(val)
		
		return val
	
	@classmethod
	def levelChunksizeFilter(cls, val):
		"""Normalize a `CHUNKSIZES` or `LEVELS` type.
		
		`None` is kept, even if given as `str`. Everything else is converted to
		`int`.
		
		Parameters
		----------
		val : `None`, *"None"*, `float`, `int`
			Value to normalize in type.

		Returns
		-------
		val : `None`, `int`
			Type-normalized value.
		"""
		
		if val in (None, str(None)):
			val = None
		else:
			val = int(val)
		
		return val

	def __new__(
			cls,
			nummacs,
			chunksize,
			activationlevels,
			weightlevels,
			intermediatelevels,
			finallevels,
			initialcliplimit,
			intermediatecliplimit,
			finalcliplimit,
			randombehave,
			allowskip,
			createdummy,
		):
			"""Create a new description of a single experiment.
		
			This basically replaces *__init__*, because that is how
			the base class `collections.namedtuple` works.
	
			Parameters
			----------
			allowskip : `bool`
				If *skipthisrun* is set and this is `False`, an exception is
				raised.
				
			createdummy : `bool`
				If set, *sqnrreference* is always `None`. This is mainly needed
				to not possible get (recursive) loops of references pointing
				to experiments they were requested from.
				
			Other Parameters
			----------------
			Values set as fields of `RUN_DESCRIPTION_CORE_CLS`. Fields
			*sqnrreference*, *issqnrreference* and *skipthisrun* are set
			automatically. `cliplimitFilter` and `levelChunksizeFilter` are
			used for type normalization.
	
			Raises
			------
			`RuntimeError`
				If the experiment would be skipped, but *allowskip* is not set.
	
			Returns
			-------
			`type`
				Is retrieved thru the base class.
	
			"""
		
			#Normalize types
			nummacs = int(nummacs)
			chunksize = cls.levelChunksizeFilter(chunksize)
			activationlevels = cls.levelChunksizeFilter(activationlevels)
			weightlevels = cls.levelChunksizeFilter(weightlevels)
			intermediatelevels = cls.levelChunksizeFilter(intermediatelevels)
			finallevels = cls.levelChunksizeFilter(finallevels)
			initialcliplimit = cls.cliplimitFilter(initialcliplimit)
			intermediatecliplimit = cls.cliplimitFilter(intermediatecliplimit)
			finalcliplimit = cls.cliplimitFilter(finalcliplimit)
			randombehave = str(randombehave)
		
			#Decide when a run makes no sense
			skipthisrun = False
			#If a single chunk includes all macs. That shall be done with
			#chunksize None
			skipthisrun = skipthisrun or ((chunksize is not None) and (chunksize >= nummacs))
			#If weight or activation bitwidth is None, they have to be quantized
			skipthisrun = skipthisrun or ((activationlevels is None) or (weightlevels is None))
			#If weight bitwidth is larger than activation bitwidth
			skipthisrun = skipthisrun or (weightlevels > activationlevels)
			#If input and output activations are quantized differently.
			skipthisrun = skipthisrun or ((finallevels is not None) and (finallevels != activationlevels))
			#If the intermediate stage is quantized and generates more levels than
			#final one
			skipthisrun = skipthisrun or ((finallevels is not None) and (intermediatelevels is not None) and (intermediatelevels > finallevels))
			#If the intermediate stage is quantized, but the final one is not
			#No need to test with whether stimulus is quantized, because this
			#is always quantized.
			skipthisrun = skipthisrun or ((intermediatelevels is not None) and (finallevels is None))
			#If we have no chunksize, but would apply intermediate quantization,
			#which is the most advanced one. When having no overchunkaxis, we
			#do not sum over that and there is one opportunity less to quantize.
			skipthisrun = skipthisrun or ((intermediatelevels is not None) and (chunksize is None))
			#But if we would apply no intermediate quantization, it does not
			#make any sense to apply chunksize
			skipthisrun = skipthisrun or ((intermediatelevels is None) and (chunksize is not None))
			#Also check that cliplimit is only applied on intermiediate, if
			#also applied on output. And additionally, that the output only
			#applies it if the input operands also have it.
			skipthisrun = skipthisrun or ((intermediatecliplimit is not None) and (finalcliplimit is None))
			skipthisrun = skipthisrun or ((finalcliplimit is not None) and (initialcliplimit is None))
			#If one of the two quantizer stages has a cliplimit, but no activated quantizer.
			#For initial cliplimit, we do not test that as weights/activations
			#from stimulus are always quantized.
			skipthisrun = skipthisrun or ((intermediatecliplimit is not None) and (intermediatelevels is None))
			skipthisrun = skipthisrun or ((finalcliplimit is not None) and (finallevels is None))
			#If intermediate and final quantizer apply a cliplimit, but not the same one
			#Similar checks against initial cliplimit
			skipthisrun = skipthisrun or ((intermediatecliplimit is not None) and (finalcliplimit is not None) and (intermediatecliplimit != finalcliplimit))
			skipthisrun = skipthisrun or ((intermediatecliplimit is not None) and (initialcliplimit is not None) and (intermediatecliplimit != initialcliplimit))
			skipthisrun = skipthisrun or ((initialcliplimit is not None) and (finalcliplimit is not None) and (initialcliplimit != finalcliplimit))
			#None cliplimit for input operands do not work. They always
			#need a limit
			skipthisrun = skipthisrun or (initialcliplimit is None)
			
			#If this run is skipped, do skip it if allowed.
			if skipthisrun and (not allowskip):
				raise RuntimeError(
						"Would skip this rundescription, but am not "
						"allowed to."
				)

			
			#Sort all reference args in a dict, which makes keeping keys
			#easier
			givenargs = dict(
					nummacs=nummacs,
					chunksize=chunksize,
					activationlevels=activationlevels,
					weightlevels=weightlevels,
					intermediatelevels=intermediatelevels,
					finallevels=finallevels,
					initialcliplimit=initialcliplimit,
					intermediatecliplimit=intermediatecliplimit,
					finalcliplimit=finalcliplimit,
					randombehave=randombehave,
			)
			
			#If this is not a dummy, include all arguments needed for references.
			if not createdummy:
				#Reference for finding SQNR: If we do intermediate quantization,
				#remove that. If we do final quantization, remove that.
				#If these stages do not have quantization, they
				#also have no cliplimit. But the stimulus is generated the exact
				#same way including same cliplimit for stimulus.
				sqnrreferenceargs = givenargs.copy()
				if sqnrreferenceargs["intermediatelevels"] is not None:
					sqnrreferenceargs["intermediatelevels"] = None
					sqnrreferenceargs["intermediatecliplimit"] = None
					#When removing interm quantization, we also have to remove
					#chunksize. The axis then does not make any difference anymore.
					sqnrreferenceargs["chunksize"] = None
				elif sqnrreferenceargs["finallevels"] is not None:
					sqnrreferenceargs["finallevels"] = None
					sqnrreferenceargs["finalcliplimit"] = None
				#If the values we set to None above are already None, we have
				#no sqnr reference.
				if (givenargs == sqnrreferenceargs):
					sqnrreference = None
				else:
					#If we set a sqnrreference, ensure that this rundescription
					#is a proper dummy which will only be used as key.
					#This breaks an __init__ recursion across runs.
					#Still, a reference has the proper dtype.
					#If we create a reference here, it shall not be skipped.
					#Because then no SQNR could be computed. Only if the run itself
					#is skipped, it could happen that the reference is also
					#skipped.
					sqnrreference = cls(
							**sqnrreferenceargs,
							allowskip=skipthisrun,
							createdummy=True,
					)

			#Otherwise, we have no reference and are not reference. The information
			#is not provided in dummies.
			else:
				sqnrreference = None
			
			#Now construct the full case with its reference. 
			#Also define whether this is a reference.
			#Every case not
			#having the intermiediate (most advanced) quantization is somewhen
			#needed as reference.
			#And whether tis run is skipped was decided above
			return super().__new__(
					cls,
					**givenargs,
					sqnrreference=sqnrreference,
					issqnrreference = (givenargs["intermediatelevels"] is None),
					skipthisrun=skipthisrun,
			)
		
	def __getnewargs__(self):
		"""Make objects of this class pickleable.
		
		See `pickle`. Allows to pass objects of this class thru
		`multiprocessing`. We have fields, which we memorize but which we
		do not support as arguments in `__new__` and vice-versa. This is
		why this method needs fixing.

		Returns
		-------
		newargs : `tuple`
			Fields *sqnrreference*, *issqnrreference* and *skipthisrun*
			are removed. *allowskip* is therefore always `True` (the
			application unpickling stuff can check this) and
			*createdummy* is set if we do not know a *sqnrreference*.
		"""
		
		newargs = super().__getnewargs__()
		#Remove args which can be re-created in __new__
		newargs = newargs[:-3]
		#Add argument allowskip: we allow it, otherwise __new__ should have
		#complained already. Also add *makedummy*, where we are one if
		#our sqnrreference was None.
		newargs = (*newargs, True, (self.sqnrreference is None))
		return newargs
	
	def copy(
			self,
			allowskip,
			createdummy,
		):
		"""Create a copy, possibly converting to/from dummies.
		
		Parameters
		----------
		allowskip : `bool`
			See `__new__`.
		createdummy : `bool`
			See `__new__`.

		Returns
		-------
		`RunDescription`
			The copy.
		"""
		
		#When recreating, skip fields that are re-derived in init
		return type(self)(
				*self[:-3],
				allowskip=allowskip,
				createdummy=createdummy,
		)
	
	def toStr(self):
		"""Convert to `str`.
		
		A format as recognized by `STR_TO_RUN_DESCRIPTION_RE` is used.

		Returns
		-------
		result : `str`
			Run description as `str`.

		"""
		
		#If that does not destroy information, try to turn cliplimits, which can
		#be float, into int to make strings shorter
		cliplimitcases = (
				self.initialcliplimit,
				self.intermediatecliplimit,
				self.finalcliplimit,
		)
		processedcliplimitcases = list()
		for cliplimit in cliplimitcases:
			if (cliplimit is not None) and (not isinstance(cliplimit, str)) and (cliplimit == int(cliplimit)):
				cliplimit = int(cliplimit)
			processedcliplimitcases.append(cliplimit)
		initialcliplimit, intermediatecliplimit, finalcliplimit = \
				processedcliplimitcases
		
		result = (
				f"rb_{self.randombehave}_"
				f"sc_{initialcliplimit}_"
				f"nm_{self.nummacs}_"
				f"al_{self.activationlevels}_"
				f"wl_{self.weightlevels}_"
				f"ic_{intermediatecliplimit}_"
				f"fc_{finalcliplimit}_"
				f"cs_{self.chunksize}_"
				f"fl_{self.finallevels}_"
				f"il_{self.intermediatelevels}"
		)
		
		return result

	@classmethod
	def fromStr(cls, thestr, allowskip, createdummy):
		"""Create object from `str`.
		
		`STR_TO_RUN_DESCRIPTION_RE` is used for the conversion.
		
		Parameters
		----------
		thestr : `str`
			Convert this.
		allowskip : `bool`
			See `__new__`.
		createdummy : `bool`
			See `__new__`.

		Raises
		------
		`ValueError`
			If *thestr* does not match `STR_TO_RUN_DESCRIPTION_RE`.

		Returns
		-------
		rundescription : `RunDescription`
			Converted object.

		"""
		
		rematch = cls.STR_TO_RUN_DESCRIPTION_RE.fullmatch(thestr)
		if not rematch:
			raise ValueError(
					f"{thestr} is not a valid run description string.",
					thestr,
			)
			
		rundescription = cls(**rematch.groupdict(), allowskip=allowskip, createdummy=createdummy)
		
		return rundescription
	
def runIter():
	"""Generate all possible experiments.
	
	This takes `RANDOM_BEHAVES`, `CLIP_LIMITS`, `NUMMACS` and `LEVELS` and
	creates all possible combinations of them. Thereby, e.g. *activationlevels*
	as well as *weightlevels* both originate from `LEVELS`, but still are drawn
	separately and can differ. The rundescriptions here come out in an order,
	where needed *sqnrreference* always occurs before the run needing it.

	Yields
	------
	rundescription : `RunDescription`
		A rundescription. Ones with *skipthisrun* are ignored.

	"""
	
	#Iterate over all combinations of values
	

	fulliter = itertools.product(
			RANDOM_BEHAVES,
			CLIP_LIMITS,
			NUMMACS,
			LEVELS,
			LEVELS,
			CLIP_LIMITS,
			CLIP_LIMITS,
			CHUNKSIZES,
			LEVELS,
			LEVELS,
	)
	
	for runcase in fulliter:
		
		#The experiment order is important. Inner most we must find first
		# added final and inter quantization and chunksize. That way, we generate
		#needed references first and keep them then. We also try to
		#pass options influencing runtime somewhere inside, as that makes
		#progress estimation easier.
		#On the outside, we must pass everything that allows us to forget
		#old references
		randombehave, \
		initialcliplimit, \
		nummacs, \
		activationlevels, \
		weightlevels, \
		intermediatecliplimit, \
		finalcliplimit, \
		chunksize, \
		finallevels, \
		intermediatelevels = runcase
				
		rundescription = RunDescription(
				nummacs=nummacs,
				chunksize=chunksize,
				activationlevels=activationlevels,
				weightlevels=weightlevels,
				intermediatelevels=intermediatelevels,
				finallevels=finallevels,
				initialcliplimit=initialcliplimit,
				intermediatecliplimit=intermediatecliplimit,
				finalcliplimit=finalcliplimit,
				randombehave=randombehave,
				#Skipping is allowed, we remove these descriptions then.
				#That way, we can combine all limits etc. blindly.
				allowskip=True,
				#Produce no dummies, but rather descriptions which are ready
				#to be simulated
				createdummy=False,
		)
		
		#THe above method sets a flag, if a run description is invalid
		if rundescription.skipthisrun:
			continue
		else:
			yield rundescription
			
def doSingleRun(rundescription, sqnrreferencereturn):
	"""Run a single experiment.
	
	- `simulateMvm` is called with *dostochastic* (see `statstoc`).
	
	- `simulateMvm` is also called with `STATISTICS_DIM` *dostatisticdummy*
	  computations with enabled self-check to validate the setup.
	  
	- `simulateMvm` is also called with a single, *dostatisticdummy* 
	  compuation with full-scale operands to detect overflows.
	  
	- `equalizeQuantizedUnquantized` and `computeSqnr` are used to calculate
	  SQNR.
	
	Clipping criterions *"occ"* are looked-up using `optimumClippingCriterion`.
	If an experiment has no *chunksize*, there is only final quantization.

	Parameters
	----------
	rundescription : `RunDescription`
		The experiment to run.
		
	sqnrreferencereturn : `dict`, `None`
		The reference for SQNR computation. Can be skipped, but there are
		no SQNR results then. Needs to provide keys *finalresult* and
		*finalmergevalues*, which contain last elements of *results* and
		*mergevaluess* returned by `simulateMvm`. If not already present,
		keys *results* and *mergevaluess* are set from these, such that this
		is compatible with `equalizeQuantizedUnquantized`. Also the key
		*fewmergevalues* must be present. This is allowed to have only two
		values, as there are only two calls to `reduceSum` for experiments
		without a chunksize. A key *mergevaluess* is set from a padded
		version to again be compatible with `equalizeQuantizedUnquantized`.
		
		.. note::
			This `dict` get new keys set, which is an in-place alternation.
			
		`singleRunWorker` returns values which can be used here.

	Raises
	------
	`RuntimeError`
		If a skipped rundescription is passed.

	Returns
	-------
	retstochastic : `dict`
		Return of *dostochastic* call of `simulateMvm`.
		
	snr : `float`, `None`
		Returned by `computeSqnr`, if *sqnrreferencereturn* was given.
		
	probabilityerror : `numpy.ndarray`, `None`
		Returned by `computeSqnr`, if *sqnrreferencereturn* was given.
		
	magnitudeerror : `numpy.ndarray`, `None`
		Returned by `computeSqnr`, if *sqnrreferencereturn* was given.
		
	squarederror : `numpy.ndarray`, `None`
		Returned by `computeSqnr`, if *sqnrreferencereturn* was given.


	"""
	
	if rundescription.skipthisrun:
		raise RuntimeError(
				f"rundescription {rundescription} is skipped, but then this "
				f"function should not have been called.",
				rundescription,
		)
				
	#Process the cliplimits. If they are the OCC, the OCC function must be
	#called and corresponding bitwidth must be passed
	cliplimitcases = (
			(rundescription.initialcliplimit, rundescription.activationlevels),
			(rundescription.initialcliplimit, rundescription.weightlevels),
			(rundescription.intermediatecliplimit, rundescription.intermediatelevels),
			(rundescription.finalcliplimit, rundescription.finallevels),
	)
	processedcliplimitcases = list()
	for cliplimit, levels in cliplimitcases:
		if cliplimit == "occ":
			#All levels are a histlen, but the OCC expects a bincount.
			levels = histlenToBincount(histlen=levels)
			cliplimit = optimumClippingCriterion(
					levels=levels,
			)
		processedcliplimitcases.append(cliplimit)
		
	activationcliplimit, weightcliplimit,  intermediatecliplimit, \
			finalcliplimit = processedcliplimitcases
			
	#Levels are made negative, if they are not None. Because negative
	#mergevalues mean: keep that number of levels
	levelcases = (
			rundescription.intermediatelevels,
			rundescription.finallevels,
	)
	processedlevelcases = list()
	for level in levelcases:
		if level is not None:
			level = -level
		processedlevelcases.append(level)
		
	intermediatelevels, finallevels = processedlevelcases
				
	commongroup = dict(
			#Is usually set by other methods
			cliplimitfixed=None,
			#Is set automatically if posweights are hist
			positionweightsonehot=None,
			#Only needed if weight and act axis is first reduced
			disablereducecarries=None,
			#Is only needed, if a hist axis is chunked
			chunkoffsetsteps=None,
			#Yes, we want fast code
			allowoptim=True,
	)
	
	#Group templates for this run
	overmacsgroup = dict(
			**commongroup,
			reduceaxes=(MAC_AXIS,),
			chunksizes=(rundescription.chunksize,),
			mergevalues=None,
			cliplimitstddev=None,
			positionweights=None,
			histaxis=HIST_AXIS,
			docreatehistaxis=True,
			mergeeffortmodel=None,
	)
	overweightsactgroup = dict(
			**commongroup,
			reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
			chunksizes=None,
			mergevalues=intermediatelevels,
			cliplimitstddev=intermediatecliplimit,
			positionweights=("hist", "hist",),
			histaxis=HIST_AXIS,
			docreatehistaxis=False,
			mergeeffortmodel="analog",
	)
	overchunksgroup = dict(
			**commongroup,
			reduceaxes=(-2,),
			chunksizes=None,
			mergevalues=finallevels,
			cliplimitstddev=finalcliplimit,
			positionweights=None,
			histaxis=HIST_AXIS,
			docreatehistaxis=False,
			mergeeffortmodel=None,
	)
	
	#If the chunksize is None, remove the operation adding over chunks.
	#That also removes one quantization. Then try to keep the final
	#quantization opportunity including its effort.
	if rundescription.chunksize is None:
		overweightsactgroup["mergevalues"] = overchunksgroup["mergevalues"]
		overweightsactgroup["cliplimitstddev"] = overchunksgroup["cliplimitstddev"]
		overweightsactgroup["mergeeffortmodel"] = overchunksgroup["mergeeffortmodel"]
		overchunksgroup = None
		
	#Assemble the groups
	if rundescription.chunksize is None:
		groups = (overmacsgroup, overweightsactgroup,)
	else:
		groups = (overmacsgroup, overweightsactgroup, overchunksgroup,)
		
	#At least there is no need to set cliplimitfixed to equalize in
	#comparison to sqnrref using applyCliplimitStddevAsFixedFrom. Because
	#when adding final quantization, our reference has no quantiztion and
	#hence also no cliplimit to copy. And when adding intermediate quant,
	#we maybe could copy cliplimit from final quant, but that stage sees
	#introduced intermediate quant, so re-finind gcliplimitstddev threre
	#makes sense.
		
	#Common arguments for simulateMvm
	commonargs = dict(
			selfcheckindummy=True,
			activationlevels=rundescription.activationlevels,
			weightlevels=rundescription.weightlevels,
			nummacs=rundescription.nummacs,
			randomclips=(activationcliplimit, weightcliplimit,),
	)
	
	#Simulate stochastic or a small statistic set for selfcheckindummy
	#or a single selfcheck with fullscale (to detect overflows)
	stochasticargs = dict(
			statisticdim=None,
			dostatisticdummy=False,
			randombehave=rundescription.randombehave,
	)
	statisticrandomargs = dict(
			statisticdim=STATISTICS_DIM,
			dostatisticdummy=True,
			randombehave=rundescription.randombehave,
	)
	statisticfullscaleargs = dict(
			statisticdim=1,
			dostatisticdummy=True,
			randombehave="fullscale",
	)

	#Stochastic experiment
	retstochastic = simulateMvm(
			groups=groups,
			**commonargs,
			**stochasticargs,
	)
	
	#Two selfchecks with random numbers and fullscale
	simulateMvm(
			groups=groups,
			**commonargs,
			**statisticrandomargs,
	)
	simulateMvm(
			groups=groups,
			**commonargs,
			**statisticfullscaleargs,
	)
	
	#If there is no reference, there is also no SQNR to compute.
	if sqnrreferencereturn is None:
		return retstochastic, None, None, None, None
	
	#results and mergevalues and maxhistvalue are provided by reference,
	#but in wrong format. We rectify that now. Do only, if another run did
	#not fix that already.
	if "results" not in sqnrreferencereturn:
		sqnrrefresults = sqnrreferencereturn["finalresult"]
		#Wrap into iterator, because the results key holds results
		#of multiple groups.
		sqnrrefresults = (sqnrrefresults,)
		sqnrreferencereturn["results"] = sqnrrefresults
		
	if "maxhistvalues" not in sqnrreferencereturn:
		sqnrrefmaxhistvalues = sqnrreferencereturn["finalmaxhistvalue"]
		#Wrap into iterator, because the maxhistvalues key holds results
		#of multiple groups.
		sqnrrefresults = (sqnrrefmaxhistvalues,)
		sqnrreferencereturn["maxhistvalues"] = sqnrrefresults
		
	#Mergevalues always exist, but if we have chunks and the previous one did
	#not, we need to add a dummy mergevalues for the intermediate quantization
	if "mergevaluess" not in sqnrreferencereturn:
		sqnrmergevaluess = sqnrreferencereturn["fewmergevaluess"]
		if (rundescription.chunksize is not None) and (rundescription.sqnrreference.chunksize is None):
			if len(sqnrmergevaluess) != 2:
				raise RuntimeError(
						f"When having no chunksize in a reference, two elems "
						f"are expected in mergevaluess, but got "
						f"{len(sqnrmergevaluess)}.",
						sqnrmergevaluess,
				)
			#Move None in for the intermediate stage. A run without chunksize
			#has only final quant.
			sqnrmergevaluess = (sqnrmergevaluess[0], None, sqnrmergevaluess[1])
		sqnrreferencereturn["mergevaluess"] = sqnrmergevaluess
		
	#Use method, which equalizes the two histograms to same length.
	#This needs results and mergevaluess and maxhistvalue of last run to work.
	#We here maybe compare sqnrreferencereturn with only one element in
	#results and maxhistvalue, but equalizeQuantizedUnquantized does not
	#care and simply selects the last value in both.
	#mergevalues has been synchronized to same length.
	resunquant, resquant = equalizeQuantizedUnquantized(
			retunquantized=sqnrreferencereturn,
			retquantized=retstochastic,
			runidx=-1,
			histaxis=HIST_AXIS,
			stataxis=STAT_AXIS,
			dostatistic=False,
			dostatisticdummy=False,
	)
	
	#Get a refbincount derived from maxhistvalue of reference experiment.
	#The unquantized one has more bins and the equalizer equalized the
	#quantized one to that.
	refhistlen = np.squeeze(sqnrreferencereturn["finalmaxhistvalue"]).item()
	refbincount = histlenToBincount(histlen=refhistlen)
	
	#Get SNR and bin-wise qauntization error powers
	snr, probabilityerror, magnitudeerror, squarederror = computeSqnr(
			unquantized=resunquant,
			quantized=resquant,
			histaxis=HIST_AXIS,
			stataxis=STAT_AXIS,
			#Pass None and ask to use shape to find this
			bincount=refbincount,
			dostatistic=False,
			dostatisticdummy=False,
			errordtype="float",
	)
	
	snr = np.squeeze(snr).item()
	
	#Return everything we have
	return retstochastic, snr, probabilityerror, magnitudeerror, squarederror

def singleRunWorker(rundescription, sqnrreferencereturn):
	"""Wrapper for `doSingleRun`.
	
	`doSingleRun` keeps all results, but that might be too much when running
	many experiments and when memorizing many results for having them ready
	as *sqnrreference*.
	

	Other Parameters
	----------------
	:
		See `doSingleRun`.
	
	Returns
	-------
	result : `dict`
		Keys are `str`:
			
		snr
			As returned by `doSingleRun`.
			
		totalmergeeffort
			`float` or `None`. A sum over *mergeeffort* of all occured
			quantizations in the experiment. `None` if there was no quantization.

		finalmaxhistvalue
			`numpy.ndarray`. Result from `simulateMvm` from last
			summation/quantization step.
			
		fewmergevaluess
			`list` of `numpy.ndarray`. Result from `simulateMvm`, which can
			have two or three entries depending on whether *chunksize* was
			present or not.
			
		finalresult
			`numpy.ndarray`. Result from `simulateMvm` from last
			summation/quantization step.
		
		cliplimitfixeds
			`numpy.ndarray`. Result from `simulateMvm`.
	"""
	
	#Most return values will be dropped. This is the space, where we make
	#the step towards reduced data possible to serialize.
	retstochastic, snr, probabilityerror, magnitudeerror, squarederror = \
			doSingleRun(
					rundescription=rundescription,
					sqnrreferencereturn=sqnrreferencereturn,
			)
			
	#Gather results we really want to keep
	if snr is not None:
		snr = np.squeeze(snr).item()
	#Gather and sum mergeefforts. These are already float dtype
	if all((i is None for i in retstochastic["mergeefforts"])):
		totalmergeeffort = None
	else:
		totalmergeeffort = sum((i for i in retstochastic["mergeefforts"] if i is not None))
	#Maxhistvalues are needed for shape equalization. Gather a final value.
	finalmaxhistvalue = retstochastic["maxhistvalues"][-1]
	#We need mergevaluess and last results for equalizeQuantizedUnquantized
	#mergevaluess is already float. We give them another name, because we
	#maybe have only 2 elems here, if not chunksize was given.
	fewmergevaluess = retstochastic["mergevaluess"]
	#We need only the last results, but have to turn them into some
	#picklable dtype
	finalresult = retstochastic["results"][-1]
	#valuescale is a list of float. We also return that, as one
	#can determine the vaue range which maybe needs to be amplified by an ADC
	cliplimitfixeds = retstochastic["cliplimitfixeds"]
	
	
	#Return just the python dtype values which we need later
	result = dict(
			snr=snr,
			totalmergeeffort=totalmergeeffort,
			finalmaxhistvalue=finalmaxhistvalue,
			fewmergevaluess=fewmergevaluess,
			finalresult=finalresult,
			cliplimitfixeds=cliplimitfixeds,
	)
	
	return result

def runAllExperiments(
			runreduced,
			runquiet,
			jsonfp,
			iterbegin,
			iterend,
			runkeys,
			processes,
			aggressive,
			progressfp,
	):
	"""Run all experiments from `runIter`.
	
	This iterates over all experiments, calls `singleRunWorker`, keeps results
	as *sqnrreference* and writes minimal results to JSON file. The stock
	of results for *sqnrreference* is cleared if  *randombehave*,
	*initialcliplimit*, nummacs*, *activationlevels* or *weightlevels* changes.
	Because these are most rarely updated fields in `runIter` and hence if
	they change, the created *sqnrreference* results are never needed again.
	This reduces memory requirements, but requires to stick to the run order
	in `runIter`, even when running just single runs.

	Parameters
	----------
	runreduced : `bool`
		If set, `RANDOM_BEHAVES`, `CLIP_LIMITS`, `NUMMACS`, `CHUNKSIZES` and
		`LEVELS` are reduced to `None` and two normal values. Experiments
		finish rapidly then, which is nice for debugging. The parameter `list`
		objects are restored afterwards.
		
	runquiet : `bool`
		If set, skip printing of progressbar to *stdout* or *progressfp*.
		
	jsonfp : `io.TextIOBase`
		File handle to write `json` results to. If this is readable, it is
		read as `json` file and results are kept, not re-created and written
		to output file together with new results in the end. Use this to
		continue from a crashed simulation.
		
	iterbegin : `float`, `None`
		If set, is multiplied by the number of all experiments to run and then
		used in `itertools.islice` to run only a subset of the experiments.
		Use this to shard experiments, to run them across different machines.
		Each shard will detect which *sqnrreference* are needed and will create
		them. `runIter` will determine the correct order to run references
		first.
		
		.. note::
			Using infinately many shards makes no sense, as all the shards will
			need to create possibly the same *sqnrreference* results.
			
	iterend : `float`, `None`
		Like *iterbegin*, but specifies where to stop. A value of *1.0* will
		still ignore the last experiment.
		
	runkeys : `None`, ((`tuple` or `list`) of `str`)
		Names of specific runs to carry out. Interpreted using
		`RunDescription.fromStr`. `runIter` is still needed to find best experiment
		order. If given, *runbegin* and *runend* are ignored. If runs with
		set *skipthisrun* are specified, they are skipped silently.
	
	processes : `int`, `None`
		How many of the experiments to run in parallel workers.
		`multiprocessing.pool.Pool` is used. If one run needs a *sqnrreference* which
		does not exist yet, process creation halts and waits for previous
		experiments to finish. `None` lets `multiprocessing.pool.Pool`
		find a good value, which is probably the number of CPUs.
		Negative values or *0* are like `None`.
		
	aggressive : `bool`
		If set, remember more sqnr references. This allows to keep more workers
		busy at the cost for a larger memory footprint. If there is only a
		single process, this switch has no effect.
		
	progressfp : `io.TextIOBase`, `None`
		If given, progress information is written to this stream instead of
		to `sys.stdout`. Ignored if *runquiet* is set. Use this for setups,
		where you don't have access to th commandline output, as in clusters.

	Raises
	------
	`ValueError`
		If a run does not have *skipthisrun* set, but its *sqnrreference* has.
	
	`RuntimeError`
		If a call to `multiprocessing.pool.Pool` (and hence to `singleRunWorker`)
		raises an exception. That exception is re-raised, but the `RuntimeError`
		states the name of the run which failed. So one can debug it using
		*runkeys*.

	"""
	
	#If that is the case, remember class attributes, clear them and re-append
	#less values
	if runreduced:
		nummacs = tuple(NUMMACS)
		chunksizes = tuple(CHUNKSIZES)
		levels = tuple(LEVELS)
		randombehaves = tuple(RANDOM_BEHAVES)
		cliplimits = tuple(CLIP_LIMITS)
		NUMMACS.clear()
		CHUNKSIZES.clear()
		LEVELS.clear()
		RANDOM_BEHAVES.clear()
		CLIP_LIMITS.clear()
		NUMMACS.extend(nummacs[:2])
		#Include None and two numeric values here
		CHUNKSIZES.extend(chunksizes[:3])
		LEVELS.extend(levels[:3])
		RANDOM_BEHAVES.extend(randombehaves[:2])
		CLIP_LIMITS.extend(cliplimits[:2])
		
	#If processes is 0 or negative, use None which determines number of processes
	if (processes is not None) and processes <= 0:
		processes = None
		
	#Try to read the JSON file to initialize the results dict. Remember
	#which results already existed when starting
	if (jsonfp is not None) and jsonfp.readable():
		storeresults = json.load(jsonfp)
		existingkeys = tuple(storeresults.keys())
	else:
		storeresults = dict()
		existingkeys = tuple()
		
	#If user did not specify exactly which results are desired, use all results
	#and possibly select a slice
	if runkeys is None:
			
		#Prepare number of elems in the plain, uncut iterator. Needed to
		#interpret iterbein and end, which are relative to that.
		runcount = sum((1 for i in runIter()))
		
		#Make runbegin and end relative
		iterbeginends = list()
		for iterbeginend in (iterbegin, iterend,):
			if iterbeginend is not None:
				iterbeginend = int(iterbeginend * runcount)
			iterbeginends.append(iterbeginend)
		iterbegin, iterend = iterbeginends
			
		#Find experiments we keep because they are in the correct slice of the
		#iterator and because we do not have the result yet.
		runiter = runIter()
		#Access only some arguments
		runiter = itertools.islice(runiter, iterbegin, iterend)
		
	#Otherwise, turn runkeys into actual descriptions which are ready for
	#being run. Specifying runs which make no sense is allowed, but they
	#are ignored. Create non-dumm descriptions like runIter does
	else:
		runiter = (RunDescription.fromStr(thestr=i, allowskip=True, createdummy=False) for i in runkeys)
		runiter = (i for i in runiter if not i.skipthisrun)
	
	#Only keep needed results. Existingkeys holds strings, which are read
	#from existing JSON
	runiter = (i for i in runiter if (i.toStr() not in existingkeys))
	wantresults = set(runiter)
	
	#From that, extract which results these need as reference.
	#Turn references into non-dummies, so they are comparable to
	#waht runIter yields.
	needreferences = (i.sqnrreference for i in wantresults)
	needreferences = (i for i in needreferences if i is not None)
	needreferences = (i.copy(allowskip=False, createdummy=False) for i in needreferences)
	needreferences = set(needreferences)
	
	#To keep the order, iterate now over all experiments and keep the ones
	#we need for the result or reference.
	#Prepare multiple iterators for counting and for main operation.
	#Keep information if a returned run is for result or reference.
	#Add flag which can be set once a run has been submitted
	alliters = list()
	for iteridx in range(2):
		newiter = ([i, (i in wantresults), (i in needreferences), False] for i in runIter())
		newiter = (i for i in newiter if (i[1] or i[2]))
		alliters.append(newiter)
		
	mainiter, countiter = alliters
	
	#Exhaust one iterator for counting
	runcount = sum((1 for i in countiter))
	
	#If we are not quiet, prepare progbars. One for showing started jobs, one
	#for finished.
	if (not runquiet) and (progressfp is None):
		submittedbar = progressbar.ProgressBar(
				max_value=runcount,
				#Makes print with progabr look ok
				redirect_stdout=True,
				#Print finished bar below submitted
				line_offset = 1,
				#Prefix text before bar
				prefix="Submitted: ",
		)
		finishedbar = progressbar.ProgressBar(
				max_value=runcount,
				#Makes print with progabr look ok
				redirect_stdout=True,
				prefix="Collected: ",
		)
	else:
		submittedbar = None
		finishedbar = None
		
	#The progressfp has higher priority than progress bar, but is also skipped
	#if the run is quiet
	if runquiet:
		progressfp = None
		
	#Append one final element, which is None and collects all results
	mainiter = itertools.chain(mainiter, ([None, False, False, False],))
		
	sqnrreferences = dict()
	#If we detect a change in some of they keys of a simulation description,
	#we know that everything changed and even the reference without any
	#quantization is re-generated. These elements are on the outside
	#of the iterator, so we can forget all old references then.
	#A reference mostly has a differen quantization in intermediat
	#and final stage. That comes with different cliplimit. The chunksize
	#can also change to not have chunks but no intern quantization.
	lastrundescription = None
	forgetrefkeys = (
			"randombehave",
			"initialcliplimit",
			"nummacs",
			"activationlevels",
			"weightlevels",
	)
	
	#Will store async results here. The key is the rundescription dummy
	asyncresults = dict()
	
	#Count submitted and collected runs here
	submittingrun = 1
	collectingrun = 1
	
	#Whether we skip the pool for debugging. E.g. if there is only asingle process
	skipworkerpool = (processes == 1)
	
	#Will run experiments in parallel in a worker pool
	if not skipworkerpool:
		workerpool = multiprocessing.Pool(processes=processes)
		
	#There are no aggressive runs with only a single worker. A single worker
	#will perform run after run.
	aggressive = aggressive and (not skipworkerpool)
		
	#In aggressive run, we will iterate multiple times over mainiter and always
	#submit all runs we can find. But therefore, we also need need to store
	#all iter elemnts to iterate multiple times.
	if aggressive:
		mainiter = tuple(mainiter)
		
	#Start progab time measurement
	if submittedbar is not None:
		submittedbar.start()
	if finishedbar is not None:
		finishedbar.start()
	
	#Wrap all experiments in a finally, such that everything that worked
	#is written to disk
	try:
		#For aggressive runs, we have to iterate over mainiter many times
		while True:
			#Use this to remember whether any iter element submitted any run
			anysubmitted = False
			for thismain in mainiter:
				
				rundescription, isresult, isreference, runsubmitted = thismain
				
				#Continue, if this run has been submitted already
				if runsubmitted:
					continue
				
				#Continue, if we are in aggressive setup and try to find all runs
				#whose reference already exists and this one does not have
				#its reference ready yet
				if (rundescription is not None) and (not ((not aggressive) or (rundescription.sqnrreference is None) or (rundescription.sqnrreference in sqnrreferences))):
					continue
				
				#For adding runname to exceptions
				if rundescription is not None:
					thisrunkeystr = rundescription.toStr()
				
				#Forget references if possible to save memory.
				#All workers which still run and need the references already
				#received copies via multiprocessing system.
				#The runs which are processes at the moment will not create
				#references, because if they would, we would not be allowed to
				#clear references here and we would have waited to create
				#more processes due to missing reference.
				#rundescription can be None in last iteration, which collects all results
				#aggressive simulations keep all references to be able to submit
				#more workers at once.
				if (not aggressive) and (lastrundescription is not None) and (rundescription is not None):
					for k in forgetrefkeys:
						if getattr(rundescription, k) != getattr(lastrundescription, k):
							sqnrreferences.clear()
							break
						
				lastrundescription = rundescription
				
				#Get according reference. But not if we right now create a result
				#just to have it as a reference and actually never need its result.
				#This breaks the dependency of needing always evne the first reference.
				if (rundescription is not None) and (rundescription.sqnrreference is not None) and isresult:
					thisrefkeystr = rundescription.sqnrreference.toStr()
					#If the reference would be skipped, it cannot exist. Check that
					#here.
					if rundescription.sqnrreference.skipthisrun:
						raise ValueError(
								f"Reference {thisrefkeystr} for run {thisrunkeystr} "
								f"is a dummy.",
								thisrefkeystr,
								thisrunkeystr,
						)
					#If the reference is missing, we have to wait for getting
					#the result. We leave the completed run inside the list
					#of runs, as results are remembered below.
					#aggressive runs have completed all references in a previous
					#iteration and know that the references were completed
					#already. Same for skipped workerpool, where only one worker
					#operates at a time, so the previous one must have finished
					#already.
					if (not aggressive) and (not skipworkerpool) and (rundescription.sqnrreference not in sqnrreferences):
						asyncrefresult, _, _ = asyncresults[rundescription.sqnrreference]
						#This call blocks until the worked got the next result.
						#Skip it if we skipped all this worker stuff
						try:
							sqnrreference = asyncrefresult.get()
						except Exception as ex:
							raise ex from RuntimeError(
									f"Getting reference result {thisrefkeystr} for run "
									f"{thisrunkeystr} failed.",
									thisrefkeystr,
									thisrunkeystr,
							)
						sqnrreferences[rundescription.sqnrreference] = sqnrreference
					#Otherwise just get stored reference
					else:
						sqnrreference = sqnrreferences[rundescription.sqnrreference]
				else:
					sqnrreference = None
					
				#Print info about submitted run to progress file or bar.
				#Do so before possibly starting a run with skipworkerpool
				#runquiet is regarded above, where it sets progressbar and
				#progress file objects to None
				if (rundescription is not None):
					if submittedbar is not None:
						#Update progabr with force, otherwise the progbar
						#updates from last burst are visible only
						submittedbar.increment(force=True)
					if progressfp is not None:
						infostr = f"Submitting run {submittingrun}/{runcount}: {thisrunkeystr}\n"
						progressfp.write(infostr)
						#Flush, otherwise the progress display gets stuck in some buffer
						progressfp.flush()
					submittingrun += 1
					
				#Apply new job for workerpool, if rundescription is not None,
				#which it is in a lasat iteration which gathers all results.
				if (rundescription is not None):
					#Create minimum key without sqnrreference name and fields which can
					#be re-created from the rest. Dummies are used as keys.
					runkey = rundescription.copy(allowskip=False, createdummy=True)
					
					#Actual run inside the pool. Apply without of pool if needed
					workerargs = dict(
							rundescription=rundescription,
							sqnrreferencereturn=sqnrreference,
					)
					if not skipworkerpool:
						asyncresult = workerpool.apply_async(func=singleRunWorker, kwds=workerargs)
					else:
						#WHen calling this directly, errors are raised here and
						#not when calling get() on the async result. Add runname
						#to exception here.
						try:
							asyncresult = singleRunWorker(**workerargs)
						except Exception as ex:
							raise ex from RuntimeError(
									f"Run {thisrunkeystr} failed.",
									thisrunkeystr,
							)
					
					asyncresults[runkey] = (asyncresult, isresult, isreference)
					
					#Remember that while iterating over mainiter we submitted
					#something and that this particular run has been submitted.
					anysubmitted = True
					thismain[-1] = True
				
				#Go thru all asyncresults and collect the ones which are ready
				collected = list()
				for thisrunkey, asyncresultissmth in asyncresults.items():
					thisasyncresult, thisisresult, thisisreference = asyncresultissmth
					
					#Leave results which are still being processed alone.
					#Only when rundescription is None, we are in last run
					#and have to collect all results.
					#If we skipped the workers, results are ready, as the runs
					#were submitted synchronously
					if (not skipworkerpool) and (not thisasyncresult.ready()):
						if (rundescription is not None):
							continue
						else:
							#If we are supposed to collect all results,
							#wait for them.
							thisasyncresult.wait()
					
					#Create string from rundescription to be able to
					#add JSON dict entries. thisrunkey is a dummy, but
					#that makes no difference for string creation
					thisrunkeystr = thisrunkey.toStr()
				
					#Get the result. Is easy if no worker was used
					try:
						if skipworkerpool:
							thisresult = thisasyncresult
						else:
							thisresult = thisasyncresult.get()
					except Exception as ex:
						raise ex from RuntimeError(
								f"Run {thisrunkeystr} failed.",
								thisrunkeystr,
						)
						
					#Update proressbar of completed runs
					if finishedbar is not None:
						#Update progabr with force, otherwise the progbar
						#updates from last burst are visible only
						finishedbar.increment(force=True)
					if progressfp is not None:
						infostr = f"Collecting run {collectingrun}/{runcount}: {thisrunkeystr}\n"
						progressfp.write(infostr)
						#Flush, otherwise the progress display gets stuck in some buffer
						progressfp.flush()
					collectingrun += 1
					
					#Remember to remove this element later
					collected.append(thisrunkey)
				
					#Append reference
					if thisisreference:
						sqnrreferences[thisrunkey] = thisresult
						
					#Append results. Only what we need. Omit things, which are only needed
					#to use results as a reference. Use str key, as using a tuple
					#as key does not work with JSON
					if thisisresult:
						storeresults[thisrunkeystr] = dict(
								#Needed for estimating quantization effort and
								#what it gives
								snr=thisresult["snr"],
								totalmergeeffort=thisresult["totalmergeeffort"],
								#Handy for estimating where on the dynamicrange the
								#ADC values are derived
								cliplimitfixeds=thisresult["cliplimitfixeds"],
								#Number of ADC levels. Use the possibly shortter
								#list representing only two runs in an unchunked
								#experiment.
								fewmergevaluess=thisresult["fewmergevaluess"],
						)
						
				#Pop runkeys from async results. They have been processed.
				for thisrunkey in collected:
					asyncresults.pop(thisrunkey)
					
			#In aggressive runs, we iterate over mainiter only once and are done now.
			#Also break, if we are in aggressive mode, but did not submit anything
			#in this iteration
			if (not aggressive) or (not anysubmitted):
				break
			
	finally:
		#Cleanup workers
		if not skipworkerpool:
			workerpool.terminate()
		#DUmp the results we want to store.
		if jsonfp is not None:
			#Firs truncate old file
			if jsonfp.seekable():
				jsonfp.seek(0, os.SEEK_SET)
				jsonfp.truncate()
				
			json.dump(
					storeresults,
					jsonfp,
					allow_nan=True,
					indent="\t",
			)
		#Finalize progbar
		if submittedbar is not None:
			submittedbar.finish(dirty=True)
		if finishedbar is not None:
			finishedbar.finish(dirty=True)
		
		
	#Resotre class attributes if they were changed
	if runreduced:
		NUMMACS.clear()
		CHUNKSIZES.clear()
		LEVELS.clear()
		RANDOM_BEHAVES.clear()
		CLIP_LIMITS.clear()
		NUMMACS.extend(nummacs)
		CHUNKSIZES.extend(chunksizes)
		LEVELS.extend(levels)
		RANDOM_BEHAVES.extend(randombehaves)
		CLIP_LIMITS.extend(cliplimits)
		
def getArgParser():
	"""Return argument parser for calling `runAllExperiments` from
	commandline.
	
	Returns
	-------
	parser : `argparse.ArgumentParser`
		Commandline parser.

	"""
	
	parser = argparse.ArgumentParser(
			prog="psumsim",
			description="Simulate how partial sums in a matrix-vector multiplier"
			"behave when being quantized. The process ist simulated using "
			"stochastic histograms of values. This program runs a set "
			"of experiments and exports the SQNR and ADC effort of each run "
			"to a JSON file. See also `runAllExperiments`.",
			epilog="Written by joschua.conrad@uni-ulm.de at Ulm University, "
			"Institute of Microelectronics. Please also see the documentation "
			"for advanced usage, acknowledgement and license.",
	)
	
	thehelp="""Names of runs to conduct as found in results file.
If not passed, a full set is run. Needed SQNR references are always added.
Invalid rundesscriptions are ignored."""
	parser.add_argument(
			"runkeys",
			action="store",
			type=str,
			nargs="*",
			help=thehelp,
	)
	
	thehelp="""Number of processes to run experiments in parallel. Pass *0*
(the default) or a negative value to determine automatically.
Pass *1* to skipp all worker and multiprocessing and run experiment in the
same process (nice for debugging)."""
	parser.add_argument(
			"-j", "--jobs",
			action="store",
			type=int,
			help=thehelp,
	)
	
	thehelp="""If set, keep more jobs busy at the cost of larger memory foorptint."""
	parser.add_argument(
			"-a", "--aggressive",
			action="store_true",
			help=thehelp,
	)
	
	thehelp="""Where in the set of all experiments to start. If you pass 0.1,
the first tenth of all experiments will be discarded. The default includes
all runs. If *runkeys* is passed, this is ignored."""
	parser.add_argument(
			"-b", "--begin",
			action="store",
			type=float,
			help=thehelp,
	)
	
	thehelp="""Where in the set of all experiments to end. If you pass 0.1,
the last tenth of all experiments will be discarded. The default includes
all runs. If *runkeys* is passed, this is ignored.
Usual indexing logic is applied: if you pass *1.0*, the very last run
will be ignored."""
	parser.add_argument(
			"-e", "--end",
			action="store",
			type=float,
			help=thehelp,
	)
	
	thehelp="""Name of the JSON file to write to. If that file already exists,
it is open for read and already existing results are not re-created.
A pattern like *psumsimresults_{begin}_{end}.json* (the default) is allowed
and includes begin and end in the filename."""
	parser.add_argument(
			"-f", "--resultsfile",
			action="store",
			type=str,
			default="psumsimresults_{begin}_{end}.json",
			help=thehelp,
	)
	
	thehelp="""Name of a file to write progressbar to. Handy for getting the
progressbar to a file when running on a cluster. Can use patterns like
*filename*. If not given, the default stdout is kept. Information is appeneded
to an existing file. A good name is *psumsimoutput_{begin}_{end}.txt*.
This file will be flushed regularly. When given, the progressbar is not shown."""
	parser.add_argument(
			"-p", "--progressfile",
			action="store",
			type=str,
			default=None,
			help=thehelp,
	)
	
	thehelp="""If set, run a reduced set of experiments for debugging.
Is ignored if runnames are already given."""
	parser.add_argument(
			"-r", "--reduced",
			action="store_true",
			help=thehelp,
	)
	
	thehelp="""If set, run quiet. No progressbar in file or terminal."""
	parser.add_argument(
			"-q", "--quiet",
			action="store_true",
			help=thehelp,
	)
	
	thehelp="Display PSumSim version."
	parser.add_argument(
			"-v", "--version",
			action="version",
			version=__version__,
			help=thehelp,
	)
	
	return parser


def main(args=None):
	"""Calling this package as a script or using the entrypoint.
	
	This connects `getArgParser` and `runAllExperiments`.
	If the JSON file already exists, it is open for update to only run
	experiments whose results do not exist yet. The progress file is
	opened for appending.

	Parameters
	----------
	args : `None`, `list`, optional
		Commandline arguments to parse. Passed to
		`argparse.ArgumentParser.parse_args`. The default is `None`, which
		uses `sys.argv`.

	"""
	
	#Needed to make multiprocess with pyinstaller work
	#https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
	multiprocessing.freeze_support()
	
	parser = getArgParser()
	argvalues = parser.parse_args(args=args)
	
	#Replace pattern in JSON path
	jsonpath = argvalues.resultsfile
	try:
		jsonpath = jsonpath.format(begin=str(argvalues.begin), end=str(argvalues.end))
	except Exception:
		pass
	jsonpath = pathlib.Path(jsonpath)
	
	#If it already exists, open for reading, such that we only run experiments
	#which did not run yet.
	#The writing then in the end truncates automatically
	if jsonpath.exists():
		jsonfp = jsonpath.open("r+")
	else:
		jsonfp = jsonpath.open("w")
		
	#Open a stream for stdout after pattern replacement
	progressfile = argvalues.progressfile
	if progressfile is not None:
		try:
			progressfile = progressfile.format(begin=str(argvalues.begin), end=str(argvalues.end))
		except Exception:
			pass
		progressfile = pathlib.Path(progressfile)
		progressfilefp = progressfile.open("a")
	else:
		progressfilefp = None
		
	#The default is an empty list of runkeys. Set that to None, such that
	#everything is run if no key was given
	runkeys = argvalues.runkeys
	if runkeys == list():
		runkeys = None
	
	#Run stuff, but alwas close files
	try:
		runAllExperiments(
				 runreduced=argvalues.reduced,
				 runquiet=argvalues.quiet,
				 jsonfp=jsonfp,
				 iterbegin=argvalues.begin,
				 iterend=argvalues.end,
				 runkeys=runkeys,
				 processes=argvalues.jobs,
				 aggressive=argvalues.aggressive,
				 progressfp=progressfilefp,
		 )
	finally:
		jsonfp.close()
		if progressfilefp is not None:
			progressfilefp.close()

if __name__ == "__main__":
	
	#Run commandline program
	main()
