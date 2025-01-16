"""Tests for :py:mod:`psumsim`.

`pytest` is the used test framework. There is a package entry-point for running
tests from commandline."""

import itertools
import numpy as np
import enum
import copy
import json
import math
import pytest
import pickle
import scipy

from psumsim.array import normalizeAxes, getValueAlongAxis, padToEqualShape
from psumsim.hist import histlenToBincount, bincountToHistlen, packHist, packStatistic, HIST_AXIS, ACT_AXIS, WEIGHT_AXIS, MAC_AXIS, STAT_AXIS
from psumsim.simulate import simulateMvm, computeSqnr, optimumClippingCriterion, equalizeQuantizedUnquantized, generateSimulationOperands, applyCliplimitStddevAsFixedFrom, quantizeClipScaleValues
from psumsim.plot import plotHist
from psumsim.experiments import runAllExperiments, RunDescription
from .rand import sinusoidal

	
class BaseTestCase:
	"""Base class to inherit some useful methods."""

	@classmethod
	def concludeFigure(cls, fig):
		"""Close a created plot figure.
		
		Add some other routine here to show figures.

		Parameters
		----------
		fig : `matplotlib.figure.Figure`
			Figure to close.

		"""
		#Contextual import, because this one takes long
		from matplotlib import pyplot as plt
		plt.close(fig)
		
	@classmethod
	@pytest.fixture(scope="module")
	def tmp_numpy(cls, tmp_path_factory):
		"""Fixture to create temporary directory to store numpy arrays.

		The stored arrays were needed for creating plots for the PSumSim paper.
		We want this to exist on a per-module scope to have all in one plcae.
		Üer-class does not work, as this is a baseclass.
		
		Parameters
		----------
		tmp_path_factory : `pytest.TempPathFactory`
			Generic `pytest.fixture` used to create module-scoped temp dirs.

		Returns
		-------
		`pathlib.Path`
			Created temporal path.
		"""
		
		return tmp_path_factory.mktemp(basename="Numpy")
	
	@classmethod
	@pytest.fixture(scope="module")
	def tmp_json(cls, tmp_path_factory):
		"""Fixture to create temporary directory to store JSON.

		Like `tmp_numpy`. But these files are not needed after tests.
		
		Parameters
		----------
		tmp_path_factory : `pytest.TempPathFactory`
			Generic `pytest.fixture` used to create module-scoped temp dirs.

		Returns
		-------
		`pathlib.Path`
			Created temporal path.
		"""
		
		return tmp_path_factory.mktemp(basename="Json")
	
@enum.unique
class SHAPE_ENUM(enum.Enum):
	"""Defined names for numbers to later compute expected shapes from.
	
	Needed in `test_simulation`.
	"""
	
	MAC_HIST_ENUM = enum.auto()
	"""*histlen* along `MAC_AXIS`."""
	
	MAC_BINS_ENUM = enum.auto()
	"""*bincount* along `MAC_AXIS`."""
	
	STOCH_ENUM = enum.auto()
	"""*statisticdim* along `STAT_AXIS`."""
	
	ACT_BINS_ENUM = enum.auto()
	"""*bincount* along `ACT_AXIS`."""
	
	WEIGHT_BINS_ENUM = enum.auto()
	"""*bincount* along `WEIGHT_AXIS`."""
	
	ACT_HIST_ENUM = enum.auto()
	"""*histlen* along `ACT_AXIS`."""
	
	WEIGHT_HIST_ENUM = enum.auto()
	"""*histlen* along `WEIGHT_AXIS`."""
	
	ACT_BINS_IN_STOCH_ENUM = enum.auto()
	"""Like `ACT_BINS_ENUM`, but *1* in stochastic runs."""
	
	WEIGHT_BINS_IN_STOCH_ENUM = enum.auto()
	"""Like `WEIGHT_BINS_ENUM`, but *1* in stochastic runs."""
	
	MAC_IN_STAT_ENUM = enum.auto()
	"""Like `MAC_HIST_ENUM`, but *1* in statistic runs."""
	
	MAC_CHUNKED_ENUM = enum.auto()
	"""Exemplary chunkcount when chunking `MAC_AXIS` with *chunksize* 5."""
	
	HIST_TO_BINS_ENUM = enum.auto()
	"""Call `histlenToBincount`."""
	
	ROUND_ENUM = enum.auto()
	"""Call `round`."""
	
	CLIPLIMIT_MERGEVALUE_ENUM = enum.auto()
	"""Estimate shape change made by `quantizeClipScaleValues`."""
	
class test_simulation(BaseTestCase):
	"""Test `simulateMvm`.
	
	Includes `reduceSum`, `probabilisticAdder`, `quantizeClipScaleValues`
	and `getHistStddev`.
	
	Test cases are parametrized and results across test calls gathered as
	described demonstrated in `test_pytestFeatures`.
	
	"""

	RELATIVE_CONFIDENCE_RELAX = 10.
	"""`float` : Relax factor in assertion confidence for statistic results."""

	RELATIVE_SAMPLE_CONFIDENCE = 0.1
	"""`float` : Share of assertions allowed to fail in statistic assertion.
	
	Because we possibly compute
	quite small histograms and still want to allow an integer number of
	failed bins, we make this criterion weak and in return the relative
	relax more strong.
	"""
	
	AVERAGEOVER = 1000
	"""`int` : *statisticdim* for running statistic `simulateMvm`."""
	
	ABSOLUTE_GROUP_COMPARE_TOLERANCE = 1e-6
	"""`float` : Absolute tolerance for asserting stochastic numbers."""
	
	HIST_SUM_TOLERANCE = 1e-6
	"""`float` : Absolute tolerance for asserting sum over histogram
	being *1.0*."""
		
	#Define which MAC operatiosn to test. Also give expected shapes after
	#each test in the form of names values to multiply to retrieve
	#a value for each dimension. Also give histogram axes present. These
	#have length 1 in uint computation.
	#Only tests which in the end have a single value and which did not
	#merge values or use cliplimitstddev/cliplimitfixed except in last step get compared to a simple, big sum.
	#In this step, either mergevalues must be None/1, or (cliplimitstddev must be None
	#and cliplimitfixed None/1).
	#Prefer to generate results with just a stat and a hist axis,
	#and with only a single usage of mergevalues in last computation
	#step.
	#because these can be compared to simple mult/add done in this
	#test routine.
	#A
	GROUPS = (
		#Sum only over the number of MACs to do, not resolving bit
		#positions.
		#Try giving reduceaxes as a scalar.
		(
			(
				dict(	
						reduceaxes=MAC_AXIS,
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#This operation transfers the MAC axis to the histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				#Store as a result, where the MAC axis has been reduced
				#without any quantization or scaling.
				"macsreducedideal",
				tuple(),
			),
		),
		#Same, but without using binomial distribution to find histograms
		#from uncorrelated reduceaxis elems with shortest hist axis
		(
			(
				dict(	
						reduceaxes=MAC_AXIS,
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=False,
				),
				#This operation transfers the MAC axis to the histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				None,
				#Shall be same as before, even without optim.
				("macsreducedideal",),
			),
		),
		#Leave reductions empty and do nothing
		(
			(
				dict(	
						reduceaxes=tuple(),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Histogram exmpty still has 2 entries being value 0 and 1.
				#Otherwise the sum over values cannot be made 1.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(2,),
				),
				(HIST_AXIS,),
				#Memorize for other cases where nothing is changed
				"nothingreducedideal",
				tuple(),
			),
		),
		#First replace the nummacs axis by a hist axis, then also merge
		#activation and weight hist position.
		#Try digital and analog merge effort.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel="digital",
						allowoptim=True,
				),
				#As before
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				None,
				#THe MAC reduction worked just as above
				("macsreducedideal",),
			),
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="analog",
						allowoptim=True,
				),
				#Be left with a single histogram axis, which holds the
				#full value range. First compute length in histlen (so
				#excluding 0 and negative numbers, as one can mult there
				#simply), then use special enum to compute bincount from
				#that. This one is a callabel, so we pass it together
				#with arguments (which other callables use).
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				#Store ideal reduction of all values for comparison
				"allreducedideal",
				tuple(),
			),
		),
		#Again reduce all values, but first reduce weights and
		#activations and swap which merge model is applied on which one.
		#This time, in each group of weight x activations elements, only
		#a single one can be set. So along reduce axis there is one histogram
		#with a single unzero value set. Set disablereducecarries to
		#normalize probabilities before computation.
		(
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=(True, True,),
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel="digital",
						allowoptim=True,
				),
				#Be left with a single histogram axis, which holds the
				#full value range
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM,),
						(
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				"weightactreducedideal",
				tuple(),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS+2,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="analog",
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								SHAPE_ENUM.MAC_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				#See comment below why this gets an own name
				"allreducedidealweightsactfirst",
				#Should be the same as reducing everything with MACs first
				#("allreducedideal",),
				#Well, we first found the value of e.g. the multiplications.
				#We then summed the multiplication result up.
				#That is the way one should go, because the single multiplication
				#results are uncorelated.
				#But in allreducedideal, we first added along the MAC axis
				#and get how often each multiplication result occurs.
				#And that includes a correlation, because if one result
				#occured already NUMMACS times, the other ones are in
				#practice, 0, which is here neglected.
				#Due to that difference, we cannot compare the results.
				tuple(),
			),
		),
		#Also reduce weights and activations in the beginning, but without
		#optimizing computations by knowing that only a single value along
		#one-hot axis can be set, which means one combines a limited
		#number of histograms.
		#This optimization in practice is never activated, because it
		#expects oldbincount==1, which we never have when adding hist
		#axes in reduceSum.
		#But there is also the optimization for knowing that only a single
		#digit in the running histogram is set when activation disablereducecarries
		#and we check that optimization here.
		(
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=(True, True,),
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel="digital",
						allowoptim=False,
				),
				#Be left with a single histogram axis, which holds the
				#full value range
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM,),
						(
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				("weightactreducedideal",),
			),
		),
		#Sum the MACs in chunks, then over its chunks and then over
		#all axes together.
		#Try giving chunksize and positionweights as tuples and
		#setting explicit values for positionweightsonehot and
		#chunkoffsetsteps. We need a single offset with value 0, as the
		#chunked MAC axis has posweights same and they do not need any
		#offset.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(4,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=(None,),
						positionweightsonehot=(None,),
						disablereducecarries=(None,),
						chunkoffsetsteps=(None,),
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#THe MAC axis is not removed, it is quartered and therefore
				#the histogram axis is only as long as a chunk
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM, 0.25,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								4,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(None,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("same",),
						positionweightsonehot=(False,),
						disablereducecarries=(False,),
						chunkoffsetsteps=(1,),
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Now be left with resolved MAC chunks
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				None,
				("macsreducedideal",),
			),
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=(True, True,),
						disablereducecarries=(False, False,),
						chunkoffsetsteps=(1, 1,),
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#As before, merge all signal into histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				("allreducedideal",),
			)
		),
		#Also chunk MAC axis, but into a fractional number of chunks.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(5,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_CHUNKED_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Now be left with resolved MAC chunks. HIst entries unsued
				#due to residual chunks have been removed automatically.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								SHAPE_ENUM.MAC_BINS_ENUM,
						),
				),
				(HIST_AXIS,),
				None,
				("macsreducedideal",),
			),
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#As before, merge all signal into histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				("allreducedideal",),
			)
		),
		#Sum over bits with chunks. Test that later the chunks can be
		#merged with bit-ike weighting. Also swap act and weight in axes
		#here.
		(
			#First sum over MAC axis as usual
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				None,
				#This was the regualr MAC reduction
				("macsreducedideal",),
			),
			#Sum over bit axes, but leave bits chunked. Treat act axis here
			#first. We need the chunkoffsetsteps to ensure that an upper
			#chunk adds more value to the hist axis. So when later combining
			#the axes, we do not have to re-add that information.
			(
				dict(
						reduceaxes=(ACT_AXIS, WEIGHT_AXIS),
						chunksizes=(5, 7,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=(5, 7,),
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM, (1./5.),),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM, (1./7.),),
						#THe offset steps cause us to get a hist axis as
						#if there would be no chunking at all.
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			#Merge bit chunks. We used the offset steps to make the chunks
			#remember their position within eight, so we can use same
			#weighting. Still, we need to set the onehot flag, because
			#the sum of how often weight values can occur is limited.
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("same", "same",),
						positionweightsonehot=(True, True,),
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Be left with a single histogram axis, which holds the
				#full value range
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Even though we chunked the act/weight axes, the result
				#still should be the ideal correct one now after
				#merging chunks.
				("allreducedideal",),
			),
		),
		#Test chunks of size 1, which should be allowed and should change
		#nothing except adding empty hist axis
		(
			(
				dict(	
						reduceaxes=MAC_AXIS,
						chunksizes=(1,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#This operation transfers the MAC axis to the histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								1,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Compare to other run which did nothing
				("nothingreducedideal",),
			),
		),
		#Also test huge chunks, whose size should be limited to treating
		#as if there would be no chunking meaning everything is added
		#as one. But the chunk axis is still kept.
		(
			(
				dict(	
						reduceaxes=MAC_AXIS,
						chunksizes=(int(1e9),),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#This operation transfers the MAC axis to the histogram
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(1,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
				),
				(HIST_AXIS,),
				None,
				#Same as ideal MAC addition
				("macsreducedideal",),
			),
		),
		#Merge first the axis over macs and then over activation and weight
		#bits, but this time remove some bits.
		#Try digital and analog merge models with the mergevalues in effect.
		#mergevalues work on histlen, not bincount.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=2,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel="digital",
						allowoptim=True,
				),
				#Mergevalues shrinks the histogram axis
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								0.5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#No way to compare numbers, because this operation does change
				#values.
				tuple(),
			),
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=2,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="analog",
						allowoptim=True,
				),
				#The histogram axis is shrinked even more.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								0.5,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								0.5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
		),
		#Similar, but remove bits only along last axis. That way, we can
		#compare to uint computations whether the right values were removed.
		#We also test here to reduce all axes in one call to reduceSum.
		#The one reduction gives us larger histograms allowing
		#us to merge more values.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=4,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								0.25,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
		),
		#Merge single values into new bins, meaning nothing should change.
		#Merge everything into one hist axis, because then results
		#are compared to one big sum computed here.
		#Note that extrememly large values for mergevalues raise an
		#exception and are not tested here.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=1,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Merging single value should change nothing compared to
				#ideally redcuing all values
				("allreducedideal",),
			),
		),
		#We also have negative values for mergevalues, which then say
		#to keep only that number of histlen.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=-5,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Cannot compare to anything, as we did not use that
				#number of bins alraedy.
				tuple(),
			),
		),
		#We also have negative values for mergevalues, which are very
		#large and result in merging no bin and keeping everything
		#as-is. Expected shape and comaprison to allreducedideal
		#just like when setting mergevalues=1.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=-int(1e9),
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Just like when setting mergevlaues=1
				("allreducedideal",),
			),
		),
		#Try cliplimitstddev.
		#TRy float mergevalue
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=1.,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.CLIPLIMIT_MERGEVALUE_ENUM, (0, 1.)),
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Nothing to compare to, as scaling changes values
				tuple(),
			),
		),
		#Try extremely large cliplimitstddev, which should be limited to change
		#nothing.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=1e9,
						cliplimitfixed=None,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Expect that this changes nothing
				("allreducedideal",),
			),
		),
		#Use cliplimitstddev, but after using residual chunks. This checks that
		#maxhistvalue is correcty used. Also combine with using
		#mergevalues here.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(5,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_CHUNKED_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=2.,
						cliplimitstddev=1.,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Now be left with resolved MAC chunks. Possibly unused hist
				#entries due to residual chunks have vanished here.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								(SHAPE_ENUM.CLIPLIMIT_MERGEVALUE_ENUM, (1, 2.)),
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
		),
		#Test cliplimitfixed similar to cliplimitstddev. cliplimitstddev also just computes
		#a cliplimitfixed. We know give a known one.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=3.,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								0.333333333333333333333,
								(SHAPE_ENUM.ROUND_ENUM, tuple(),),
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Nothing to compare to, as scaling changes values
				tuple(),
			),
		),
		#Cliplimitfixed which has no effect including comparing to run without
		#any effect.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=1.,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Nothing to compare to, as scaling changes values
				("allreducedideal",),
			),
		),
		#Similar for very small cliplimitstddev, which is set to 1
		(
			(
				dict(
						reduceaxes=(MAC_AXIS, WEIGHT_AXIS, ACT_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=1e-9,
						positionweights=(None, "hist", "hist"),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#Nothing to compare to, as scaling changes values
				("allreducedideal",),
			),
		),
		#Very large values are limited as well and to maxhistvalue. We check
		#that here wiht residual chunks, where maxhistvalue is most complex.
		#Also add mergevalues here.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(5,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_CHUNKED_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								5,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=None,
						mergevalues=2.,
						cliplimitstddev=None,
						cliplimitfixed=1e9,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Now be left with resolved MAC chunks. Possibly unused hist
				#entries due to residual chunks have vanished here.
				#But the huge cliplimitfixed is limited to reduce everything into
				#one magnitude bin.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
								1,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
		),
		#Merge MACs, merge bits searately and then combine the two
		#histogram axes. When merging the hist axes for weights and
		#activations first, set disablereducecarries flags.
		(
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=(True, True,),
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Add a new hist axis with bit contents
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_HIST_ENUM,),
						(
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				#There is already a case, where we first merged weights
				#and activations
				("weightactreducedideal",),
			),
			(
				dict(
						reduceaxes=(MAC_AXIS+1,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS-1,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#Merge MACs as before on a first hist axis
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_BINS_ENUM,),
						(
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				#There is now one more histogram axis. If there are multiple
				#ones, we have to specify the one-hot one. For each
				#multiplication result, there is only one occurence count.
				#That is the one-hot one.
				(HIST_AXIS-1, HIST_AXIS,),
				None,
				#NOthing to compare to due to second hist axis
				tuple(),
			),
			(
				dict(
						reduceaxes=(HIST_AXIS,),
						chunksizes=None,
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						#Make sure to now merge into the axis which was
						#the one-hot before.
						histaxis=HIST_AXIS-1,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				#And merge the two hist axes into a single one.
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
								SHAPE_ENUM.MAC_HIST_ENUM,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				#No matter what happens, we now have only a single
				#hist axis left.
				(HIST_AXIS,),
				None,
				#Merging all hist axes into a single result still
				#should give a good result
				#Merging single value should change nothing compared to
				#ideally redcuing all values.
				#("allreducedideal",),
				#But we merge the weights and activations first. See
				#this group for a reason why that is different to
				#allreaducedideal.
				#("allreducedidealweightsactfirst",),
				#But we here reduce the mac hist axis onto weight/act
				#hist axis. allreducedidealweightsactfirst did that
				#vice-versa and added the correlations in different
				#order.
				tuple(),
			),
		),
		#A full run, where we first gather in the analog domain all
		#weight/act positions and chunked MACs. These results go thru an
		#ADC. The ADC words of different chunks are then combined in
		#the digital domain.
		(
			(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(5,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_CHUNKED_ENUM,),
						(SHAPE_ENUM.ACT_BINS_ENUM,),
						(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
						(
							5,
							(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						mergevalues=2,
						cliplimitstddev=1.,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="analog",
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(SHAPE_ENUM.MAC_CHUNKED_ENUM,),
						(
								5,
								SHAPE_ENUM.ACT_HIST_ENUM,
								SHAPE_ENUM.WEIGHT_HIST_ENUM,
								(SHAPE_ENUM.CLIPLIMIT_MERGEVALUE_ENUM, (1, 2.),),
								(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
			(
				dict(
						reduceaxes=(-2,),
						chunksizes=None,
						mergevalues=-2,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="digital",
						allowoptim=True,
				),
				(
						(SHAPE_ENUM.STOCH_ENUM,),
						(
							2,
							(SHAPE_ENUM.HIST_TO_BINS_ENUM, tuple(),),
						),
				),
				(HIST_AXIS,),
				None,
				tuple(),
			),
		),
	)
	"""`tuple` : Specification of calls made to `simulateMvm`.
	
	Lists for each call and for each simulation step:
		
	- The *groups* argument
	
	- Expected shape specified using product of `SHAPE_ENUM`
	
	- The histogram axis or even axes
	
	- Name under which to store results
	
	- Names of results to compare to
	
	"""
	
	#Expected shapes for all fields in return value of simulation
	#Remember expected shape for
	#each opernd or retrieve from groups.
	#Also remember just like for groups which hist axes we set to
	#expected length 1.
	#Also rmember the dtype with which a run-determined one was
	#possibly overwritten.
	#Also remember which fields have a dummy stat axis
	#Also remember which fields have a dummy hist axis forcibly.
	#Also remmber which fields have a dummy hist axis, if the run
	#supports it, because it uses int numbers instead of hist as a result.
	CHECKEDFIELDS = (
			#Results get their assumed shape from GROUPS
			("results", None, None, None, False, False, True,),
			#Multiplied bit-by-bit results. All combinations of
			#act/weight bit position are present.
			(
					"multiplied",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					None,
					False,
					False,
					False,
			),
			#The maximum hist value. Just like multiplied, but the
			#statistics axis is length 1. No need to set hist axis length
			#1, because we do not have a hist axis here.
			(
					"firstmaxhistvalue",
					(
							(
									(1,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					"int",
					True,
					False,
					False,
			),
			#Maxhistvalue checked very similar to the result
			(
					"maxhistvalues",
					None,
					None,
					#This is always int. Even when doing stochastic
					##computation.
					"int",
					#Dummy stat axis
					True,
					#Dummy hist axis in all runs. Even when results would
					#have a hist axis.
					True,
					#Thi switch has no effect then, as it sets dummy hist axis
					#in only some runs.
					False,
			),
			#Activations as bits
			(
					"activations",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.ACT_BINS_ENUM,),
							),
					),
					#The bins are a histogram axis, where the sum over
					#bins should be 1
					((-1,),),
					None,
					False,
					False,
					False,
			),
			#Weights as bits
			(
					"weights",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					((-1,),),
					None,
					False,
					False,
					False,
			),
			#Activations as uint, where the bit axis is 1.
			#In stochastic, the MAC axis is not filled with MACs,
			#but with values of bins along which the acitvation
			#histogram is sampled.
			(
					"activationsint",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(
											SHAPE_ENUM.MAC_IN_STAT_ENUM,
											SHAPE_ENUM.ACT_BINS_IN_STOCH_ENUM,
									),
									(1,),
							),
					),
					(tuple(),),
					"int",
					False,
					False,
					False,
			),
			#Same for weights
			(
					"weightsint",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(
											SHAPE_ENUM.MAC_IN_STAT_ENUM,
											SHAPE_ENUM.WEIGHT_BINS_IN_STOCH_ENUM,
									),
									(1,),
							),
					),
					(tuple(),),
					"int",
					False,
					False,
					False,
			),
			#Index matrix, which looks up all activation bit
			#positions broadcastable to multiplied
			(
					"activationhistidx",
					(
							(
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					"uint",
					False,
					False,
					False,
			),
			#Same for weights
			(
					"weighthistidx",
					(
							(
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					"uint",
					False,
					False,
					False,
			),
			#All activation bits looked up from activations using
			#activationhistidx. These are now bit values ready to
			#be multiplied bit by bit.
			(
					"activationformult",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					None,
					False,
					False,
					False,
			),
			#Same for weights
			(
					"weightformult",
					(
							(
									(SHAPE_ENUM.STOCH_ENUM,),
									(SHAPE_ENUM.MAC_HIST_ENUM,),
									(SHAPE_ENUM.ACT_BINS_ENUM,),
									(SHAPE_ENUM.WEIGHT_BINS_ENUM,),
							),
					),
					(tuple(),),
					None,
					False,
					False,
					False,
			),
	)
	"""`tuple` : Expected dtypes and shapes of results returned by
	`simulateMvm`."""
	
	#Which weight and activation levels to check.
	#DO not check excessively large levels, because these are slow
	#in sttatistic dummy implemetnation
	LEVELS = (
			dict(
					activationlevels=1,
					weightlevels=1,
			),
			dict(
					activationlevels=7,
					weightlevels=3,
			),
	)
	"""`tuple` : Specification of simulated activation and weight levels.
	
	Combine different values and also test very small counts, which makes
	it easier to track errors down."""
	
	#Number of macs matching chunksize or not
	NUMMACS = (
			16,
			8,
	)
	"""`tuple` Specification of tested *nummacs*."""
	
	#If set, groups are not compared against each other
	SKIP_GROUP_COMPARISON = False
	"""`bool` : If set, skip comparison of results across runs."""
	
	#Activating for debugging jut a single case
	#Deactivate group comparison, because groups to compare to might not
	#have been defiend yet
	#GROUPS = GROUPS[-2:-1]
	#LEVELS = LEVELS[1:2]
	#NUMMACS = NUMMACS[1:2]
	#SKIP_GROUP_COMPARISON = True
		
	@classmethod
	def statStocCompare(cls, stat, stoc, stataxis, msg=None):
		r"""Assert statistic and stochastic result equality.

		See `statstoc`.
		
		An assertion fails, if for *N* statistic results a share of
		`RELATIVE_SAMPLE_CONFIDENCE` results has a deviation larger than
		:math:`\frac{1}{\sqrt N}` times `RELATIVE_CONFIDENCE_RELAX`.
		So roughly a more exact assertion is made for more statistical results.
		
		Parameters
		----------
		stat : `numpy.ndarray`
			Result from *dostatisticdummy*.
			
		stoc : `numpy.ndarray`
			Result from *dostochastic*.
			
		stataxis : `int`
			The statistic axis. Often, `STAT_AXIS` is used. Even in
			*dostochastic*, this is needed and should refer to a length-1
			dimension.
			
		msg : `str`, `None`, optional
			Message for failed assertion. Default is `None`.

		"""
		
		stataxis, = normalizeAxes(axes=stataxis, referencendim=stat.ndim)
		#Find over how many elements we average, as that influences assertion
		#precision
		averageover = stat.shape[stataxis]
		#Average the bits along statistics axis to get probabilities
		statavg = packStatistic(
				topack=stat,
				axis=stataxis,
				keepdims=True,
		)
		#Compute absolute deviation in probability
		deviation = np.abs(statavg - stoc)		
		#That probability shall not be too large and we expect
		#a more accurate number the more experimetns we run.
		failpos = deviation > (1. / math.sqrt(averageover) * cls.RELATIVE_CONFIDENCE_RELAX)
		#Get the ratio of failed experiments
		failrate = np.average(failpos, axis=None, keepdims=False)
		#And limit that
		assert failrate <= cls.RELATIVE_SAMPLE_CONFIDENCE, msg
		
	@classmethod
	@pytest.fixture(scope="class")
	def simulationState(cls):
		"""Generated and assert results across simulation results.
		
		This is like `test_pytestFeatures.sumassertion`. Tests fill this across
		many tests and once the *class* scope of this fixture ends, assertions
		are made.
		
		Yields
		------
		resultsstock : `dict`
			Results are stored under names here to later assert their similarity,
			meaninging that some `GROUPS` are different, but shall yield same
			result.
			
		comparisons : `list`
			These are the compare assertions memorized by tests."""
		
		#Store results for comparison across groups here. The data levels are:
		#levels as str
		#nummacs as str
		#simulation name (e.g. retstochastic)
		#Name of a group (e.g. allreducedideal)
		#fieldname (e.g. results)
		resultsstock = dict(
		)
		comparisons = list()
		
		yield resultsstock, comparisons
		
		for referencekys, runresult, histaxes, comparegroupname in comparisons:
			#Use referencekeys to access the result
			reference = resultsstock
			for k in referencekys:
				reference = reference[k]
			
			#If dimension count mismatch, a reshape is
			#attempted
			if (runresult is None) or (reference is None) or (reference.ndim == runresult.ndim):
				reshapethis = None
				reshapeto = None
				reshaped = reference
				unreshaped = runresult
			elif reference.ndim  < runresult.ndim:
				reshapethis = reference
				reshapeto = runresult.shape
				unreshaped = runresult
			elif reference.ndim > runresult.ndim:
				reshapethis = runresult
				reshapeto = reference.shape
				unreshaped = reference
				
			#After finding who needs a reshape, do
			#it. Ignore if it does not work and continue.
			if reshapethis is not None:
				try:
					reshaped = np.reshape(
							reshapethis,
							reshapeto,
					)
				except ValueError:
					reshaped = reshapethis
					
			#A smaller array can be padded to a larger
			#one. This is needed, if residual chunks
			#have been in use.
			if (reshaped is not None) and (unreshaped is not None):
				padded, unpadded = padToEqualShape(
						a=reshaped,
						b=unreshaped,
						excludeaxes=None,
						padsymmetricaxes=histaxes,
				)
			else:
				padded = reshaped
				unpadded = unreshaped
			
			#For comparing float, we check absolute difference
			if (unpadded is not None) and np.issubdtype(unpadded.dtype, np.floating):
				error = np.abs(unpadded - padded)
			#But for integers, the numbers are exactly
			#the same and pass the test, or they differ
			#by one or more, the error is 1 and that 
			#is larger than absolute tolerance.
			else:
				error = unpadded != padded
			assert np.all(error < cls.ABSOLUTE_GROUP_COMPARE_TOLERANCE), \
					f"Result not the same as reference. Compared to " \
					f"{comparegroupname} from {referencekys}."
		
	@classmethod
	def pytest_generate_tests(cls, metafunc):
		"""Parameterize testcases.
		
		This uses `itertools.product` to combine `GROUPS`, `LEVELS` and
		`NUMMACS`. Works like `test_pytestFeatures.pytest_generate_tests` and
		creates a *simulationCase* `pytest.fixture`.

		Parameters
		----------
		metafunc : `pytest.Metafunc`
			If a *simulationCase* fixture request is found here, it is
			parametrized.

		"""
		
		if "simulationCase" in metafunc.fixturenames:
			cases = itertools.product(cls.GROUPS, cls.LEVELS, cls.NUMMACS)
			cases, ids = itertools.tee(cases)
			ids = (str(dict(group=i, levels=j, nummacs=k)) for i, j, k in ids)
			metafunc.parametrize("simulationCase", cases, ids=ids, scope="class")
		
	@classmethod
	@pytest.fixture(scope="class")
	def simulationResult(cls, simulationCase):
		"""Make the actual `simulateMvm` calls for one specific combination.

		Similar to `test_pytestFeatures.caseprocessed`.

		Parameters
		----------
		simulationCase : `tuple`
			One element from `GROUPS`, `LEVELS` and `NUMMACS` respectively.
			This `pytest.fixture` is parametrized by `pytest_generate_tests`.

		Returns
		-------
		:
			All information from *simulationCase* as well as results from
			*dostochastic*, *dostatistic*, *dostatisticdummy* as well as
			full-scale results.

		"""
		
		groupsshapeshistaxesgroupnames, levels, nummacs = simulationCase
		groups = tuple((i[0] for i in groupsshapeshistaxesgroupnames))
		shapesfromgroups = tuple((i[1] for i in groupsshapeshistaxesgroupnames))
		histaxesfromgroups = tuple((i[2] for i in groupsshapeshistaxesgroupnames))
		storegroupnamesfromgroups = tuple((i[3] for i in groupsshapeshistaxesgroupnames))
		comparegroupnamesfromgroups = tuple((i[4] for i in groupsshapeshistaxesgroupnames))
			
		simulationargs = dict(
				nummacs=nummacs,
				**levels,
				selfcheckindummy=True,
				randomclips=(2., 2.,),
		)
		defaultrandombehave = "uniform"
		
		#Simulate with probabilities
		retstochastic = simulateMvm(
			statisticdim=None,
			groups=groups,
			**simulationargs,
			randombehave=defaultrandombehave,
			dostatisticdummy=False,
		)
		#Simulate with bits, but many times.
		#Do addition by adding uint numbers
		retstatistic = simulateMvm(
			statisticdim=cls.AVERAGEOVER,
			groups=groups,
			**simulationargs,
			randombehave=defaultrandombehave,
			dostatisticdummy=False,
		)
		#Same, but addition is done with a bit-wise adder.
		retstatisticdummy = simulateMvm(
			statisticdim=cls.AVERAGEOVER,
			groups=groups,
			**simulationargs,
			randombehave=defaultrandombehave,
			dostatisticdummy=True,
		)
		
		#Get cliplimitfixeds from retstochastic, which is needed if cliplimitstddev
		#was introduced. Same for getting it from statistic. We will
		#need both to create corresponding fullscales.
		#statistic with and without dummy
		fullscalegroups = list()
		fullscalegroupcases = (retstochastic, retstatistic)
		for cliplimitfixedfrom in fullscalegroupcases:
			newfullscalegroups = applyCliplimitStddevAsFixedFrom(
					groups=groups,
					fromreturn=cliplimitfixedfrom,
					onlyatindices=None,
			)	
			fullscalegroups.append(newfullscalegroups)
		fullscalegroupsstoc, fullscalegroupsstat = fullscalegroups
		
		#Full scale computation with uint values to later check fullscale
		#estimated during computation. In full scale, all values are
		#the same, so a single sample over statistics is enough.
		#Do this for both cliplimitfixeds: from statistic and stochastic.
		retfullscalestoc = simulateMvm(
			statisticdim=1,
			groups=fullscalegroupsstoc,
			**simulationargs,
			randombehave="fullscale",
			dostatisticdummy=False,
		)
		
		retfullscalestat = simulateMvm(
			statisticdim=1,
			groups=fullscalegroupsstat,
			**simulationargs,
			randombehave="fullscale",
			dostatisticdummy=False,
		)
		
		return \
				groupsshapeshistaxesgroupnames, \
				levels, \
				nummacs, \
				groups, \
				shapesfromgroups, \
				histaxesfromgroups, \
				storegroupnamesfromgroups, \
				comparegroupnamesfromgroups, \
				retstochastic, \
				retstatistic, \
				retstatisticdummy, \
				retfullscalestoc, \
				retfullscalestat,
		
	@classmethod
	def test_shapesTypes(cls, simulationResult, subtests, simulationState):
		"""Assert shapes and dtypes of result fields.
		
		`CHECKEDFIELDS` defines what to check.
		
		This also fills `simulationState` with results and comparisons to
		run.

		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		simulationState : `tuple`
			Filled from `pytest.fixture` `simulationState`. Results and which
			results to compare against each other are memorized here.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		resultsstock, comparisons = simulationState
		
		#Now assert expected operand shapes. Assert for all run
		#simulation kinds. Also remember the effective length of statistics
		#axis, which is 1 in stochastic simulation. And remember which
		#simulations work on uint and have no histogram axis.
		#And remember which ones are stochastic and have a value axis in
		#int operands.
		#And remember the dtype mostly used.
		checkedruns = (
				(retstochastic, "retstochastic", 1, False, True, "float"),
				(retstatistic, "retstatistic", cls.AVERAGEOVER, True, False, "int"),
				(retstatisticdummy, "retstatisticdummy", cls.AVERAGEOVER, False, False, "bool"),
				(retfullscalestoc, "retfullscalestoc", 1, True, False, "int"),
				(retfullscalestat, "retfullscalestat", 1, True, False, "int"),
		)

		checkedrunsfields = itertools.product(checkedruns, cls.CHECKEDFIELDS)
		for therun, thefield in checkedrunsfields:
			runobj, runname, statlen, dummyhistaxrun, isstochastic, rundtype = therun
			fieldname, expectedshapes, histaxes, fielddtype, dummystatax, dummyhistaxforce, dummyhistaxfield = thefield
			#FInd expected dtype from run and possibly overwrite with something
			#operand specific
			expecteddtype = rundtype
			if fielddtype is not None:
				expecteddtype = fielddtype
			#The hist axis is only a dummy, if the field and the run
			#both know that. A uint run has dummy hist ax in results,
			#but operands still have a hist axis to weight along.
			#But fields can force to always have a dummy hist ax.
			dummyhistax = (dummyhistaxrun and dummyhistaxfield) or dummyhistaxforce
			#Prepare dict, where shape symbols are related to actual
			#numbers, which depend on the exact test case.
			#The fill other dict entries
			actbincount = histlenToBincount(histlen=levels["activationlevels"])
			weightbincount = histlenToBincount(histlen=levels["weightlevels"])
			nummacchunks = math.ceil(nummacs / 5)
			shapelookup = {
					#Length of statistics axis, which is 1 in stochastic
					#or for some special fields
					SHAPE_ENUM.STOCH_ENUM : ((dummystatax and 1) or statlen),
					#Number of MACs we sum over
					SHAPE_ENUM.MAC_HIST_ENUM : nummacs,
					#Bins the maccounts create in a hitogram
					SHAPE_ENUM.MAC_BINS_ENUM : histlenToBincount(histlen=nummacs),
					#Maximum value activation/weight can have
					SHAPE_ENUM.ACT_HIST_ENUM : (levels["activationlevels"]),
					SHAPE_ENUM.WEIGHT_HIST_ENUM : (levels["weightlevels"]),
					#Histogram lengths
					SHAPE_ENUM.ACT_BINS_ENUM : actbincount,
					SHAPE_ENUM.WEIGHT_BINS_ENUM : weightbincount,
					#Number of steps activation/weight can have.
					#But only in stochastics. Otherwise, this is 1.
					SHAPE_ENUM.ACT_BINS_IN_STOCH_ENUM : ((isstochastic and actbincount) or 1), 
					SHAPE_ENUM.WEIGHT_BINS_IN_STOCH_ENUM : ((isstochastic and weightbincount) or 1), 
					#Number of macs again, but only in non-stochastics.
					#Otherwise this is 1.
					SHAPE_ENUM.MAC_IN_STAT_ENUM : (((not isstochastic) and nummacs) or 1),
					#The chunk count yielded from creating fractional
					#number of chunks along MAC aixs in experiment
					SHAPE_ENUM.MAC_CHUNKED_ENUM : nummacchunks,
					#Turns a hist to bin counts. Special enum
					SHAPE_ENUM.HIST_TO_BINS_ENUM : lambda e: histlenToBincount(histlen=e),
					#Rounds a number
					SHAPE_ENUM.ROUND_ENUM : lambda e: round(e),
					#Function, which scales shape according to cliplimitfixed
					#from runidx and mergevalues. Pass current expected,
					#valuescalidx and mergevalues
					SHAPE_ENUM.CLIPLIMIT_MERGEVALUE_ENUM : lambda e, vsidx, mv : round(round(e / runobj["cliplimitfixeds"][vsidx]) / max(mv / runobj["cliplimitfixeds"][vsidx], 1)),
			}
			#The value only makes it to stock and is compared between
			#groups, if it gives a shape for each group.
			#Otherwise it is not a value being suitable to be compareed
			#on a per-group basis.
			fieldsupportscomparison = (expectedshapes is None) and (not cls.SKIP_GROUP_COMPARISON)
			#Expected shape can depends on MAC groups instead of the 
			#operand. Same for hist axes to set to 1.
			if expectedshapes is None:
				expectedshapes = shapesfromgroups
			if histaxes is None:
				histaxes = histaxesfromgroups
			#Get actual ndarray
			runresults = runobj[fieldname]
			#Results are given as list of ndarray. Operands not. Give
			#it as a  common ground.
			if not isinstance(runresults, (tuple, list)):
				runresults = (runresults,)
			#If this field does not support group comparison,
			#reaplce the group names by dummies
			if fieldsupportscomparison:
				storegroupnames = storegroupnamesfromgroups
				comparegroupnames = comparegroupnamesfromgroups
			else:
				storegroupnames = itertools.repeat(None)
				comparegroupnames = itertools.repeat(tuple())
				
			#Iterate over ndarrays inside the result list/tuple
			for resultidx, runresultsexpectedshapeshistaxescomparegroupnames in \
					enumerate(zip(runresults, expectedshapes, histaxes, storegroupnames, comparegroupnames)):
				runresult, expectedshape, histaxes, storegroupname, runcomparegroupnames = \
						runresultsexpectedshapeshistaxescomparegroupnames
						
				#Store all results of a run in stock, if required
				if (storegroupname is not None) and fieldsupportscomparison:
					storetarget = resultsstock.setdefault(str(levels), dict())
					storetarget = storetarget.setdefault(str(nummacs), dict())
					storetarget = storetarget.setdefault(runname, dict())
					storetarget = storetarget.setdefault(storegroupname, dict())
					storetarget[fieldname] = runresult
						
				#Now fill the expected shape axis by axis
				expandedshape = list()
				for shapeentry in expectedshape:
					#STart with a value 1 and comoute in float domain to
					#be able do divide by 4.
					expandedentry = 1.
					for shapecomponent in shapeentry:
						#If shapecomponent is a tuple, we have callable and
						#args
						if isinstance(shapecomponent, tuple):
							callableinst, callableargs = shapecomponent
							callableinst = shapelookup[callableinst]
							expandedentry = callableinst(expandedentry, *callableargs)
							
						else:
							#Try to interpret as shape symbol
							shapecomponent = shapelookup.get(
									shapecomponent,
									shapecomponent,
							)

							expandedentry *= shapecomponent
					#Back to int domain. Round up, because the
					#general philosophy is to create more bins and chunks
					#than needed, if chunksccounts are residual.
					#This also works nice, as float precision usually
					#rounds down and ceil gets us to the correct value
					#then.
					#Use SHAPE_ENUM.ROUND_ENUM if you want something else.
					expandedentry = int(math.ceil(expandedentry))
					#Remember length of this axis
					expandedshape.append(expandedentry)
				#THe uint computation uses hist axes as dummy only
				if dummyhistax:
					for histax in histaxes:
						expandedshape[histax] = 1
				expandedshape = tuple(expandedshape)
				#If we have it, take first hist axis, sum over it and
				#assert sum later. As usual, take the first hist axis as
				#the effective one, which is used for more than 
				#setting length to 1.
				if (not dummyhistax) and (len(histaxes) > 0) and (runresult is not None):
					sumoverhist = np.sum(
							runresult,
							axis=histaxes[0],
							keepdims=True,
					)
				else:
					sumoverhist = None
				
				#Actual assertion
				with subtests.test(
						runname=runname,	
						fieldname=fieldname,
						resultidx=resultidx,
				):
					assert runresult.shape == \
							expandedshape, \
							"Not the expected array shape"
					assert runresult.dtype == \
							expecteddtype, \
							"Not the expected array dtype"
					#Assert sum over histogram if given
					if sumoverhist is not None:
						assert np.all(np.absolute(1 - sumoverhist) < cls.HIST_SUM_TOLERANCE), \
								"Sum over histogram not 1"
					#Compare results against earlier results from other
					#runs. Shapes are already asserted within each run
					#assertions, so compare values only.
					#Only memorize comparisons and do them in the end.
					if fieldsupportscomparison:
						for comparegroupname in runcomparegroupnames:
							comparisons.append((
									(str(levels), str(nummacs), runname, comparegroupname, fieldname),
									runresult,
									histaxes,
									comparegroupname,
							))
							
							
	@classmethod
	def test_resultsLength(cls, simulationResult, subtests):
		"""Assert length of *results* list returned by `simulateMvm`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#Assert correct number of results
		checkedruns = (
				(retstochastic, "retstochastic",),
				(retstatistic, "retstatistic",),
				(retstatisticdummy, "retstatisticdummy",),
				(retfullscalestoc, "retfullscalestoc",),
				(retfullscalestat, "retfullscalestat",),
		)
		for runobj, runname in checkedruns:
			with subtests.test(
					runname=runname,	
			):
				assert len(runobj["results"]) == \
						len(groupsshapeshistaxesgroupnames), \
						"There are not as many results as run " \
						"experiments."
						
						
	@classmethod
	def test_mergeeffort(cls, simulationResult, subtests):
		"""Assert *mergeeffort* result returned by `simulateMvm`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#Now assert the ADC Energy. Check in all runs and
		#remember where histlen is unknwon, such that we have to
		#assert to stochastic run, which itself is cehcked at first
		#and has a histlen. We remember which runresult we use to read
		#the histlen from, as that depends on cliplimitfixed and might be
		#different due to different histlen between stochastic
		#and statistic result.
		checkedruns = (
				(retstochastic, "retstochastic", None,),
				(retstatistic, "retstatistic", retstatisticdummy,),
				(retstatisticdummy, "retstatisticdummy", None,),
				(retfullscalestoc, "retfullscalestoc", retstochastic,),
				(retfullscalestat, "retfullscalestat", retstatisticdummy,),
		)
		for runobj, runname, refobj in checkedruns:
			with subtests.test(
					runname=runname,
					msg="There are not as many merge efforts as runresults.",
			):
				assert len(runobj["results"]) == \
						len(runobj["mergeefforts"])
				
			for runidx, mergeeffortresultgrouphistaxes in enumerate(zip(
					runobj["mergeefforts"],
					runobj["results"],
					runobj["maxhistvalues"],
					groups,
					histaxesfromgroups,
			)):
				runmergeeffort, runresult, maxhistvalue, group, histaxes = \
						mergeeffortresultgrouphistaxes
				mergeeffortmodel = group["mergeeffortmodel"]
				firsthistaxis = histaxes[0]
				maxhistvaluemin = np.min(maxhistvalue)
				maxhistvaluemax = np.max(maxhistvalue)
				#If maxhistvalue is all equal, the bincount used in ADC
				#model is the hist axis length. THe number of times
				#the ADC has been called is read from product over
				#other shape components.
				if np.all(maxhistvaluemin == maxhistvaluemax):
					bincount = runresult.shape[firsthistaxis]
					numberruns = np.array(runresult.shape)
					numberruns[STAT_AXIS] = 1
					numberruns[firsthistaxis] = 1
					numberruns = np.prod(numberruns)
				#Otherwise, we have to ask maxhistvalue for the bincount.
				#The number of runs will not be set, as each ADC run
				#might have needed a different effort and we have to
				#sum over these values later
				else:
					bincount = maxhistvalue
					bincount = histlenToBincount(histlen=bincount)
					numberruns = None
				#If we have a refobj, simply compare to that
				if refobj is not None:
					expectedmergeeffort = refobj["mergeefforts"][runidx]
				#If we have no merge model, assert none
				elif mergeeffortmodel is None:
					expectedmergeeffort = None
				#Otherwise, we have to compute some number
				else:
					if mergeeffortmodel == "analog":
						expectedmergeeffort = np.array(
								bincount,
								dtype="float",
								copy=True,
						)
					elif mergeeffortmodel == "digital":
						expectedmergeeffort = np.log2(bincount, dtype="float")
					else:
						expectedmergeeffort = None
					#Now apply the number of ADC runs. We either have
					#a factor because all the ADC runs needed same effort,
					#or we sum over efforts.
					if numberruns is None:
						expectedmergeeffort = np.sum(expectedmergeeffort)
					else:
						expectedmergeeffort = np.multiply(
								expectedmergeeffort,
								numberruns,
						)
					#Get python item
					expectedmergeeffort = expectedmergeeffort.item()


				with subtests.test(
						runname=runname,
						runidx=runidx,
				):
					assert type(runmergeeffort) == \
							type(expectedmergeeffort), \
							"Merge effort does not have expected type."
					if runmergeeffort is not None:
						assert runmergeeffort == pytest.approx(
										expectedmergeeffort,
										rel=1e-9,
										abs=0.,
								), \
								"Merge effort does not have expected value."
								
	@classmethod
	def test_maxHistValue(cls, simulationResult, subtests):
		"""Assert *maxhistvalues* result returned by `simulateMvm`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
								
		#Assert correct number of maxhistvalue
		checkedruns = (
				(retstochastic, "retstochastic",),
				(retstatistic, "retstatistic",),
				(retstatisticdummy, "retstatisticdummy",),
				(retfullscalestoc, "retfullscalestoc",),
				(retfullscalestat, "retfullscalestat",),
		)
		for runobj, runname in checkedruns:
			with subtests.test(
					runname=runname,
					msg="There are not as many maxhistvalues as run experiments."
			):
				assert len(runobj["maxhistvalues"]) == \
						len(groupsshapeshistaxesgroupnames)
						

		#Assert the maximum histogram values by taking stochastic result,
		#where the histogram shape has already been asserted, and by
		#getting full scale from there.
		#As soon as we have residual chunking or cliplimitstddev or a
		#cliplimitfixed, the simply computed
		#expectation is wrong, because the residual chunk masks some
		#histogram values or because the cliplimitstddev introduces a factor
		#we dont know. For such cases, we also look at the return
		#values of the simulation with fullscale operands.
		chunksizecliplimitstddevcliplimitfixedintroduced = False
		for runidx, resultobjhistaxgroup in enumerate(zip(retstochastic["results"], histaxesfromgroups, groups)):
			resultobj, histaxes, group = resultobjhistaxgroup
			expectedmaxhistvalue = np.ones(shape=resultobj[0:1].shape, dtype="int")
			for histax in histaxes:
				histlen = bincountToHistlen(bincount=expectedmaxhistvalue.shape[histax])
				expectedmaxhistvalue = getValueAlongAxis(
						value=expectedmaxhistvalue,
						start=0,
						stop=1,
						step=None,
						axis=histax,
				)
				np.multiply(expectedmaxhistvalue, histlen, out=expectedmaxhistvalue)
			chunksizecliplimitstddevcliplimitfixedintroduced = chunksizecliplimitstddevcliplimitfixedintroduced or (group["chunksizes"] is not None) or (group["cliplimitstddev"] is not None) or (group["cliplimitfixed"] is not None)
			#Now that the expected value is known, iterate over made
			#simulations and assert maxhistvalue in each one.
			#We here note for each run which fullscale run has the
			#result which should be equal to the maxhistvalue. There
			#are two full scale results derived from different cliplimitfixeds
			#copied from statistic or stochastic experiment.
			checkedruns = (
					(retstochastic, "retstochastic", retfullscalestoc,),
					(retstatistic, "retstatistic", retfullscalestat,),
					(retstatisticdummy, "retstatisticdummy", retfullscalestat,),
					(retfullscalestoc, "retfullscalestoc", retfullscalestoc,),
					(retfullscalestat, "retfullscalestat", retfullscalestat,),
			)
			for runobj, runname, refobj in checkedruns:
				maxhistvalue = runobj["maxhistvalues"][runidx]
				with subtests.test(
						runname=runname,
						runidx=runidx,	
				):
					#Assert on hand-crafted expected value
					if not chunksizecliplimitstddevcliplimitfixedintroduced:
						assert maxhistvalue.shape == \
							expectedmaxhistvalue.shape, \
							"Shape of maxhistvalue not as predicted."
						assert np.all(maxhistvalue==expectedmaxhistvalue), \
							"Value of maxhistvalue not as predicted."
					#refobj is a fullscale simulation result
					refobj = refobj["results"][runidx]
					#Assert on value from full-scale simulation. Here we can
					#only assert the digits, where the full scale is set,
					#because the full scale is one-hot e.g. along weight and
					#activation axis and is there only set at highest value.
					#Yes, that does not test all digits, but if a group
					#reduces everything to a single digits, there is a thorough
					#check of whether the fullscale was correctly carried thru
					#all reduceSum calls.
					checkmask = refobj > 0
					assert maxhistvalue.shape == \
						refobj.shape, \
						"Shape of maxhistvalue not as in fullscale sim."
					assert np.all(maxhistvalue[checkmask]==refobj[checkmask]), \
						"Value of maxhistvalue not as in fullscale sim."
								
	@classmethod
	def test_clipLimitFixed(cls, simulationResult, subtests):
		"""Assert *cliplimitfixeds* result returned by `simulateMvm`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
						
		#Assert cliplimitfixed result field, which is special because it
		#could be None or float, but never numpy.
		checkedruns = (
				(retstochastic, "retstochastic",),
				(retstatistic, "retstatistic",),
				(retstatisticdummy, "retstatisticdummy",),
				(retfullscalestoc, "retfullscalestoc",),
				(retfullscalestat, "retfullscalestat",),
		)
		for runobj, runname in checkedruns:	
			cliplimitfixeds = runobj["cliplimitfixeds"]
			with subtests.test(
					runname=runname,
					msg="There is not one cliplimitfixed per run",
			):
				assert len(runobj["results"]) == \
						len(cliplimitfixeds)
			
			for resultidx, cliplimitfixedgrouphistax in enumerate(zip(cliplimitfixeds, groups, histaxesfromgroups)):
				cliplimitfixed, group, histaxes = cliplimitfixedgrouphistax
				cliplimitfixedarg = group["cliplimitfixed"]
				mergevaluesarg = group["mergevalues"]
				expectnone = (group["cliplimitstddev"] is None) and (cliplimitfixedarg is None)
				with subtests.test(
						runname=runname,
						resultidx=resultidx,
				):
					if expectnone:
						assert cliplimitfixed is \
								None, \
								"Cliplimitfixed should be None"
					else:
						#DO not use isinstance, because float np.ndarray
						#is True on that test, too
						assert type(cliplimitfixed) is \
								float, \
								"Cliplimitfixed should be float"
						#We know the cliplimitfixed exactly, if it was given
						#as arg and not computed from standard devaition.
						#FOr negative mergevalues, we cannot restore
						#how to compensate mergevalues to know the
						#histlen to limit cliplimitfixed.
						if (cliplimitfixedarg is not None) and ((mergevaluesarg is None) or (mergevaluesarg > 0)):
							histaxis = histaxes[0]
							#Get histlen from run, where it never is 1
							#due to int based computation as in statisticdummy
							bincount = retstochastic["results"][resultidx].shape[histaxis]
							histlen = bincountToHistlen(bincount=bincount)
							#Expected value from argument, but is actually
							#limited to not be below 1.
							expected = cliplimitfixedarg
							expected = max(expected, 1)
							#For very large cliplimitfixeds, which brings us
							#down to having only a histlen of 1, we
							#do not know to which value the value scale
							#should have been limited. Because for that,
							#we need to know the shape after reducing
							#all axes but before going into scaling and
							#merging of values.
							if histlen > 1:
								assert cliplimitfixed == \
										expected, \
										"Cliplimitfixed does not have " \
										"expected value from argument."
						
	@classmethod
	def test_mergeValues(cls, simulationResult, subtests):
		"""Assert *mergevaluess* result returned by `simulateMvm`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#Returned mergevalues are similar checked to cliplimitfixed
		checkedruns = (
				(retstochastic, "retstochastic",),
				(retstatistic, "retstatistic",),
				(retstatisticdummy, "retstatisticdummy",),
				(retfullscalestoc, "retfullscalestoc",),
				(retfullscalestat, "retfullscalestat",),
		)
		for runobj, runname in checkedruns:	
			mergevaluess = runobj["mergevaluess"]
			cliplimitfixeds = runobj["cliplimitfixeds"]
			with subtests.test(
					runname=runname,
					msg="There is not one mergevalues per run",
			):
				assert len(runobj["results"]) == \
						len(cliplimitfixeds)
						
			for resultidx, mergevaluescliplimitfixedsgroup in enumerate(zip(mergevaluess, cliplimitfixeds, groups)):
				mergevalues, cliplimitfixed, group = mergevaluescliplimitfixedsgroup
				mergevaluesarg = group["mergevalues"]
				expectnone = (mergevaluesarg is None)
				with subtests.test(
						runname=runname,
						resultidx=resultidx,
				):
					if expectnone:
						assert mergevalues is \
								None, \
								"Mergevalues should be None"
					else:
						#DO not use isinstance, because float np.ndarray
						#is True on that test, too
						assert type(mergevalues) is \
								float, \
								"Mergevalues should be float"
						#Now try to assert actual value. If we have a
						#negative mergevalues argument, we only know
						#that the resulting value still should be at
						#least 1. We do not know the shape before
						#applying mergevalues, which would tell us how
						#to translate the negative mergevalues (bins to keep)
						#to a positive one (bins to merge)
						#Otherwise, we still know nothing if cliplimitfixed
						#was returned, because that can partially replace
						#mergevalues.
						#That a negative mergevalue translates to keeping
						#that number of bins is regarded in shape checks.
						if (mergevaluesarg < 0) or ((cliplimitfixed is not None) and (cliplimitfixed != 1)):
							assert mergevalues >= \
									1, \
									"Returned mergevalues need to " \
									"be positive."
						#Otherwise, the returned value should be the
						#argument.
						else:
							assert mergevalues == \
									mergevaluesarg, \
									"Returned mergevalues should be " \
									"value from argument."
									
									
									
	@classmethod
	def test_results(cls, simulationResult, subtests):
		"""Assert *results* result returned by `simulateMvm`.
		
		If possible, other fields are used to re-compute the expected
		MVM result of statistic computations using `int` operations and no
		special histogram stuff.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#For comparison with own computation in this case, which
		#simply sums over all axes, we need to have only a final
		#non-None value in "mergevalues". 
		#We actually retrieve the applied mergevalue from a results field,
		#because we then even can work on negative mergevalues, where we
		#otherwise do not now to which number of bins tto merge they
		#translate.
		firstbitskept = all((i is None for i in retstatistic["mergevaluess"][:-1]))
		#By default, the effective, single mergevalue we recongize is
		#None. We only set it, if there really is only a single, final
		#mergevalue. If that one is None, we set it to 1 here, which
		#eases computations.
		effectivemergevalue = None
		if firstbitskept:
			effectivemergevalue = retstatistic["mergevaluess"][-1]
			#None is later treated as 1
			if effectivemergevalue is None:
				effectivemergevalue = 1
		#We also can handle a final cliplimitfixed on last call of reduceSum.
		#We do not care if this originates from cliplimitfixed or cliplimitstddev,
		#we can handle both. We just read the scale from return value.
		firstunscaled = all((i is None for i in retstatistic["cliplimitfixeds"][:-1]))
		effectivecliplimitfixed = None
		if firstunscaled:
			effectivecliplimitfixed = retstatistic["cliplimitfixeds"][-1]
			if effectivecliplimitfixed is None:
				effectivecliplimitfixed = 1
			
		#Also get maximum value needed for clipping when applying
		#cliplimitfixed. All results have that, but use some statistic one,
		#as statistic and stochastic have different stddev and
		#hence different cliplimitfixed
		maxhistvalue = retstatistic["maxhistvalues"][-1]
		
		#Compute expected int value and compare to statistical one.
		#Only works if statistical compute threw away bits only in the
		#last itertation and if cliplimitfixed was only used in a last
		#group.
		#Also skip this, if we tested a group not reducing all axes to
		#a final result.
		#We also cannot do this assertion, if mergevalues AND a
		#cliplimitfixed are both given, because we then do not know the maxhistvalue
		#which was used for clipping after applying the cliplimitfixed, because
		#we only have te rounded value after applying mergevalue
		if (effectivemergevalue is not None) and (effectivecliplimitfixed is not None) and (retstatistic["results"][-1].size == cls.AVERAGEOVER) and ((effectivemergevalue == 1) or (effectivecliplimitfixed == 1)):
			
			#So far, everything was fcussed around retstatistic. But
			#we also have a fullscale result which used the exact
			#same cliplimitfixed and hence also same mergevalues.
			#It must behave the same and we assert it similar.
			#The runs only differ in length of statistic axis, as
			#fullscale experiment has only a dummy stat axis.
			checkedruns = (
					(retstatistic, "retstatistic", cls.AVERAGEOVER,),
					(retfullscalestat, "retfullscalestat", 1,),
			)
			
			for runobj, runname, effectiveaverageover in checkedruns:
				with subtests.test(runname=runname):
					expected = runobj["activationsint"] * runobj["weightsint"]
					#We compute here with statistics axis only. Remove hist
					#ax also from maxhistvalue
					maxhistvalue = np.squeeze(maxhistvalue, axis=histaxesfromgroups[0])
					#Activations and weights have a dimension -1 for bit positions,
					#which is 1 as we work with int values, and dimension -2 for
					#the many pairs of weights/activations.
					expected = np.sum(expected, axis=(-2, -1), keepdims=False)
					#Apply the cliplimitfixed. This was done by clipping to some
					#max value. And we have that max value as an output, which
					#is elsewhere asserted against fullscale result, which
					#is asserted against some uint computation in here.
					if effectivecliplimitfixed != 1:
						np.clip(expected, a_min=-maxhistvalue, a_max=maxhistvalue, out=expected)
					#Apply the single used mergevalue.
					if effectivemergevalue != 1:
						#Effectivemergevalue is reduced by cliplimitfixed
						expected = np.divide(expected, max((effectivemergevalue / effectivecliplimitfixed), 1), dtype="float")
						np.round(expected, out=expected)
						expected = expected.astype(dtype="int")
					testedreshaped = np.reshape(
							runobj["results"][-1],
							(effectiveaverageover,),
					)
					assert np.all(testedreshaped == expected), \
							"Simple integer and bit-wise compute shall " \
									"yield same result"
									
									
	@classmethod
	def test_statisticDummy(cls, simulationResult, subtests):
		"""Assert that *dostatistic* and *dostatisticdummy* yield same result.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#Retrieve statistic integer results as uint, which is nice
		#to compare.
		#The statistic one is already uint. The statisticdummy
		#is a boolean in shape compatible to stochastic result and
		#we have to pack that histogram. We do not use HIST_AXIS, but
		#we ask the most recent created group for the histaxes it
		#contains and we merge all of them to one big int.
		statisticuintresult = retstatistic["results"][-1]
		statisticdummyintresult = retstatisticdummy["results"][-1]
		isfirsthistax = True
		for histax in histaxesfromgroups[-1]:
			statisticdummyintresult = packHist(
					topack=statisticdummyintresult,
					axis=histax,
					keepdims=True,
					#This is nonstrict, if there are multiple hist
					#axes. Imagine you reduced to two axes, one having
					#multiplication result values, the other one
					#how often they occured along MAC axis. If both
					#would be strict, there would be only a singel True
					#value allowed. But different multiplication results
					#could occur and that is ok. We ask the user to give
					#the strict axis first.
					strict=isfirsthistax,
			)
			isfirsthistax = False
			
		#The two ways for getting statistics shall yield the same.
		#They both work on bits and not probabilities, but one accumulates
		#with uint datatype, the other one by implementing bit-wise
		#addition. If this does not work, the probability stuff is
		#doomed to fail.
		#This check is similar to selfcheckindummy.
		assert np.all(
						statisticuintresult == \
						statisticdummyintresult
				), \
				"Deriving statistics with uint dtype and with " \
						"bitwise computation shall yield same result."
							
						
	@classmethod
	def test_operandStatStoc(cls, simulationResult, subtests):
		"""Compare statistic and stochastic operands using `statStocCompare`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		#Check weight and activations in statistic and stochastic simulation.
		#They shall be the same just like the result later shall be the same.
		checkoperands = ("activations", "weights")
		for checkoperand in checkoperands:
			with subtests.test(
					checkoperand=checkoperand,
					topmsg="Statistic and stochastic input operands shall match",
			):
				cls.statStocCompare(
						stat=retstatisticdummy[checkoperand],
						stoc=retstochastic[checkoperand],
						stataxis=STAT_AXIS,
				)
				#No need to also check with retstatistic, they
				#derive weights and activations the same way as
				#retstatistics and it anyhow is asserted below, that compuations
				#yield exactly the same
				
				
	@classmethod
	def test_resultsStatStoc(cls, simulationResult, subtests):
		"""Compare statistic and stochastic MVM results using `statStocCompare`.
		
		Parameters
		----------
		simulationResult : `tuple`
			Filled from `pytest.fixture` `simulationResult`.
			Contains results from `simulateMvm` call for one element of
			`GROUPS`.
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		"""
		
		groupsshapeshistaxesgroupnames, \
		levels, \
		nummacs, \
		groups, \
		shapesfromgroups, \
		histaxesfromgroups, \
		storegroupnamesfromgroups, \
		comparegroupnamesfromgroups, \
		retstochastic, \
		retstatistic, \
		retstatisticdummy, \
		retfullscalestoc, \
		retfullscalestat = simulationResult
		
		#Check that statistics have same bit probabilities as forecasted
		#with stochastic computing. Compare to dummy, because
		#that one has bool and not uint results and is hence
		#comparable
		for resultidx in range(len(retstatistic["results"])):
			with subtests.test(
					resultidx=resultidx,
					topmsg="Bit probabilities in stochastics should "
					"match statistics.",
			):
				#Access bit and probability results. Compare statistics
				#and dummy stochastic, because they both are histograms.
				#Default statistics are uint. But that is ok, as we already
				#compared default and dummy statistics.
				stat = retstatisticdummy["results"][resultidx]
				stoc = retstochastic["results"][resultidx]
				
				#Cliplimitfixed is derived from stddev and that is different
				#between statistic and stochastic. But that gives a shape.
				#We now pad shorter axes with 0 and keep given values
				#centered. We could also derive statistic with cliplimitfixed
				#of stochastic, but we want to test somewhere, that
				#they derive similar stddev.
				#Do not equalize along statistics axis, it is expected
				#that stochastics have only a single element there.
				stat, stoc = padToEqualShape(
						a=stat,
						b=stoc,
						excludeaxes=(STAT_AXIS,),
						padsymmetricaxes=histaxesfromgroups[resultidx],
				)
				
				cls.statStocCompare(
						stat=stat,
						stoc=stoc,
						stataxis=STAT_AXIS,
				)
		

class test_unquantQuantComparisonPlot(BaseTestCase):
	"""Test and compare unquantized and quantized results.
	
	This checks `applyCliplimitStddevAsFixedFrom`, `equalizeQuantizedUnquantized`,
	`plotHist` and `computeSqnr` in *dostochastic* and *dostatistic*.
	
	This asserts that an experiment as in `experiments` yields a useful
	SQNR.
	
	Automatic fixture parameterization and class-scoped state
	as in `test_pytestFeatures`.
	"""
	
	ASSERT_SNR_TOL = 1e-1
	"""`float` : Assert SNR similarity with *0.1* dB precision."""
	
	#Groups for reference
	REF_GROUPS=(
			dict(
					reduceaxes=(MAC_AXIS,),
					chunksizes=(8,),
					mergevalues=None,
					cliplimitstddev=None,
					cliplimitfixed=None,
					positionweights=None,
					positionweightsonehot=None,
					disablereducecarries=None,
					chunkoffsetsteps=None,
					histaxis=HIST_AXIS,
					docreatehistaxis=True,
					mergeeffortmodel=None,
					allowoptim=True,
			),
			dict(
					reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
					chunksizes=None,
					mergevalues=None,
					cliplimitstddev=None,
					cliplimitfixed=None,
					positionweights=("hist", "hist",),
					positionweightsonehot=None,
					disablereducecarries=None,
					chunkoffsetsteps=None,
					histaxis=HIST_AXIS,
					docreatehistaxis=False,
					mergeeffortmodel="analog",
					allowoptim=True,
			),
			dict(
					reduceaxes=(-2,),
					chunksizes=None,
					mergevalues=None,
					cliplimitstddev=None,
					cliplimitfixed=None,
					positionweights=None,
					positionweightsonehot=None,
					disablereducecarries=None,
					chunkoffsetsteps=None,
					histaxis=HIST_AXIS,
					docreatehistaxis=False,
					mergeeffortmodel=None,
					allowoptim=True,
			),
	)
	"""`tuple` of `dict` : Template for *groups* in `simulateMvm`."""
	
	#Common arguments for simulateMvm
	COMMON_ARGS = dict(
			selfcheckindummy=True,
			activationlevels=7,
			weightlevels=3,
			nummacs=32,
			randombehave="norm",
			randomclips=(2., 2.,),
	)
	"""`dict` : Arguments always passed to `simulateMvm`."""
	
	#Whether quantization is added in final or intermediate stage.
	#Also give some confidence relax. In general, having a cliplimit
	#at the interm stage is problematic,
	#because teh cliplimit first introduces a gain varying between stat
	#and stoc and that moves where quantize levels sit at the output.
	#Also remember if the raw data is stored to file.
	FINAL_INTERM_CASES = (
			#Quantize finalize, but with cliplimit in intern stage
			("finalclipbefore", 1., True,),
			#Quantize only in final group
			("final", 1., True),
			#Quantize only in final group, but without cliplimit
			("finalunclipped", 1., True),
			#Quantize in intermediate group and final group
			("intermediate", 2., False),
			#Quantize in intermediate group only
			("intermediateonly", 100., False),
	)
	"""`tuple` : How to get *quantized* results by modifying `REF_GROUPS`."""
	
	#Remember statisticdim and name of a run und then do the same stuff
	#statistic and stochastic. Also remember whether stochastic run
	#should update cliplimit from stddev to fixed to grant same hist axes.
	#But well, statistic computation just gives a slightly different stddev
	#and if we correct that we no more have the same SNR. So rather keep it.
	#Also remember if data is stored to file
	STOC_STAT_CASES = (
			(None, "Stochastic", False, True,),
			(10000, "Statistic", False, False,),
	)
	"""`tuple` : Statistic and stochastic simulation cases."""

	@classmethod
	@pytest.fixture(scope="class")
	def unquantQuantComparisonPlotState(cls, tmp_numpy):
		"""Manage state across tests as in `test_pytestFeatures.sumassertion`.
		
		In the end asserts deviation of SNR between `STOC_STAT_CASES` and
		stores data to *npz* file.
		
		Parameters
		----------
		tmp_numpy : `pathlib.Path`
			Created by `tmp_numpy` `pytest.fixture`.

		Yields
		------
		:
			`list` and `dict` objects to be filed by the many
			`test_unquantQuantComparisonPlot` calls created by parameterization.

		"""
		
		snrs = dict()
		storearrays = dict()
		lastfinalinterm = [None]
		unquantgroups = list()
		quantgroups = list()
		unquantgroupsoriginal = list()
		quantgroupsoriginal = list()
		cliplimitsapplied = [False]
		
		yield snrs, storearrays, lastfinalinterm, unquantgroups, quantgroups, unquantgroupsoriginal, quantgroupsoriginal, cliplimitsapplied
		
		#Export plots
		np.savez(
				file=(tmp_numpy / "psumsim_plot_data_quant_unquant"),
				**storearrays,
		)
		
		#Check that computed SNRs between statstoccases are similar
		for finalintermconfidencerelax in cls.FINAL_INTERM_CASES:
			finalinterm, confidencerelax, _ = finalintermconfidencerelax
			snrsfinalinterm = snrs[finalinterm]
			#Do not compare across aded final or interm quantization, there the
			#values will differ
			refsnr = None
			for casename, snr in snrsfinalinterm.items():
				snr = np.squeeze(snr).item()
				if refsnr is None:
					refsnr = snr
				else:
					#print("{:.3f}, {:.3f}".format(refsnr, snr))
					snrdeviation = abs(snr - refsnr)
					assert snrdeviation < \
							(cls.ASSERT_SNR_TOL * confidencerelax), \
							f"SNR deviates too much from reference value in {finalinterm} {casename}"
								
	@classmethod
	def pytest_generate_tests(cls, metafunc):
		"""Parameterize testcases.
		
		This uses `itertools.product` to combine `FINAL_INTERM_CASES` and
		`STOC_STAT_CASES`. Works like
		`test_pytestFeatures.pytest_generate_tests` and
		creates a *unquantQuantComparisonPlotCase* `pytest.fixture`.

		Parameters
		----------
		metafunc : `pytest.Metafunc`
			If a *unquantQuantComparisonPlotCase* fixture request is found
			here, it is parametrized.

		"""
		if "unquantQuantComparisonPlotCase" in metafunc.fixturenames:
			cases = itertools.product(cls.FINAL_INTERM_CASES, cls.STOC_STAT_CASES)
			cases, ids = itertools.tee(cases)
			ids = (str(dict(finalinterm=i[0], statstoc=j[1])) for i, j in ids)
			metafunc.parametrize("unquantQuantComparisonPlotCase", cases, ids=ids)
		
	@classmethod
	def test_unquantQuantComparisonPlot(
			cls,
			subtests,
			unquantQuantComparisonPlotState,
			unquantQuantComparisonPlotCase,
		):
		"""Assert *mergeeffort* result returned by `simulateMvm`.
		
		Parameters
		----------
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		
		unquantQuantComparisonPlotState : `tuple`
			Filled from `pytest.fixture` `unquantQuantComparisonPlotState`.
			Contains test state in *class* scope.
			
		unquantQuantComparisonPlotCase : `tuple`
			Filled from `pytest.fixture` *unquantQuantComparisonPlotCase*
			defined in `pytest_generate_tests`.
			Contains the kind of quanized experiment to run and to compare
			to unquantized result.

		"""
		
		#Contextual import, because this one takes long
		from matplotlib import pyplot as plt
		
		snrs, storearrays, lastfinalinterm, unquantgroups, quantgroups, unquantgroupsoriginal, quantgroupsoriginal, cliplimitsapplied = unquantQuantComparisonPlotState
		finalintermconfidencerelax, statstoc = unquantQuantComparisonPlotCase
		finalinterm, confidencerelax, storefinalinterm = finalintermconfidencerelax
		statisticdim, casename, updateunquantgroups, storestatstoc = statstoc
		
		#Re-compute groups, if we switched between quantizing final or
		#intermediate
		if finalinterm != lastfinalinterm[0]:
			#Unquantized groups in principle are the references
			unquantgroups.clear()
			unquantgroups.extend(copy.deepcopy(cls.REF_GROUPS))
			#When adding intermediate quant, the final quant exists in both
			if finalinterm == "intermediate":
				unquantgroups[-1]["cliplimitstddev"] = 3
				unquantgroups[-1]["mergevalues"] = -7
			#The later runs can update the unquantizedgroups. Store original one
			#here.
			unquantgroupsoriginal.clear()
			unquantgroupsoriginal.extend(copy.deepcopy(unquantgroups))
			
			#THe quantized run now introduces even more quant
			quantgroups.clear()
			quantgroups.extend(copy.deepcopy(unquantgroups))
			if finalinterm in ("intermediate", "intermediateonly",):
				quantgroups[-2]["cliplimitstddev"] = 3
				quantgroups[-2]["mergevalues"] = -3
			#In final quant, we add the cliplimit also to the reference, as
			#plots look nice then
			elif finalinterm == "finalclipbefore":
				unquantgroups[-1]["cliplimitstddev"] = 3
				quantgroups[-1]["cliplimitstddev"] = 3
				quantgroups[-1]["mergevalues"] = -7
			#There is also a version which even adds cliplimit just finally
			elif finalinterm == "final":
				quantgroups[-1]["cliplimitstddev"] = 3
				quantgroups[-1]["mergevalues"] = -7
			#And a version adding final quant, but no cliplimit
			elif finalinterm == "finalunclipped":
				quantgroups[-1]["mergevalues"] = -7
			
			quantgroupsoriginal.clear()
			quantgroupsoriginal.extend(copy.deepcopy(quantgroups))
			
			#Is set after one statstoccase changed the groups with known
			#cliplimits
			cliplimitsapplied[0] = False
				
		lastfinalinterm[0] = finalinterm


		#Centrally defined, as they are needed in multiple spots.
		dostatistic = statisticdim is not None
		dostatisticdummy = False
		
		caseargs = dict(
				statisticdim=statisticdim,
				dostatisticdummy=dostatisticdummy,
		)
		
		FIGURE_NAME = f"Unquant_Quant_Comparison_{finalinterm}_{casename}"
		fig = plt.figure(num=FIGURE_NAME)
	
		#Unquantized experiment
		retunquant = simulateMvm(
				groups=unquantgroups,
				**cls.COMMON_ARGS,
				**caseargs,
		)
		
		#Create new group, where cliplimitfixed is set from stochastic experiment,
		#to have the two comparable. But only, if the quantized experiment
		#does not introduce the cliplimitstddev. Do not compare
		#against unquantgroups, because that maybe got a cliplimitfixed
		#instead of cliplimitstddev above.
		onlyatindices = ((i["cliplimitstddev"] == j["cliplimitstddev"]) and (i["cliplimitstddev"] is not None) for i, j in zip(unquantgroupsoriginal, quantgroupsoriginal))
		onlyatindices = enumerate(onlyatindices)
		onlyatindices = [i for i, j in onlyatindices if j]
		#If we would equalize the last group, but also introduce
		#quantization in intermediate stage, skip the equalization.
		#Because the intermediate quantization will scramble up
		#stddev, totally, such that the kep cliplimitfixed makes no sense
		if (2 in onlyatindices) and (quantgroupsoriginal[1]["mergevalues"] != unquantgroupsoriginal[1]["mergevalues"]):
			onlyatindices.remove(2)
		#If an earlier run already aplied cliplimitfixed, do not do it
		#here.
		if cliplimitsapplied[0]:
			onlyatindices = tuple()
		

		cliplimitfixedgroups = applyCliplimitStddevAsFixedFrom(
				groups=quantgroups,
				fromreturn=retunquant,
				onlyatindices=onlyatindices,
		)

		#Run quantized experiment
		retquant = simulateMvm(
				groups=cliplimitfixedgroups,
				**cls.COMMON_ARGS,
				**caseargs,
		)
		
		#Use method, which equalizes the two histograms to same length.
		resunquant, resquant = equalizeQuantizedUnquantized(
				retunquantized=retunquant,
				retquantized=retquant,
				runidx=-1,
				histaxis=HIST_AXIS,
				stataxis=STAT_AXIS,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
		)
		
		#Get a refbincount derived from maxhistvalue of unquantized experiment.
		#The unquantized one has more bins and the equalizer equalized the
		#quantized one to that.
		refhistlen = np.squeeze(retunquant["maxhistvalues"][-1]).item()
		refbincount = histlenToBincount(histlen=refhistlen)
		
		#Get SNR and bin-wise qauntization error powers
		snr, probabilityerror, magnitudeerror, squarederror = computeSqnr(
				unquantized=resunquant,
				quantized=resquant,
				histaxis=HIST_AXIS,
				stataxis=STAT_AXIS,
				bincount=refbincount,
				dostatistic=dostatistic,
				dostatisticdummy=dostatisticdummy,
				errordtype="float",
		)
		
		#Remember snr for later assertion
		snrsfinalinterm = snrs.setdefault(finalinterm, dict())
		snrsfinalinterm[casename] = snr
		
		#Remember arrays to store in file
		if storestatstoc and storefinalinterm:
			storekeypattern = f"{casename}_{finalinterm}_{{}}"
			storearrays[storekeypattern.format("unquant")] = resunquant
			storearrays[storekeypattern.format("quant")] = resquant
			storearrays[storekeypattern.format("probabilityerror")] = probabilityerror
			storearrays[storekeypattern.format("magnitudeerror")] = magnitudeerror
		
		#Possibyl upadte the groups with a cliplimitfixed, such
		#that statistic and stochastic plot use the same stuff here
		if updateunquantgroups:
			unquantgroups = applyCliplimitStddevAsFixedFrom(
					groups=unquantgroups,
					fromreturn=retunquant,
					onlyatindices=None,
			)
			quantgroups = applyCliplimitStddevAsFixedFrom(
					groups=quantgroups,
					fromreturn=retquant,
					onlyatindices=None,
			)
			#Remember that cliplimit was applied
			cliplimitsapplied[0] = True
		
		plotcases = [
				(resunquant, "Unquantized",),
				(resquant, "Quantized",),
				(probabilityerror, "Probability Error",),
				(magnitudeerror, "Magnitude Error",),
				(squarederror, "Squared Error",),
		]
		axes = fig.subplots(nrows=len(plotcases), sharex=True, squeeze=False)
		axes = np.squeeze(axes, axis=1)
		axesiter = iter(axes)
		
		for res, label in plotcases:
			ax = next(axesiter)
			islast = ax is axes[-1]
			plotHist(
					hist=res,
					histaxis=HIST_AXIS,
					stataxis=STAT_AXIS,
					refbincount=refbincount,
					axorfigname=ax,
					xlabel=((islast or None) and "Result Value"),
					ylabel=label,
					label=label,
			)
		
		cls.concludeFigure(fig=fig)
		

class test_misc(BaseTestCase):
	"""Other, simple tests, which do not need the concepts of
	`test_pytestFeatures`."""
	
	@classmethod
	def test_statStocComparisonPlot(cls, tmp_numpy):
		"""Compare *dostatisticdummy* and *dostochastic* results.
		
		No SQNR computation, just `simulateMvm`,
		`applyCliplimitStddevAsFixedFrom` and `plotHist` are used.

		Parameters
		----------
		tmp_numpy : `pathlib.Path`
			Created by `tmp_numpy` `pytest.fixture`.

		"""
		
		#Test and plot same experiment, but once stochastic and once statistic.
		#Just check that it does not crash.
		
		#Contextual import, because this one takes long
		from matplotlib import pyplot as plt
		
		COMMON_GROUPS=(
				dict(
						reduceaxes=(MAC_AXIS,),
						chunksizes=(8,),
						mergevalues=None,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=True,
						mergeeffortmodel=None,
						allowoptim=True,
				),
				dict(
						reduceaxes=(WEIGHT_AXIS, ACT_AXIS),
						chunksizes=None,
						#mergevalues=2,
						mergevalues=None,
						#cliplimitstddev=2,
						cliplimitstddev=None,
						cliplimitfixed=None,
						positionweights=("hist", "hist",),
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel="analog",
						allowoptim=True,
				),
				dict(
						reduceaxes=(-2,),
						chunksizes=None,
						#mergevalues=4,
						mergevalues=None,
						cliplimitstddev=3,
						#cliplimitstddev=None,
						#cliplimitfixed=4,
						cliplimitfixed=None,
						positionweights=None,
						positionweightsonehot=None,
						disablereducecarries=None,
						chunkoffsetsteps=None,
						histaxis=HIST_AXIS,
						docreatehistaxis=False,
						mergeeffortmodel=None,
						allowoptim=True,
				),
		)
		
		COMMON_ARGS = dict(
				selfcheckindummy=True,
				activationlevels=7,
				weightlevels=3,
				nummacs=32,
				randombehave="norm",
				randomclips=(2., 2.,),
		)
		
		FIGURE_NAME = "Stat_Stoc_Comparison"
		
		fig = plt.figure(num=FIGURE_NAME)
		
		axes = fig.subplots(nrows=2, sharex=True)
		
		#This is how we call simulateMvm
		retstochastic = simulateMvm(
				statisticdim=None,
				dostatisticdummy=False,
				groups=COMMON_GROUPS,
				**COMMON_ARGS,
		)
		
		#Create new group, where cliplimitfixed is set from stochastic experiment,
		#to have the two comparable.
		cliplimitfixedgroups = applyCliplimitStddevAsFixedFrom(
				groups=COMMON_GROUPS,
				fromreturn=retstochastic,
				onlyatindices=None,
		)
		
		retstatistic = simulateMvm(
				statisticdim=10000,
				dostatisticdummy=False,
				groups=cliplimitfixedgroups,
				**COMMON_ARGS,
		)
		
		#Extract single results
		resstochastic = retstochastic["results"][-1]
		resstatistic = retstatistic["results"][-1]
		
		plotcases = (
				(resstochastic, "Stochastic",),
				(resstatistic, "Statistic",),
		)
		axesiter = iter(axes)
		
		#The stochastic generates the bincount, which the int result does not
		#know but which it needs to be turned into a hist
		refbincount = None
		
		#Results from plothist, where stat -> stoc conversion has been made
		#For storing in file
		storearrays = dict()
		
		for res, label in plotcases:
			ax = next(axesiter)
			islast = ax is axes[-1]
			_, _, histvalues, _, refbincount = plotHist(
					hist=res,
					histaxis=-1,
					stataxis=0,
					refbincount=refbincount,
					axorfigname=ax,
					xlabel=((islast or None) and "Result Value"),
					ylabel="Probability",
					label=label,
			)
			storearrays[label] = histvalues
		
		cls.concludeFigure(fig=fig)
		
		#Export plots
		np.savez(
				file=(tmp_numpy / "psumsim_plot_data_stat_stoc"),
				**storearrays,
		)
	
	
	@classmethod
	def test_runAllExperiments(cls, tmp_json):
		"""Test `runAllExperiments`.
		
		Just checks, that sharding or different number of processes does
		not influence the result. It is also checked, that already existing
		results are correctly read.

		Parameters
		----------
		tmp_json : `pathlib.Path`
			Created by `tmp_json` `pytest.fixture`.

		"""
		
		#Results created in two blocks
		jsonpathscliced = (tmp_json / "psumsim_result_test_sliced.json")
		#And in one chunk
		jsonpathfull = (tmp_json / "psumsim_result_test_full.json")
		#And path for simulating single runs
		jsonpathspecific = (tmp_json / "psumsim_result_test_specific.json")
		#And path for proress file
		progresspath = (tmp_json / "psumsim_progress_test.txt")
		#Names of single runs. Choose some which need references
		specificruns = (
				"rb_norm_sc_3_nm_64_al_3_wl_3_ic_3_fc_3_cs_10_fl_3_il_3",
				"rb_uniform_sc_3_nm_64_al_3_wl_3_ic_3_fc_3_cs_10_fl_3_il_3",
		)
		
		COMMON_ARGS = dict(
				runreduced=True,
				runquiet=True,
				processes=None,
				runkeys=None,
				progressfp=None,
				aggressive=False,
		)
		
		#Create some results from scratch
		#Overwrite exisitng results file and do not search for exisiting
		#results, as this is non readable
		jsonfp = jsonpathscliced.open("w")
		runAllExperiments(
				**COMMON_ARGS,
				jsonfp=jsonfp,
				iterbegin=0.3,
				iterend=0.7,
		)
		jsonfp.close()
		
		#Create results on top. Readable filepath keeps old results
		jsonfp = jsonpathscliced.open("r+")
		runAllExperiments(
				**COMMON_ARGS,
				jsonfp=jsonfp,
				iterbegin=None,
				iterend=None,
		)
		jsonfp.close()
		
		#Create results in one big chunk. Try using progbar and aggressive
		#running here.
		jsonfp = jsonpathfull.open("w")
		runAllExperiments(
				**{
						**COMMON_ARGS,
						"runquiet" : False,
						"aggressive" : True,
				},
				jsonfp=jsonfp,
				iterbegin=None,
				iterend=None,
		)
		jsonfp.close()
		
		#Read results back
		readresults = list()
		for jsonpath in (jsonpathscliced, jsonpathfull):
			fp = jsonpath.open("r")
			readresults.append(json.load(fp))
			fp.close()
			
		assert readresults[0] == \
				readresults[1], \
				"Sliced and full experiment do not create same results"
		
		#Also check running two specific runs without the multiprocessing
		progressfp = progresspath.open("a")
		jsonfp = jsonpathspecific.open("w")
		runAllExperiments(
				jsonfp=jsonfp,
				iterbegin=None,
				iterend=None,
				#Do not run reduced, as we now specify which runs to run
				runreduced=False,
				#Also not quiet, because we want to write progress file
				runquiet=False,
				#A single job skips multiprocessing
				processes=1,
				#No aggressive, as that has no influence with single worker
				aggressive = False,
				#Specify exact experiments to run
				runkeys=specificruns,
				#Write progress file
				progressfp=progressfp,
		)
		progressfp.close()
		jsonfp.close()
		
	@classmethod
	@pytest.mark.parametrize("skipthisruncase", (True, False))
	def test_runDescription(cls, skipthisruncase):
		"""Test `RunDescription`
		
		Especially checks
		
		- that *sqnrreference* and *skipthisrun* are set correctly.
		
		- that *createdummy* works.
		
		- that the objects can be used as *key* in `dict` (see `hash`)
		
		- that the objects can be used with `pickle`
		
		- that the objects can be converted to/from strings
		

		Parameters
		----------
		skipthisruncase : `bool`
			If set, the created `RunDescription` has *skipthisrun* set.
			Set by `@pytest.mark.parametrize`.

		"""
		if not skipthisruncase:
			rundesc = RunDescription(
					nummacs=4,
					chunksize=3,
					activationlevels=10,
					weightlevels=9,
					intermediatelevels=7,
					finallevels=10,
					initialcliplimit="occ",
					intermediatecliplimit="occ",
					finalcliplimit="occ",
					randombehave="uniform",
					allowskip=False,
					createdummy=False,
			)
		else:
			#This is anyhow skipped, so we can test all combinations of
			#types for e.g. cliplimit to test that they are restored
			#correctly from string.
			rundesc = RunDescription(
					#Is always turned to int
					nummacs=4,
					#Chunksize could be None, not just a number as above
					chunksize=5,
					#Levelcount can be a number or None. Skip this run
					#by specifying weight precision larger than activation
					activationlevels=10,
					weightlevels=11,
					intermediatelevels=None,
					finallevels=None,
					#Cliplimit can be None, number or OCC string
					initialcliplimit="occ",
					intermediatecliplimit=None,
					finalcliplimit=3,
					#Is always converted to str
					randombehave="uniform",
					allowskip=True,
					createdummy=False,
			)
			
		#Assert the skip flag
		assert skipthisruncase == rundesc.skipthisrun, "Test run description skips"
		
		#Turn to str. This will disregard dummy info
		rundescstr = rundesc.toStr()
		
		#This will create a dummy without reference info
		rundescdummy = rundesc.copy(allowskip=skipthisruncase, createdummy=True)
		
		assert rundescdummy.sqnrreference is \
				None, \
				"A dummy must have a None SQNR reference."
		
		#The None case has no intermediate or final quant. It hence
		#has no reference. We can there assert that the dummy changes
		#nothing.
		if skipthisruncase:
			assert rundesc == \
					rundescdummy, \
					"THe dummy shold not change anything here."
		#In the other case, we have an SQNR reference and can check that
		#the chain of rundescription breaks there
		else:
			assert rundesc.sqnrreference.sqnrreference is \
					None, \
					"A SQNR reference must have a None SQNR reference."
		
		#Go back, this will re-derive dummy info
		rundescrederived = RunDescription.fromStr(
				thestr=rundescstr,
				allowskip=skipthisruncase,
				createdummy=False,
		)
		
		assert rundesc == \
				rundescrederived, \
				"Going to and from str changes rundescription"
				
		#Check that rundescription is a valid key
		testdict = dict()
		testdict[rundesc] = None
		
		#Pickel and unpickle and compare that they are the same
		rundescpickled = pickle.dumps(rundesc)
		rundescunpickled = pickle.loads(rundescpickled)
		
		assert rundesc == rundescunpickled, \
				"Pickling changes rundescription"
	
	@classmethod
	def test_stimulusPlot(cls, tmp_numpy):
		"""Plot and store some exemplary stimulus.
		
		This tests `generateSimulationOperands` and `plotHist`.

		Parameters
		----------
		tmp_numpy : `pathlib.Path`
			Created by `tmp_numpy` `pytest.fixture`.

		"""
		
		toexport = dict()
		
		for randombehave in ("norm", "truncnorm", "uniform", "fullscale", "sinusoidal"):
			stimuli = generateSimulationOperands(
					statisticdim=None,
					dostatisticdummy=False,
					activationlevels=15,
					weightlevels=1,
					nummacs=1,
					randombehave=randombehave,
					randomclips=(2., 2.),
			)
			stimuli = stimuli["activations"]
			stimuli = np.squeeze(stimuli, axis=1)
			toexport[randombehave] = stimuli
			
			fig, _, _, _, _ = plotHist(
					hist=stimuli,
					histaxis=HIST_AXIS,
					stataxis=STAT_AXIS,
					refbincount=histlenToBincount(histlen=stimuli.shape[HIST_AXIS]),
					axorfigname=f"Stimulus_{randombehave}",
					xlabel="Value",
					ylabel="Probability",
					label=None,
			)
			
			cls.concludeFigure(fig=fig)
		
		#Export stuff
		np.savez(
				file=(tmp_numpy / "psumsim_plot_data_stimulus"),
				**toexport
		)
		
	@classmethod
	def test_quantNoiseFormula(cls, subtests, tmp_numpy):
		r"""Get SNR for quantizing sinusoidal signal.
		
		Gets sinusoidal signal at different bitwidths and computes SQNR on them.
		Is supposed to re-produce the :math:`1.76 + 6.02n` equation [QUANT]_.
		Some things to regard:
			
		- `computeSqnr` always needs the histogram of the unquantized signal.
		  For the equation, we would need a histogram with an infinite number
		  of bins. But we only have one with a large number of bins. Hence,
		  for large bitwidths, the values deviate from equation. Because
		  the error added by quantizing to *n* bits no more dominates the
		  error added by having an already quantized reference.
		  
		- The :math:`+1.76` is only true for a sinusoid. It describes the SQNR
		  reached when a quantizer has 0 levels and outputs 0 all the time.
		  The quantization error (Noise) is then in [QUANT]_ assumed to be a signal
		  with uniform distribution bound to +/- half the full scale. We define
		  that full scale to be 1 and that signal then has a power of
		  :math:`\frac{1}{12}`. To derive the *1.76*, we use that noise power
		  and the full signal power to derive an SQNR. And the full signal
		  power depends on the signal, normalized to be limited on an interval
		  :math:`[-0.5;0.5]`. We derive that value here for each signal form
		  by asking its random process for `scipy.stats.rv_continuous.moment`
		  and ask for the second moment. That second moment is a
		  probability-weighted sum of signal squares without any mean subtracing
		  and exactly that is the signal power. We export these numbers.

		- To observe the equation correctly, one needs to use *uniform* instead
		  of *sinusoidal*. Because, for very low bitwidths, the quantization
		  error becomes dependent of the signal waveform and can no more be
		  approximated with a bunch of triangles. And only uniformly distributed
		  signals yield by a linear waveform have triangular-shaped quantization
		  error even in that case.
		  
		- The bincounts we set are a number of levels, not bits. And they
		  are not a power of two. So if *n* is the bincount this function
		  exports, you will observe :math:`6.02 log_{2}(n)`.
		  
		- Lastly, the number of quantization steps and bincount *n* is then often
		  used to derive a maximum quantization error :math:`\frac{1}{2n}`
		  referred to full scale. But that is not entirely true: if one has
		  *n* quantization steps, the quantization step height is :math:`n-1`
		  and that is what the equation cares about. So what this function then
		  actually computes is :math:`6.02 log_{2}(n-1)`.
		
		The SNRs and the finite, large bincount are stored to JSON and NPZ files.
		
		The test uses `generateSimulationOperands`, `quantizeClipScaleValues`
		and `computeSqnr`.
		 
		Parameters
		----------
		subtests : `pytest.fixture`
			Needed to specify assertions in sub-tests.
		
		tmp_numpy : `pathlib.Path`
			Created by `tmp_numpy` `pytest.fixture`.

		"""
		
		#Will fill JSON/NPZ data here
		toexport = dict()
		
		#Operands with this histlens (related to quant bitwidth) are created.
		#FIrst one is the reference used for SQNR computation.
		histlens = (8191, 4095, 2047, 1023, 511, 255, 127, 63, 31, 16, 7, 3, 1)
		#Bincounts will be computed and filled here
		bincounts = list()
		
		#the randomclip we use to draw operands.
		randomclip = 2.
		
		#Mass of gaussian process in bounds +-randomclip
		mass = scipy.stats.norm.cdf(randomclip, loc=0., scale=1.) - \
				scipy.stats.norm.cdf(-randomclip, loc=0., scale=1.)
		
		#These operand distributions are drawn. For each signal we also 
		#derive the power of the signal scaled to swing 1.0. This is needed
		#to compute the 1.76 for a quantized sin, but is different for other
		#signal shapes.
		#So we take the random distributions, scale them to full scale 1.0
		#and ask for the second momentum. That momentum is the expectation
		#value of squared drawn values and exactly that is the signal power.
		randombehavesignalpowers = (
				#Sinusoidal is scaled to full scale 1
				("sinusoidal", sinusoidal.moment(order=2, loc=0, scale=0.5),),
				#Same for uniform. loc gives position of left edge,
				#scale gives width
				("uniform", scipy.stats.uniform.moment(order=2, loc=-0.5, scale=1),),
				#Truncnorm uses an unscaled normal distribution and lets
				#that yield values between -randomclip and +randomclip and these
				#values are divided by (2*randomclip) and that gives a distribution,
				#yielding only values with a swing of 1 and sampling the correct
				#number of sigmas of a gauss.
				("truncnorm", scipy.stats.truncnorm.moment(
						order=2,
						loc=0,
						scale=1/2./randomclip,
						a=-randomclip,
						b=randomclip,
				),),
				#Same for norm dist, but this one does not accept arguments
				#a and b. So we instead ask truncorm, revert its normalization to
				#have area 1 of pdf in range [a;b]. We then add power of clipped
				#bins. We do all this for an unscaled gauss, so we in the end scale
				#down from yielding numbers in [-randomclip;randomclip] to
				#yielding [-1;1] and then [-0.5;0.5], becaue we shall compute power
				#of the signal with swing 1.
				("norm", (
						((scipy.stats.truncnorm.moment(
								order=2,
								loc=0.,
								scale=1.,
								a=-randomclip,
								b=randomclip,
						) * mass) + (1.-mass) * (randomclip**2.)) / \
						(randomclip * randomclip * 2. * 2.)
				),),
		)
		#And we check statistic and stochastic experiment. We also need to
		#pass name of the field returned by generateSimulationOperands to get
		#a length-1 hist axis in statistics.
		statisticdims = ((None, "activations"), (1000, "activationsint"))
		
		#Only test assertion
		#randombehaves = randombehaves[1:]
		#statisticdims = statisticdims[0:1]
		
		#Also define power of full-scale quantization noise, meaning quantizing
		#without any bits. Our signal in the math used here is normalized to
		#swing 1.0. So we would get a uniformly distributed quant error
		#in bounds -0.5/+0.5. The power of this signal is 1./12., or
		#scipy.stats.uniform.moment(order=2, loc=-0.5, scale=1)
		worstnoisepower = 1./12.
		
		#Format of fields in results dict
		exportnameformat = "randombehave_{randombehave}_dostatistic_{dostatistic}"
		#Compute signal-dependent equation offset in these fields
		exportoffsetnameformat = "randombehave_{randombehave}_equation_offset"
		
		#Field name to do assertion against equation: Use the one with least
		#inaccuracies described in docstring
		expectedassertname = "randombehave_uniform_dostatistic_False"
		#In assertion, ignore some big bitwidths, where reference is no more
		#good enough.
		expectedassertignore = 5
		
		cases = itertools.product(randombehavesignalpowers, statisticdims)
		
		for randombehavesignalpower, statisticdim in cases:
			randombehave, _ = randombehavesignalpower
			statisticdim, operandfield = statisticdim
			dostatistic = statisticdim is not None
			exportname = exportnameformat.format(
					randombehave=randombehave,
					dostatistic=dostatistic,
			)
			
			with subtests.test(
					topmsg="Computing SQNRs",
					randombehave=randombehave,
					dostatistic=dostatistic,
			):
			
				#For each case, re-generate reference
				ref = None
				
				#Prepare fresh results field to export this case
				snrlist = list()
				toexport[exportname] = snrlist
				
				#Walk over bitwidths
				for histlen in histlens:
					#Create and remember bincount, the actual number of quant steps.
					#Only the first case will o this.
					bincount = histlenToBincount(histlen=histlen)
					if bincount not in bincounts:
						bincounts.append(bincount)
					#First bitwidth is the refernece
					if ref is None:
						refhistlen = histlen
						refbincount = bincount
					#Draw quantized operand. We draw activations and create minimum
					#effort dummy weights and nummacs.
					simulated = generateSimulationOperands(
							statisticdim=statisticdim,
							dostatisticdummy=False,
							activationlevels=histlen,
							weightlevels=1,
							nummacs=1,
							randombehave=randombehave,
							#DOes not matter for uniform and sinusoidal distributions
							randomclips=(randomclip, randomclip),
					)
					#Extract numpy array and purge unneeded dimensions. We then
					#have statistic and hist dimensions.
					simulated = simulated[operandfield]
					simulated = np.squeeze(simulated, axis=1)
					
					#If we create the SQNR reference right now, we remember it, but
					#we cannot compute an SQNR without own reference
					if ref is None:
						ref = simulated
					#Otherwise, compute SQNR
					else:
						#Equalize data by up-scaling quantized bins.
						#equalizeQuantizedUnquantized does this, too but expects
						#more complex input arguments yield by simulateMvm
						rescaled, _, _, _ = quantizeClipScaleValues(
								toprocess=simulated,
								#Add stat and hist dims to maxhistvalue. But they
								#both are supposed to have length 1.
								maxhistvalue=np.expand_dims(
										np.array(histlen),
										axis=(0, 1)
								),
								mergevalues=None,
								dostatistic=dostatistic,
								dostatisticdummy=False,
								cliplimitfixed=None,
								#This is the feature used for equalization. Scale
								#bin values up to reach desired bincount.
								valuescale=(float(refhistlen) / float(histlen)),
								histaxis=HIST_AXIS,
								scaledtype="float",
						)
						#Get SQNR
						snr, _, _, _ = computeSqnr(
								unquantized=ref,
								quantized=rescaled,
								histaxis=HIST_AXIS,
								stataxis=STAT_AXIS,
								bincount=refbincount,
								dostatistic=dostatistic,
								dostatisticdummy=False,
								errordtype="float",	
						)
						#Remember SQNR
						snrlist.append(snr.item())
					
		#More results fields
		toexport["histlens"] = histlens[1:]
		toexport["bincounts"] = bincounts[1:]
		toexport["refbincount"] = refbincount
		
		#Expected results from equation. Do not use 6.02, but rather its
		#definition.
		#The -1 described in docstring is applied here.
		noisegain = 20. * math.log10(2.)
		equation = [noisegain * math.log2(i-1.) for i in bincounts[1:]]
		toexport["equation"] = equation
		
		#But we also need the +1.76, which depends on the quantized signal.
		#Compute and store that value.
		for randombehave, signalpower in randombehavesignalpowers:
			with subtests.test(
					topmsg="Storing Equation Offset",
					randombehave=randombehave,
			):
				exportname = exportoffsetnameformat.format(
						randombehave=randombehave,
				)
				signalpower = signalpower.item()
				offset = 10. * math.log10(signalpower / worstnoisepower)
				toexport[exportname] = offset
		
		#Export as json for debugging
		with open((tmp_numpy / "psumsim_plot_data_snrs.json"), "wt") as fobj:
			json.dump(toexport, fp=fobj, indent="\t")
			
		#Export as numpy for plotting
		toexport = {k:np.array(v) for k, v in toexport.items()}
		np.savez(
				file=(tmp_numpy / "psumsim_plot_data_snrs"),
				**toexport
		)
		
		#Assert values against formula
		with subtests.test(
				topmsg="Equation Matching",
		):
			result = toexport[expectedassertname][expectedassertignore:]
			expected = toexport["equation"][expectedassertignore:]
			#Assert with only an absolute tolerance
			expected = pytest.approx(expected, abs=1e-2,)
			assert expected == result, "Values do not match equation"
		
		#Assert stat against stoc
		for randombehave, _ in randombehavesignalpowers:
			with subtests.test(
					topmsg="SQNR in stat/stoc",
					randombehave=randombehave,
			):
				statname = exportnameformat.format(
						randombehave=randombehave,
						dostatistic=True,
				)
				stocname = exportnameformat.format(
						randombehave=randombehave,
						dostatistic=False,
				)
				statresult = toexport[statname]
				stocresult = toexport[stocname]
				statresult = pytest.approx(statresult, abs=0.5)
				assert statresult == stocresult, \
						"SQNR differs in stat vs. stoc."
			
	@classmethod
	@pytest.mark.parametrize("levels", (1, 3, 7, 15, 31, 63, 64, 127,))
	def test_optimumClippingCriterion(cls, levels):
		"""Test `optimumClippingCriterion`.
		
		This tests, that the iterative function actually converges with the
		default arguments.
		
		Only for 64 levels (6 bits), the value can be asserted against one
		from [OCC]_.

		Parameters
		----------
		levels : `int`
			The *bincount* to derive OCC for. Set with
			`@pytest.mark.parametrize`. 

		"""
		
		#Test OCC function across bitwidths
		
		abstol = 1e-6
		
		#This levels/value pair was read from the paper
		paperlevels = 64
		#Value also generated by tested function with better abstol, but
		#compared to Fig. 2 in paper
		paperocc = 3.2869142152137605
		
		#Value shall raise with bitwidth
		lastocc = None	
		occ = optimumClippingCriterion(
				levels=levels,
				#Go with the default values to check that they are chosen useful
				#abstol=abstol,
				#maxiter=maxiter,
		)
		assert type(occ) is \
				float, \
				"OCC has wrong datatype."
		if lastocc is not None:
			assert occ > \
					lastocc, \
					"OCC should rise with number of levels."
		if levels == paperlevels:
			assert occ == pytest.approx(
						paperocc,
						rel=0.,
						abs=abstol,
				), \
				"OCC does not match expectation from paper."
		lastocc = occ
		
class test_pytestFeatures:
	"""Simple tests asserting that the way we use `pytest` works.
	
	We need
	
	#. A state filled by test functions and evaluated once all tests are done
	
	#. A generator generating a ton of testcases
	
	in a *class* scope to do expensive computations invoked in `pytest.fixture`
	just once and to be able to collect information across test functions."""
		
	@classmethod
	def pytest_generate_tests(cls, metafunc):
		"""Generate parametrized testcases from function.
		
		We here do not parametrize test functions, but instead fixtures.
		A `pytest.fixture` *caseraw* is here created and parametrized with two
		strings. This emulates a creation of test cases from some more
		complex code.
		
		This hook is called by `pytest`.
		See `_pytest.hookspec.pytest_generate_tests`.

		Parameters
		----------
		metafunc : `pytest.Metafunc`
			If a requested `pytest.fixture` *caseraw* is found here, it is
			parametrized.
		"""
		if "caseraw" in metafunc.fixturenames:
			metafunc.parametrize("caseraw", ("Hallo", "Welt"), scope="class")
			
	@classmethod
	@pytest.fixture(scope="class")
	def caseprocessed(cls, caseraw):
		"""Process a raw case and return something.
		
		This `pytest.fixture` invokes the parametrized *caseraw* fixture setup in
		`pytest_generate_tests` and then does some expensive computation on
		each case: here `len`.

		Parameters
		----------
		caseraw : `str`
			`pytest.fixture` created in `pytest_generate_tests`.

		Returns
		-------
		`int`
			Length of *caseraw*.

		"""
		
		return len(caseraw)
	
	@classmethod
	@pytest.fixture(scope="class")
	def sumassertion(cls):
		"""`pytest.fixture` generating a state.
		
		The fixture object is a mutable `list`, which is created with a
		*class* scope and once the scope ended (all test ran), the result
		is asserted. We here assert that tests took `caseprocessed` and added
		1.
		
		Yields
		------
		allpostcases : `list`
			Filled by test functions.

		"""
		
		allpostcases = list()
		yield allpostcases
		assert sum(allpostcases) == len("Hallo") + len("Welt") + 2
		
	@classmethod
	def test_setup(cls, caseprocessed, sumassertion):
		"""An exemplary test function.

		Parameters
		----------
		caseprocessed : `int`
			Returned by `pytest.fixture` `caseprocessed`. 1 is added on this
			number.
		sumassertion : `list`
			Returned by `pytest.fixture` `sumassertion`. This is the test
			state, where we add the increased number.

		"""
		
		assert caseprocessed > 0
		sumassertion.append(caseprocessed+1)
		
