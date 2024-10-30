.. _concepts:

Concepts
========
This section covers some basic, repeating terms and concepts, which are needed
in several other locations.

.. _views:

Views
-----
To prevent copies of `numpy.ndarray`, PSumSim relies on `numpy.ndarray.view`
a lot. If arrays are indexed (see utilities in `array`), most functions do
not return a copy, but a *view* on the indexed data without copying it. This
is faster and consumes less memory.

.. warning:
	If you change a view, the change will reflect in the original
	`numpy.ndarray`.
	
.. _inplaceops:
	
In-Place Operations
-------------------
Operations on `numpy.ndarray` in general do not modify a given array, but
instead copy it before doing so. The performance is thereby slightly
decreased, but the code becomes much moe robust. However, inside functions,
the
`out parameter <https://numpy.org/doc/stable/reference/ufuncs.html#index-0>`_
is then often used to modify arrays
in-place. So for the majority of the operations (happening inside functions),
copies are prevented. 

.. _dataformat:

Data Format
-----------
The whole prupose of PSumSim is to compute using histograms. If an array has
shape *[128, 15]*, it means that it holds *128* histograms. Each histogram
is *15* entries long. It represents signed numbers from *-7* up to *+7*.
PSumSim does not care about the exact value of each bin in the histogram, it
only assumes that they are evenly spaced. The histogram has then exactly
enough bins to cover all possibly occuring values.

The upper example has *bincount* *15*. Its *histlen* is *7* referring to the
maximum magnitude. *2* is the only valid even *bincount* and then by definition
refers to bin values *0* and *1*.

.. _statstoc:

Stochastic vs. Statistic
------------------------
PSumSim computes probabilities of occuring values in an MVM application.
These probabilities usually are ground-truth, stochastic probabilities:
operand distributions are known with full confidence and all results are
derived from there with full confidence.

However, PSumSim also supports doing computations using statistics. In this
operation, a set of operands (not probabilities) is drawn from the random
distributions. These many operands are then MVM'ed in parallel, one in the
end gets many integer MVM results and can derive the (estimated) result
probabilities. This is used for debugging mainly.

The simulation modes are:

dostochastic
	The simulation is made stochastic. All numbers are `float` probabilities.
	There are long histogram axes, giving the probability for each possible
	value. Stochastic simulation is assumed, if *dostatistic* is not set.
	
dostatistic
	The default statistic simulation. Histogram axes are only 1 long and an
	`int` dtype is used to represent actual values processed in an MVM
	application. In return, a long axis, which sweeps along the different MVM
	computations, is added. This simulation can be used to validate stochastic
	results or to circumvent `limitations`.
	
dostatisticdummy
	Another statistic case, which is similar to *dostatistic*. But the dtype
	is bool and in return the histogram axis gets its original length. So all
	values of the MVM computations are encoded in one-hot histograms. Gives
	extrordinary bad compute performance, but this is basically like computing
	with histograms, where all values have probability *0.0* or *1.0*. This
	is excellent for debugging statistic computations, because the same code
	can run a dostatistic or dostatisticdummy computation.
	
`checkStatisticsArgs` checks the simulation mode. A *statisticdim* is either a
number stating how many statistic computations to run or `None` to request
stochastic simulation.

.. _limitations:

Limitations
-----------
Currently, stochastic simulation treats probabilities as independent.
This is an error, if e.g. 128 numbers are summed up: the simulated
probability tells, that getting 100 numbers with value 1 and 100 with value
2 results in some probability of the result
:math:`100\times 1 + 100 \times 2`. But that case can in reality never occur,
because if 100 values are 1, only 28 are left to be 2. However, the error
made by this simplification is neglgible, because getting so many results
with saem value is so unlikely.

Furthermore, stochastic simulation can not simulate on a per-bit basis.
So MVM systems using a bit-parallel bit-serial scheme [BPBS]_ cannot
be simulated. The reason is, that to represent a value probability
by its bit probability, one cannot treat bit probabilities as independent.
Because many value histograms share the same bit probabilities.

.. _quantization:

Quantization
------------
Quantization is defined in PSumSim by `quantizeClipScaleValues`. The term
*mergevalues* is often used in that context and describes how many distinct
bins of a histogram are merged into one quantized bin.
The quantization is defined by :math:`round(value / mergevalues)`.
`round` instead of `math.ceil` or `math.floor` is used, because it limits the
quantizatio-error magnitude to half a quantization step.

PSumSim in general does not refer to *bits*, but rather to *levels*. This
provides more flexibility and is necessary due to the `dataformat`: valid bincounts
are odd values. But odd values refer to a fractional number of bits. Furthermore,
MVM applications often use a sign-magnitude representation for signed
numbers and not a two's-complement form. And this form also gives an odd number
of levels which can be represented.

.. _clipping:

Clipping
--------
In MVM applications, many numbers are multiplied and accumulated. In the end,
full-scale results are very likely and most occuring result values are close to
0. So spacing the quantization values evenly over the entire full-scale range
introduces a huge error. Clipping reduces the bincount by taking a number of
maximum and minimum bins/values and by clipping these values. The bincount
thereby reduces and the same number of quantization levels samples the actually
seen value-probability distribution better [PACT]_.

In general, clipping is parameterized in terms of standard deviations. A
*cliplimitstddev* of *2* means: find the standard deviation and then clip
values at +/- 2 standard deviations. *cliplimitfixed* then refers to what
that :math:`\text{2}\sigma` value actually was.
`getHistStddev` finds standard deviations.
`optimumClippingCriterion` is a pre-defined clip rule from [OCC]_.

When comparing results of simulations, ensure that they used the same cliplimit.
Run a first simulation with *cliplimitstddev* and apply the
*cliplimitfixed* this one found in the next simulation. This resolves the
problem, that the second simulation might find a different :math:`\sigma`.
`applyCliplimitStddevAsFixedFrom` helps with that.
