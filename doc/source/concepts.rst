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
