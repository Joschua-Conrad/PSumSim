"""Basic plot functionality.

Plotting is done using `matplotlib`, which is imported contextual: the import
takes some time and we don't want the commandline to halt every time it is
called.
"""

import numpy as np
from .array import normalizeAxes
from .hist import getHistValues, unpackHist, packStatistic


def plotHist(hist, histaxis, stataxis, refbincount, axorfigname, xlabel, ylabel, label):
	"""Plot a histogram.

	Parameters
	----------
	hist : `numpy.ndarray`
		Some array. A return value of *dostatisticdummy* can get its statisitc
		axis resolved to probabilities. A return value of *dostatistic* is first
		unpacked to `bool` histogram and then also resolved to probabilities.
		So outputs of all simulation modes (see `statstoc`) are supported.
		
	histaxis : `int`
		The axis along which one finds the different histogram bins.
		
	stataxis : `int`, `None`
		If given, `packStatistic` is used along this axis to turn actual
		statistic computation results into value probabilities. See `statstoc`.
		
	refbincount : `int`, `None`
		The *bincount* is the number of bins along the x-axis. By default,
		the length of *histaxis* is used. But results of *dostatistic*
		(see `statstoc`) return a length-1 *histaxis*, are unpacked using
		`unpackHist` and that needs a *bincount*. That value is taken from this
		parameter. If *histaxis* has a larger length, this parameter is
		ignored.
		
	axorfigname : `matplotlib.axes.Axes`, `str`
		The object to plot into. Either pass an existing axes object or the
		name of a new `matplotlib.figure.Figure`.
		
	xlabel : `str`, `None`
		If given, this is the new x-axis label.
		
	ylabel : `str`, `None`
		If given, this is the new y-axis label.
		
	label : `str`, `None`
		If given, this is the legend text of the created plot. Later use
		`matplotlib.axes.Axes.legend` to create the legend.
		
	Raises
	------
	`ValueError`
		If *histaxis* and *stataxis* are the same, or if *refbincount* was
		not passed, but it needed.

	Returns
	-------
	fig : `matplotlib.figure.Figure`
		Figure object.
		
	ax : `matplotlib.axes.Axes`
		Axes object.
		
	hist : `numpy.ndarray`
		The bar heights as plotted after possibly using `unpackHist` and
		`packStatistic`.
		
	histplot : `matplotlib.container.BarContainer`
		Returned by `matplotlib.axes.Axes.bar`.
		
	bincount : `int`
		The now used *bincount*, which is ready for being used on a next plot.
		Often, one would like to plot *dostochastic* and *dostatistic* in
		one figure. Plot the stochastic data then first, remember this returned
		*bincount*, and use it as *refbincount* for plotting statistic data.
		The x-axis lengths are the same then.

	"""
	
	#Contextual import, because this one takes long
	from matplotlib import pyplot as plt
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=hist.ndim)
	#Stataxis can be none meaning "there is no stat axis"
	if stataxis is not None:
		stataxis, = normalizeAxes(axes=stataxis, referencendim=hist.ndim)
		
		if histaxis == stataxis:
			raise IndexError(
					f"histaxis {histaxis} and stataxis {stataxis} cannot be "
					f"the same.",
					histaxis,
					stataxis,
			)
	
	#User can give axes to plot into or just a tring meaning, create a figure
	#with this na,e
	if isinstance(axorfigname, str):
		fig = plt.figure(num=axorfigname)
		ax = fig.subplots()
	else:
		ax = axorfigname
		fig = ax.get_figure()
		
	#Get bincount from values to plot
	bincount = hist.shape[histaxis]
	
	#If we have length 1 along hist axis, unpack. Use bincount maybe known
	#from previous call. That way, one can plot uint results with
	#known bincount or statistic results and let us find the bincount.
	if bincount == 1:
		if refbincount is None:
			raise ValueError(
					"Plotting a histogram, which needs unpacking. "
					"But *refbincount* was not passed."
			)
		
		hist = unpackHist(hist, bincount=refbincount, axis=histaxis)
		#This increases the bincount from 1 to some actual hist
		bincount = int(refbincount)
		
	#Get x axis values
	histvalues = getHistValues(
			bincount=bincount,
			axis=histaxis,
			ndim=hist.ndim,
			padlower=False,
			padupper=True,
	)

	#If there is a stat axis, possibly use it to turn booleans
	#into float and remove the stat axis then.
	if stataxis is not None:
		if hist.shape[stataxis] > 1:
			hist = packStatistic(
					topack=hist,
					axis=stataxis,
					keepdims=False,
			)
		#If the axis existed but has length 1, still squeeze it to have
		#1D hist
		else:
			hist = np.squeeze(hist, axis=stataxis)
	
	histplot = ax.bar(
			x=histvalues,
			height=hist,
			#Using width 1 is very important, otherwise we get aliasing
			width=1.05,
			align="center",
			bottom=0,
			label=label,
	)
	
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	
	return fig, ax, hist, histplot, bincount
