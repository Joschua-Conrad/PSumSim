import numpy as np
from .array import normalizeAxes
from .hist import packUnpackHistCommon, unpackHist, packStatistic


def plotHist(hist, histaxis, stataxis, refbincount, axorfigname, xlabel, ylabel, label):
	
	#Contextual import, because this one takes long
	from matplotlib import pyplot as plt
	
	histaxis, = normalizeAxes(axes=histaxis, referencendim=hist.ndim)
	#Stataxis can be none meaning "there is no stat axis"
	if stataxis is not None:
		stataxis, = normalizeAxes(axes=stataxis, referencendim=hist.ndim)
	
	#User can give axes to plot into or just a tring meaning, create a figure
	#with this na,e
	if isinstance(axorfigname, str):
		fig = plt.figure(num=axorfigname)
		ax = fig.add_subplot(111)
	else:
		ax = axorfigname
		fig = ax.get_figure()
		
	#Get bincount from values to plot
	bincount = hist.shape[histaxis]
	
	#If refbincount is None, use just found bincount
	if refbincount is None:
		refbincount = bincount
	
	#If we have length 1 along hist axis, unpack. Use bincount maybe known
	#from previous call. That way, one can plot uint results with
	#known bincount or statistic results and let us find the bincount.
	if bincount == 1:
		hist = unpackHist(hist, bincount=refbincount, axis=histaxis)
		#This increases the bincount from 1 to some actual hist
		bincount = int(refbincount)
		
	#Get x axis values
	histvalues = packUnpackHistCommon(
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
	)
	
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	
	return fig, ax, hist, histplot, bincount
