#!/usr/bin/env python3

"""Behavior for running the tests via *psumsim_test* command.

See `mainTest`.

This module can be run as script.
"""

import os
import pathlib
import pytest
import sys

#And will launch pytest from bare module, because if we first import psumsim
#and hence progressbar and thereafter pytest, progressbar will crash, because
#it uses some old stdout which is disabled and redirected by pytest. But
#that needs to happen BEFORE importing progressbar.

def mainTest(argv=None, forwardsysexit=True):
	"""Invoke `pytest.main`.
	
	This adapts and then later restores the working directory using
	`os.chdir` to work in the repository directy some levels above the
	path of this sourcefile.
	
	This must be defined an an own separate module, which does not import
	:py:mod:`psumsim`. Otherwise, that already import the progressbar, that
	memorizes *stdout* file descriptors, `pytest` then makes them invalid to capture
	*stdout* and the progressbar crashes. So we here in this module only import
	`pytest`, `pytest.main` finds :py:mod:`psumsim` and imports it AFTER having
	*stdout* updated.
	
	.. note:
		Check `pytest.main`. This is not intended to be called multiple times
		from the same process.

	Parameters
	----------
	argv : (`list` of `str`), `None`, optional
		Commandline arguments. The default `None` will use `sys.argv`.
		
	forwardsysexit : `bool`, optional
		If set, the exitcode returned by `pytest.main` is forwarded to
		`sys.exit`. The whole program and the Python interpreter then end after
		the tests are done. If the tests fail, one can retreive success via
		exit code of *psumsim_test*. If not set, the exit code is returned
		to the caller. The default calls `sys.exit`.

	Returns
	-------
	exitcode : `int`
		If *forwardsysexit* did not call `sys.exit`, the exit code from
		`pytest.main` is returned here.

	"""
	
	#Will adapt cwd
	oldcwd = os.getcwd()
	pytestdir = pathlib.Path(__file__).parent.parent

	try:
		os.chdir(pytestdir)
		exitcode = pytest.main(args=argv)
	finally:
		#Restore cwd
		os.chdir(oldcwd)
		
	if forwardsysexit:
		sys.exit(exitcode)
		
	return exitcode


if __name__ == "__main__":
	#Our main is the unittest main, which makes this script support e.g.
	#plot names.
	mainTest()

	#Debug does not catch any exceptions. Works better with post mortem debugging
	#testcase = test_Psum_Quantization(methodName="test_debugSimulation")
	#testcase = test_Psum_Quantization(methodName="test_statStocComparisonPlot")
	#testcase = test_Psum_Quantization(methodName="test_unquantQuantComparisonPlot")
	#testcase = test_Psum_Quantization(methodName="test_optimumClippingCriterion")
	#testcase = test_Psum_Quantization(methodName="test_runAllExperiments")
	#testcase = test_Psum_Quantization(methodName="test_runDescription")
	#testcase = test_Psum_Quantization(methodName="test_stimulusPlot")
	#testsuite = ut.TestSuite(tests=(testcase,))
	#testsuite.debug()
