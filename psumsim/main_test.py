import os
import pathlib
import pytest
import sys

#And will launch pytest from bare module, because if we first import psumsim
#and hence progressbar and thereafter pytest, progressbar will crash, because
#it uses some old stdout which is disabled and redirected by pytest. But
#that needs to happen BEFORE importing progressbar.

def mainTest(argv=None, forwardsysexit=True):
	
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
