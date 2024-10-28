#!/usr/bin/env python3

"""Behavior for running the package es script. See `main`.

Use commandline and help switch or `commandlineinterface` for information on
usage.
"""

from psumsim.experiments import main
import multiprocessing

if __name__ == "__main__":
	
	#Needed to make multiprocess with pyinstaller work
	#https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
	multiprocessing.freeze_support()
	main()
