"""PSumSim: A Simulator for Partial-Sum Quantization in Analog Matrix-Vector
Multipliers

See `concepts` and included models."""

#This lib is a backport of importlib.resources
#Provides accessing package data files.
import importlib_resources as resources

#Read version and strip whitespacw
version = resources.files("psumsim") / "VERSION.txt"
version = version.read_text()
version = version.strip()
"""`str` : The package version number."""

__version__ = version
"""`str` : The package version number."""
