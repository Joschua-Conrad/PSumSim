# Readme

**PSumSim: A Simulator for Partial-Sum Quantization in Analog Matrix-Vector Multipliers**

This project includes a Python package useful for simulating partial-sum
quantization in matrix-vector multipliers (MVMs) using a histogram-based
data format.
Written at the Institute of Microelectronics, Ulm University, Ulm, Germany
by Joschua Conrad under the supervision of Prof. Maurits Ortmanns.
Published at ISCAS 2025.
Please acknowledge the license: {std:ref}`infolicense`.

(readmecite)=
## How to Cite
When using the code or concepts shown here for your own work, please cite
the published paper:

```bibtex
@inproceedings{Conrad_Confidence_Boosting_2024,
	author = {Conrad, Joschua and Wilhelmstätter, Simon and Mandry, Holger and Asthana, Rohan and Belagiannis, Vasileios and Ortmanns, Maurits},
	title = {PSumSim: A Simulator for Partial-Sum Quantization in Analog Matrix-Vector Multipliers},
	booktitle = {2025 IEEE International Symposium on Circuits and Systems {(ISCAS)}},
	eventtitle = {2025 IEEE International Symposium on Circuits and Systems {(ISCAS)}},
	publisher = {IEEE},
	location = {Londong, GB},
}
```

## What PSumSim Can Do

- Create operands for MVM applications as numbers drawn from a random process
  or as histograms ({any}`generateSimulationOperands`)
  
- Implement several numeric operations on histograms
	
	- SQNR computation ({any}`computeSqnr`)
	
	- Quantization and clipping ({any}`quantizeClipScaleValues`)
	
	- Summation ({any}`probabilisticAdder`)
	
	- Standard-deviation computation ({any}`getHistStddev`)
	
	- Make unquantized and quantized histograms comparable
	  ({any}`equalizeQuantizedUnquantized`)
	  
- Simulate full MVM applications using stochastic processes and histograms
  ({any}`simulateMvm`). One can thereby
  
	- Observe full-scale values ({std:ref}`maxhistvalue`)
	
	- Observe probability for the occurence of each possible result value
	  ({std:ref}`dataformat`)
	
	- Check a good ADC setup in terms of its full-scale ({std:ref}`clipping`)
	  and bitwidth ({std:ref}`quantization`)
	  
	- Check the effects of the algorithm being too large for the hardware
	  ({std:ref}`chunks`)
  
- Explore the design-space of a MVM application regarding algorithm and
  hardware size and SQNR with a highly parallel simulator
  ({any}`runAllExperiments`)
  
- Evaluate the oprimum clipping criterion (OCC) [OCC]_
  ({any}`optimumClippingCriterion`)

## TL;DR
Time is precious and you just want a one-liner to run in a Linux or MAC shell?
Download the code, open a commandline in the directory with downloaded files
and run
```bash
python3 -m pip install --upgrade pip && \
python3 -m pip install --upgrade setuptools virtualenv wheel && \
python3 -m virtualenv venv && \
. venv/bin/activate && \
python3 -m pip install --editable ".[docs,test]" && \
sphinx-build -b html -E doc/source doc/build && \
psumsim_test && \
deactivate
```

This installs everything, runs tests and builds the documentation website
in *docs/build*. You need Python3 to be installed already. Packages
*virtualenv*, *pip*, *setuptools* and *wheel* are installed or upgraded
system- or user-wide. But PSumSim and its dependencies are installed
isolated into a folder *venv*.

(readmeinstall)=
## How to Install
This provides a python package. Downloading all files and using
```bash
python3 -m pip install --editable .
```
in the same directory where this README is found installs the package and
allows all changes in source-files to reflect immediately.
Remove the `--editable` switch to install the package like any other Python
package, but you cannot change the provided *.py* files.

Also consider using a
[virtual environment](https://docs.python.org/3/library/venv.html) to install
packages into. This prevents any possible package-version clashes with things
you already have installed.

Also consider running
```bash
python3 -m pip install --upgrade pip setuptools wheel
```
to upgrade the Python package-managing before installing PSumSim.

## How to Use
After installing, there are several things provided by PSumSim.

### Docs
To build the documentation website, first install dependencies using
```bash
python3 -m pip install --editable ".[docs]"
```
and again possibly ommit `--editable` just like in
[Installation Guidelines](#how-to-install). Then, run
```bash
sphinx-build -b html -E doc/source doc/build
```
in the main directory. After the command completes, you'll find the documentation
as a website *docs/build/index.html*. Add *-W* to fail on warnings. This
README is also found there with all hyperlinks working.

### Start Reading
Checkout {std:ref}`concepts`, which summarizes some concepts and terms used
throughout the entire documentation.

### Commandline
Use
```bash
psumsim --help
```
to get an overview over the commandline interface. This is made for running
the same experiments as for the paper to get the scatter-plot data.
E.g. use
```bash
psumsim -j 4 -b 0.1 -e 0.2
```
to run 10% of all experiments with 4 CPU jobs. Several of these calls can be
used to keep the simulation machine busy.

See {std:ref}`commandlineinterface` for full documentation or simply call
`psumsim --help`. To run your own design-space exploration, check and
modify {any}`experiments`.

### As a Package
Check {any}`simulateMvm`, which basically is what is called for
each simulated experiment. Use this in your own Python script to get histograms
reflecting your own MVM application.

### Tests
To simply check that the installation worked, first install test dependencies by
running
```bash
python3 -m pip install --editable ".[test]"
```
and again possibly ommit `--editable` just like in
[Installation Guidelines](#how-to-install). Then, run `psumsim_test`.
Under the hood, *pytest* is used. Possible commandline arguments are descibed
[here](https://docs.pytest.org/en/stable/how-to/usage.html). Common usage
is to run a specific test case and exit immediately if that one fails,
as run e.g. by `psumsim_test -x -k "test_optimumClippingCriterion"` for the
test {any}`test_optimumClippingCriterion`.

(readmeacknowledge)=
## Acknowledgment
This work was funded by the *German National Science Foundation (DFG)* under
grant number *BE 7212/7-1 | OR 245/19-1*. Furthermore, the authors acknowledge
support by the state of Baden-Württemberg through bwHPC.

## ToDos for Publishing After Review

- [X] Update license
- [X] Fill [How to Cite](#how-to-cite)
- [ ] Add *date-published*, *doi*, *url* to *CITATION.cff*
- [ ] Add *date*, *doi*, *url*, *pages* to *CITATION.bib* and *README.md*
- [ ] Make repository public
- [X] Update repository link in manuscript
- [X] Remove *Information for Reviewers*
