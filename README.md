# PSumSim: A Simulator for Partial-Sum Quantization in Analog Matrix-Vector Multipliers
This project includes a Python package useful for simulating partial-sum
quantization in matrix-vector multipliers (MVMs) using a histogram-based
data format.

## Information for Reviewers
In case of acceptance of the provided manuscript, this repository is made available
publicly under an open-source license. Currently, it is solely provided
for the purpose of reviewing the manuscript at-hand. See [license](LICENSE.md).

## How to Install
This provides a python package. Using
```bash
python3 -m pip install --editable .
```
installs the package and allows all changes in sourcefiles to reflect immediately.
Remove the `--editable` switch to install the package as any other Python
package.

## How to Use
After installing, there are several things provided by PSumSim.

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

### As a Package
Check `psumsim.simulation.simulateMvm`, which basically is what is called for
each simulated experiment. Use this in your own Python script to get histograms
reflecting your own MVM application.

### Tests
To simply check that the installation worked, run `psumsim_test`.

## Acknowledgment
This work was funded by the *German National Science Foundation (DFG)* under
grant number *BE 7212/7-1 | OR 245/19-1*. Furthermore, the authors acknowledge
support by the state of Baden-Württemberg through bwHPC.

## ToDos for Publishing After Review

- [ ] Update license
- [ ] Add how-to-cite
- [ ] Make repository public
- [ ] Update repository link in manuscript
- [ ] Remove [Information for Reviewers](#information-for-reviewers)
