"""The actual hook file like on `GitHub example <https://github.com/pyinstaller/hooksample/blob/master/src/pyi_hooksample/__pyinstaller/hook-pyi_hooksample.py>`_.

Imports of *Pyinstaller* are not needed when just running `psumsim`, because
import of this file is dangling and only needed when using the PyInstaller
entrypoint in *pyproject.toml*.

Do not add data files here, rather add them in *pyproject.toml* as datafiles of the
package.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = [
		#Include e.g. VERSION.txt as data file for pyinstaller.
		#Disvocer such files through pyproject.toml
		*collect_data_files(package="psumsim"),
]
"""`list` of `tuple` of `str` : Additional data files.

Add all data-files defined in *pyproject.toml* as data-files to be regarded by
*pyinstaller*.
"""

hiddenimports = [
		'scipy.stats',
]
"""`list of `str` : All hidden imports which are not found by PyInstaller.

`rand.sinusoidal_gen` here causes trouble. The need of `scipy.stats` is
not detected automatically."""
