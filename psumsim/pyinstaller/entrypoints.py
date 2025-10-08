"""Define *pyproject.toml* entrypoints.

See `hookexample <https://github.com/pyinstaller/hooksample/blob/master/src/pyi_hooksample/__pyinstaller/__init__.py>`_.
"""

import pathlib

def getHookDirs():
	"""Return pyinstaller dir as search path for hooks.
	
	Returns
	-------
	`list` of single `str`
		This module path as new hook search path.
	"""
	
	return [str(pathlib.Path(__file__).parent.resolve()),]
