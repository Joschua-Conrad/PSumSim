import importlib_resources as resources

version = resources.files("psumsim") / "VERSION.txt"
version = version.read_text()
version = version.strip()
__version__ = version
