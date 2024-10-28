import importlib_resources as resources

version = resources.read_text("psumsim", "VERSION.txt")
version = version.strip()
__version__ = version
