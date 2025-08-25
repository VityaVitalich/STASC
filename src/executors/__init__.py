import importlib
import pkgutil

from executors.baseline import BaseExecutor
from executors.cove import CoveExecutor
from executors.stasc import STASCExecutor

__all__ = ["CoveExecutor", "BaseExecutor", "STASCExecutor"]


for _, module_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_name}")
