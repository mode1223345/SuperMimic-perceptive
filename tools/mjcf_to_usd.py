"""Convert the G1 MJCF to USD with the standalone Isaac Sim MJCF importer (no Kit)."""
import os
import sys

from pxr import Plug

here = os.path.dirname(os.path.abspath(__file__))
import isaacsim.asset.importer.utils.impl as _iu

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

src = os.path.join(here, "..", "data", "g1_description", "g1_29dof.xml")
dst = os.path.join(here, "..", "data", "g1_usd")
cfg = MJCFImporterConfig(mjcf_path=os.path.normpath(src), usd_path=os.path.normpath(dst))
print(MJCFImporter(cfg).import_mjcf())
