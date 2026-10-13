"""All scripts read and write data under DATA (default: the package's data/ folder; override with the SHAPE_DATA environment variable)."""
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SHAPE_DATA", os.path.join(ROOT, "data"))
