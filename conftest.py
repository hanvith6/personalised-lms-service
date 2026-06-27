"""Make the project root importable for tests (so `from utils...` works)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
