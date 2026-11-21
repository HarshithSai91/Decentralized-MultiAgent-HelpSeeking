"""Make ``backend/src`` importable and run every script from the project root."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "backend", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.chdir(ROOT)
