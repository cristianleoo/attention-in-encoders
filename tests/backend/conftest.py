import sys
import os

# Make `app.backend.main` importable from the tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
