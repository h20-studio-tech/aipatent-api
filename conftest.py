import os
import sys

# Determine the project root (assumes conftest.py is in the tests folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)