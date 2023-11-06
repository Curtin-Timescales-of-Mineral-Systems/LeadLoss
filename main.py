import sys
import os

# Get the absolute path of your project's root directory
project_root = os.path.abspath(os.path.dirname(__file__))

# Construct the path to the 'src' directory
src_path = os.path.join(project_root, 'src')

# Add 'src' to the Python path
sys.path.insert(0, src_path)
