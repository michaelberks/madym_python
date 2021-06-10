'Tools for DCE-MRI analysis from the QBI lab, University of Manchester'

import os
v_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION'))
with open(v_file, 'r') as f:
    v = f.readlines()

__version__= v[1].strip()
__all__ = ['dce_models', 'image_io', 't1_mapping', 'tools', 'helpers']