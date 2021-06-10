'''
Tools for DCE-MRI analysis from the QBI lab, University of Manchester

Notes:
The version is read directly from the latest git repo tag, which is itself
set auto-magically by semantic-release in the project's GitLab CI/CD pipeline
'''

import git
repo = git.Repo('.')

__version__= str(repo.tags[-1]).split('v')[1]
__all__ = ['dce_models', 'image_io', 't1_mapping', 'tools', 'helpers']