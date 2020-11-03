#------------------------------------------------------------------------------------
# Created: 01-May-2019
# Author: Michael Berks 
# Email : michael.berks@manchester.ac.uk 
# Phone : +44 (0)161 275 7669 
# Copyright: (C) University of Manchester 
#-------------------------------------------------------------------------------------
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import platform
import os
from shutil import copyfile
from madym_interface import utils
from madym_interface.run_madym_tests import run_madym_tests

def install_madym(
    qbi_share_path=None, operating_system=None, 
    madym_root=None, version=None, run_tests=1, plot_test_output=True):
    ''' 
    INSTALL_MADYM install latest version of Madym C++ tools to your local
    machine. REQUIRES that you have set your local madym root first (see
    below)
       [] = install_madym(qbi_share_path)
    
     Inputs:
          qbi_share_path (str) - path to the root of the shared QBI drive (eg Q:\)
    
          operating_system (str) - must match {'Windows', 'Ubuntu', MacOs'}
    
          madym_root (str) - local directory where madym tools will be copied
    
          version (str - []) - version to install, if empty will install
          latest version. If set, must match a version on the shared drive
    
          run tests (int - 1) - flag to run tests on successful install. Set 0
          not to run tests. Values >1 run advanced tests (currently not
          implemented)
    
          plot_test_output (Bool) - if true, show test plots. Set to false to run non-interactively
          eg for installing on CSF
    
     Outputs: None
    
     Example: install_madym('Q:', 'Windows')
              install_madym('Volumes/qbi', 'MacOs')
    
     Notes:
    ''' 
    if qbi_share_path is None:
        root = tk.Tk()
        root.overrideredirect(1)
        root.withdraw()
        messagebox.showinfo(title='Step 1: locate shared drive', message='Use finder to select root of QBI share drive')
        qbi_share_path = filedialog.askdirectory(
            title='Find shared drive'
        )
        root.destroy()
        if not qbi_share_path:
            return        

    if operating_system is None:
        operating_system = platform.system()

    if operating_system == 'Darwin':
        operating_system = 'MacOS'
        
    print(f'Operating system is {operating_system}')
    if operating_system not in ['Windows', 'Linux', 'MacOS']:
        raise ValueError(
            f'Operating system {operating_system} not recognised. ' 
            'Must match Windows, Linux or MacOS')

    if madym_root is None:
        madym_root = utils.local_madym_root(False)  

    #Set the madym root - if it's empty above, the set function will ask the
    #user to choose it.
    if not madym_root:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title='Step 2: set Madym root', message='Use finder to choose where madym tools will be installed')
        root.destroy()
    utils.set_madym_root(madym_root)

    #Finally, call local_madym_root again, as we dont know if in the set
    #function above, the user may have chosen not to overwrite an existing
    #madym root
    madym_root = utils.local_madym_root(False)

    if version is None:
        version = utils.latest_madym_version(qbi_share_path)
    
    if not version:
        raise ValueError(
            f'Could not retrieve madym version from the share drive.'
            ' Check the shared paths: {qbi_share_path}.')   

    #Make path to the latest madym version and get file list
    madym_path = os.path.join(
        qbi_share_path,
        'software',
        'madym_versions',
        version,
        operating_system)

    #Copy files
    os.makedirs(madym_root, exist_ok=True)
    for entry in os.scandir(madym_path):
        if entry.is_file():           
            copyfile(
                entry.path,
                os.path.join(madym_root, entry.name))
            print(f'Copied {entry.name} from {madym_path} to {madym_root}')   

    #Check version now matches
    local_version = utils.local_madym_version()

    if local_version == version:
        print(f'*****************************************************')
        print(f'Successfully installed version {version} on this machine')
        print(f'*****************************************************')
    else:
        print(f'*****************************************************')
        print(f'Problem installing: local version returned {local_version}, '
            'this suggests version {version} from the share drive did not install')
        print(f'*****************************************************')   

    #Run tests if set
    if run_tests:
        run_madym_tests(test_level=run_tests, plot_output=plot_test_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Install madym cxx tools from QBI server')
    
    parser.add_argument('-Q', dest='qbi_share_path', default=None, type=str, 
        help='path to the root of the shared QBI drive (eg Q:\\).')
    
    parser.add_argument('-os', dest='operating_system', 
        default=platform.system(), type=str, 
        choices=['Windows', 'Linux', 'MacOs'],
        help='operating system must match one of [Windows, Linux, MacOs].')

    parser.add_argument('-l', dest='madym_root', default=None, type=str, 
        help='local directory where madym tools will be copied.')

    parser.add_argument('-v', dest='version', default=None, type=str, 
        help='version to install, if empty will install'
            'latest version. If set, must match a version on the shared drive.')

    parser.add_argument('-t', dest='run_tests', default=1, type=int, 
        help='flag to run tests on successful install.'
        ' Set 0 not to run tests.'
        ' Values >1 run advanced tests (not yet implemented).')
    options = parser.parse_args()
    install_madym(
        qbi_share_path=options.qbi_share_path, 
        operating_system=options.operating_system, 
        madym_root=options.madym_root, 
        version=options.version, 
        run_tests=options.run_tests)