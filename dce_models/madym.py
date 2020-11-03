import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import subprocess
import shutil
import dce_models.dce_aif as dce_aif


def run_madym_analysis(
    visit_dir,
    model_type:str,
    output_dir:str = 'mdm_analysis',
    T1_path:str = '', 
    FA_paths:list = [], 
    dynamic_path:str = '',
    roi_path:str = '',
    aif_path:str = '',
    pif_path:str = '',
    injection_image:str = '',
    relax_coeff:float = 3.4,
    noise_thresh:float = 20,
    command_exe = 'C:/isbe/code/obj_msvc2015/manchester_qbi/bin/Release/madym',
    overwrite:bool = True,
    extra_args:str = '',
    dummy_run:bool = False): 
    #Process system call to run madym

    command_args = (' -m {model_type} -o {output_dir}'
        .format(
            model_type = model_type,
            output_dir = output_dir + '_' + model_type))

    if len(T1_path):
        command_args += " -T1 " + T1_path
    
    elif len(FA_paths):
        command_args += " -vfa "
        for idx, FA_path in enumerate(FA_paths):
            if idx:
                command_args += ","
            command_args += FA_path
    
    if len(dynamic_path):
        command_args += " -dyn " + dynamic_path
    
    if len(roi_path):
        command_args += " -roi " + roi_path

    if len(aif_path):
        command_args += " -aif " + aif_path

    if len(pif_path):
        command_args += " -pif " + pif_path

    command_args = ('{command_args} -r1 {relax_coeff} -i {injection_image} -noise {noise_thresh}'
        .format(
            command_args = command_args,
            relax_coeff = relax_coeff,
            injection_image = injection_image,
            noise_thresh = noise_thresh))

    if overwrite:
        command_args += ' -overwrite'

    if len(extra_args):
        command_args += extra_args
    
    #command_args = ' -h'   
    print('Working directory: ', visit_dir)
    print('Command to run:', command_exe+command_args)
    if not dummy_run:       
        print('Running command...')
        cmd_list = (command_exe+command_args).split()
        print(cmd_list)
        p = subprocess.Popen(cmd_list,
            #stdout=sys.stdout, 
            #stderr=subprocess.PIPE,
            cwd=visit_dir)
        _, stderr = p.communicate()
        #print(stdout.decode("utf-8"))
        #print(stderr.decode("utf-8"))
        if p.returncode != 0:
            raise RuntimeError(stderr)
        # cmd_res = subprocess.check_output(command_exe+command_args,
        #     shell=True,
		# 	stdout=subprocess.PIPE,
		# 	stderr=subprocess.PIPE,
        #     cwd=visit_dir)#
        return p.returncode
