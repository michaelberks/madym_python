'''
This script is designed to be run over the output from a call to Madym
to convert any generated NIFTI output maps to DICOM images
'''
#%%
import os
import glob

from QbiPy.image_io.nifti2dicom import convert_nifti_dir

'''
What do we need?

For T1 output:

From dicom_convert dir:
    - Index of T1 input series
    - Flip x/y/z

From dicom_sort config:
    - Location/name of sort files

From T1 config:
    - Location of T1 output

'''

def get_latest_output_config(output_dir):
    config_files = [
        f for f in glob.glob(os.path.join(output_dir, 'madym*_config.txt'))
         if "override" not in f]
    return config_files[-1]

def parse_list_value(list_str):
    if len(list_str) == 2:
        values = []

    else:
        values = [parse_single_value(v.rstrip().lstrip()) for v in 
            list_str[1:-1].split(',')]
    return values

def parse_single_value(value_str):
    if value_str[0] == '"':
        value = ''

    else:
        try:
            value = float(value_str)
        except ValueError:
            value = value_str
    return value


def parse_value(value_str):
    if not value_str:
        value = ''

    elif value_str[0] == '[':
        value = parse_list_value(value_str)

    else:
        value = parse_single_value(value_str)

    return value

def read_madym_config(config_file):
    options = dict()

    with open(config_file, 'rt') as cf:
        lines = cf.readlines()

    for line in lines[1:]:
        parts = line.split('=')
        option = parts[0].rstrip().lstrip()
        value_str = parts[1].rstrip().lstrip()
        options[option] = parse_value(value_str)

    return options

def get_madym_options(output_dir):
    config_file = get_latest_output_config(output_dir)
    options = read_madym_config(config_file)
    return options

def get_series_index(series_idx):
    if type(series_idx) == str:
        return int(series_idx.split('-')[0])
    if type(series_idx) == list:
        return int(series_idx[0].split('-')[0])
    else:
        return int(series_idx)

#%%
def madym_output_to_dicom(
    dicom_sort_dir, dicom_convert_dir, output_dir, series_type,
    voxel_min = -0.1,
    voxel_max = 1e6):
    
    #Get sort and convert options
    sort_options = get_madym_options(dicom_sort_dir)
    convert_options = get_madym_options(dicom_convert_dir)

    #Get index of series to use as template dicom
    series_idx = get_series_index(convert_options[series_type + '_series'])

    #Locate series filenames file and load first filename
    dcm_series = sort_options['dicom_series']
    series_file = os.path.join(dicom_sort_dir, 
        f'{dcm_series}_series{series_idx}_filenames.txt')

    with open(series_file, 'rt') as sf:
        dicom_name = sf.readline().rstrip()

    #Call nift convert on the output folder
    series_number = convert_nifti_dir(
        output_dir, 
        dicom_name,
        time_series=False,
        flip_x = convert_options['flip_x'],
        flip_y = convert_options['flip_y'], 
        flip_z = convert_options['flip_z'],
        voxel_min = voxel_min, voxel_max = voxel_max)

    #If it's dynamic output, check for Ct_sig and Ct_mod outputs
    if series_type == 'dyn':
        for Ct in ['sig', 'mod']:
            Ct_dir = os.path.join(output_dir, 'Ct_' + Ct)
            if os.path.exists(Ct_dir):
                series_number += 1
                convert_nifti_dir(
                    Ct_dir, 
                    dicom_name, 
                    series_start = series_number,
                    time_series=True,
                    flip_x = convert_options['flip_x'],
                    flip_y = convert_options['flip_y'], 
                    flip_z = convert_options['flip_z'],
                    voxel_min = voxel_min, voxel_max = voxel_max)
    
if __name__ == "__main__":
    
    import sys
    if len(sys.argv) < 5:
        raise RuntimeError("Expected at least 4 input args")
    dicom_sort_dir = sys.argv[1]
    dicom_convert_dir = sys.argv[2]
    output_dir = sys.argv[3]
    series_type = sys.argv[4]
    print('Running madym_output_to_dicom with args:')
    print(f'\tdicom_sort_dir = {dicom_sort_dir}')
    print(f'\tdicom_convert_dir = {dicom_convert_dir}')
    print(f'\toutput_dir = {output_dir}')
    print(f'\tseries_type = {series_type}')

    madym_output_to_dicom(
        dicom_sort_dir, dicom_convert_dir, output_dir, series_type
    )