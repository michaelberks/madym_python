''' Functions for reading and writing the xtr files used by madym to encode information not stored in Analyze format image headers (eg flip angle and TR)

Created: 08-Jan-2019
Author: Michael Berks 
Email : michael.berks@manchester.ac.uk 
Phone : +44 (0)161 275 7669 
Copyright: (C) University of Manchester
'''

import math
import re

def write_xtr_file(xtr_path, append=False, **kwargs):
    '''WRITE_XTR_FILE write a text file of name/value pairs, as used in Madym to
    #specify additional information not contained in Analyze75 img headers (eg.
    #scan flip angle, TR etc)
    #   [] = write_xtr_file(xtr_path, append, **kwargs)
    #
    # Parameters:
    #      xtr_path - path to write xtr file (typically with extension .xtr)
    #
    #      append - If file already exists, append or overwrite
    #
    #      kwargs - List of fieldname/value pairs
    #
    #
    # Returns:
    #
    # Example: write_xtr_file('temp.xtr', False, TR=2.4, FlipAngle=20.0,    TimeStamp=12345)
    #
    '''

    #Open a file identifier, either in write or append mode
    if append:
        mode = 'a'
    else:
        mode = 'w'

    with open(xtr_path, mode) as xtr_file: 
        #Write out each name/value pair
        for key, value in kwargs.items():
            print(key, file=xtr_file, end=' ')
            
            #Check if value is scalar or list, tuple, array etc
            try:
                len(value) #If scalar throws error, switches to except block
                for v in value:
                    print(v, file=xtr_file, end=' ')
            except:
                print(value, file=xtr_file, end='')

            print('', file=xtr_file)

def read_xtr_file(xtr_path, append=False, **kwargs):
    '''
    XTR files are simple text files used by the QBI lab to store extra
    data not included in analyze header files

    Parameters:
        xtr_path : str
            Path to xtr file to read

    Returns:
        xtr_data : dictionary 
            Dictionary of field/value pairs 
    '''
    #Open file and read in each line
    xtr_lines = []
    with open(xtr_path, "rt") as xtr_file:
        for line in xtr_file:
            xtr_lines.append(line.strip('\n'))

    #Define local auxiliary functions to parse old and new formats
    ##----------------------------------------------------------------
    def parse_old_fields(xtr_line):
        '''
        '''
        field,values = xtr_line.split(':')
        field = field.split(' ')[0]
        values = values.split(' ')

        if field == 'voxel': #dimensions
            field_list = ['VoxelDimensionsX','VoxelDimensionsY','VoxelDimensionsZ'] 
            value_list = [float(v) for v in values[0:4]]
        elif field == 'flip': #angle
            field_list = ['FlipAngle']
            value_list = [float(values[0])]
        elif field == 'TR':
            field_list = [field]
            value_list = [float(values[0])]
        elif field == 'timestamp':
            field_list = [field]
            value_list = [float(values[3])]              
        else:
            warnings.warn(f'Field name {field} not recognised.')
            field_list = [field]
            value_list = [float(values[0])]
        return field_list, value_list

    ##----------------------------------------------------------------
    def read_old_xtr(xtr_lines):
        '''
        More of a pain, we have a fixed format, with an annoying timestamp
        '''
        xtr_data = dict()
        for xtr_line in xtr_lines:
            field_list, value_list = parse_old_fields(xtr_line)            
            for field,value in zip(field_list, value_list):
                xtr_data[field] = value
        return xtr_data

    ##----------------------------------------------------------
    def read_new_xtr(xtr_lines):
        '''The new format is easier as everything is a name/value pair. 
        Just need to convert the values to numbers'''
        xtr_data = dict()
        for xtr_line in xtr_lines:
            #Split line on tabs or white space characters
            field, value = re.split('\s|\t', xtr_line)
            xtr_data[field] = float(value)
        
        return xtr_data
    #--------------------------------------------------------
    
    #Check if new or old format, then call appropriate reader
    if xtr_lines[0].split(' ')[0] == "voxel":
        xtr_data = read_old_xtr(xtr_lines)
    else:
        xtr_data = read_new_xtr(xtr_lines)
    
    return xtr_data

def secs_to_timestamp(t_in_secs):
    '''Convert time in seconds into the xtr timestamp format
    hhmmss.msecs represented as a single decimal number
    '''
    hh = math.floor(t_in_secs / (3600))
    mm = math.floor((t_in_secs - 3600*hh) / 60)
    ss = t_in_secs - 3600*hh - 60*mm
    timestamp = 10000*hh + 100*mm + ss
    return timestamp

def mins_to_timestamp(t_in_mins):
    '''Convert time in minutes (the form used for dynamic time in madym) 
    into the xtr timestamp format
    hhmmss.msecs represented as a single decimal number
    '''
    return secs_to_timestamp(60*t_in_mins)
        


