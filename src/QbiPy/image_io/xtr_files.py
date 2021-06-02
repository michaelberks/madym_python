''' Functions for reading and writing the xtr files used by madym to encode information
    not encoding in Analyze format image headers
# Created: 08-Jan-2019
# Author: Michael Berks 
# Email : michael.berks@manchester.ac.uk 
# Phone : +44 (0)161 275 7669 
# Copyright: (C) University of Manchester'''

import math
def write_xtr_file(xtr_path, append=False, **kwargs):
    '''WRITE_XTR_FILE write a text file of name/value pairs, as used in Madym to
    #specify additional information not contained in Analyze75 img headers (eg.
    #scan flip angle, TR etc)
    #   [] = write_xtr_file(xtr_path, append, **kwargs)
    #
    # Inputs:
    #      xtr_path - path to write xtr file (typically with extension .xtr)
    #
    #      append - If file already exists, append or overwrite
    #
    #      kwargs - List of fieldname/value pairs
    #
    #
    # Outputs:
    #
    # Example: write_xtr_file('temp.xtr', 'TR', 2.4, 'FlipAngle', 20.0, 'TimeStamp', 12345)
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
    '''WRITE_XTR_FILE write a text file of name/value pairs, as used in Madym to
    #specify additional information not contained in Analyze75 img headers (eg.
    #scan flip angle, TR etc)
    #   [] = write_xtr_file(xtr_path, append, **kwargs)
    #
    # Inputs:
    #      xtr_path - path to write xtr file (typically with extension .xtr)
    #
    #      append - If file already exists, append or overwrite
    #
    #      kwargs - List of fieldname/value pairs
    #
    #
    # Outputs:
    #
    # Example: write_xtr_file('temp.xtr', 'TR', 2.4, 'FlipAngle', 20.0, 'TimeStamp', 12345)
    #
    # Notes:
    '''
    #Open a file identifier, either in write or append mode
    with open(xtr_path, 'r') as xtr_file: 
        pass

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
        


