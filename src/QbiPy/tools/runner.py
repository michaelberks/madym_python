'''
This module provides a class QbiRunner that can be used to wrap any python
function, so that the function can be called from the command line, reading
inputs from a config file or environment variables, and generating program log
files to an output folder with minimal extra code required.

It is primarily designed for functions that read data from a given folder (and 
sub-folders), then process that data to generate results saved in the output
folder, but it can be run with any function.

For example, consider a simple function:

def print_a_plus_b(var_a, var_b):
    print(a, b)

By creating a new QbiRunner object, we can call this function:

args = ['a', '2', 'b', '3']

runner = QbiRunner('')
runner.parser.add('--var_a', type=int)
runner.parser.add('--var_b', type=int)
runner.run(print_a_plus_b, args)

This will run and create the following objects:

- an audit log folder `audit_logs` in the current directory,
containing an audit log listing the function called, the user,
the host and the location of the program log.

- an output folder, `print_a_plus_b_output` containing a program log
that contains stdout and stderr from the function call, so in this case
reads 3. The output folder will also contain a config file listing
all options the function was run with.

If the above snippet is placed in a script in the main block,
the args will be read from the command line. So for example if
we create a file run_print_a_plus_b.py, containing:

from QbiPy.tools.runner import QbiRunner

def print_a_plus_b(var_a, var_b):
    print(a, b)

if __name__ == "__main__":
    runner = QbiRunner('')
    runner.parser.add('--var_a', type=int)
    runner.parser.add('--var_b', type=int)
    runner.run(print_a, args)

We can now call this at the command line like:
> python print_a_plus_b --var_a 2 --var_b 3

The QbiRunner class automatically deals with config files, so we
could create a simple file config.txt containing:
var_a = 2
var_b = 3

And now call:
> python print_a_plus_b --config config.txt

We can also call for help in the usual way:
> python print_a_plus_b -h
> python print_a_plus_b --help

Any additional arguments to the function can be added in the same way,
by calling runner.parser.add.

Note the parser here is an ArgParser from the configargparse package 
(https://pypi.org/project/ConfigArgParse/), for which the add function
has identical behaviour to python's in-built argparse class.

By using configargparse, these options can be set either at the command
line, via an environment variable or in a config file. Options are
set with the following precendence:
command line > environment variables > config file values > defaults

# In-built options in the runner class

In addition to those set manually, by calling the function through 
the runner, we automatically have access to the following options:

--config (str, None) to call a config file as in the above example. The config file
will be read using configargparse (https://pypi.org/project/ConfigArgParse/)

--data_dir (str, None) path to a folder from which the output folder is relative. If
not set, defaults to the current working directory

--output_dir (str, None) relative path to the output folder from data_dir. If not set
defaults to the function name suffixed with `_output`

--overwrite (bool, True) if set to false, an error is returned if the output
folder already exists
        
--program_log (str, '_program_log') name of the program log, 
will be appended to the function name and suffixed with a timestamp

--audit_dir (str, 'audit_logs') directory of the audit log, either absolute or 
relative to the cwd

--audit_log (str, '_audit_log') name of the audit log, will be appended 
to the function name and suffixed with a timestamp

--config_log (str, '_config_log') name of the output config log, will be appended 
to the function name and suffixed with a timestamp
        
--no_log (bool, False), flag to turn off program logging. 
stdout and stderr will be piped to the screen as normal

--no_audit (bool, False) flag to turn off audit logging

--error_exit_code (int, 1) value to return if the function raises
an exception. The exception will be caught and printed to the output
logs. On some systems (eg XNAT) it is preferable for python to still
return a zero exit code so all output handlers run correctly, in which
case error_exit_code can be set to 0

Note the function you pass to the runner is welcome to re-use any of
the above options. Indeed it is anticipated it will make use of 
data_dir (to load input data) and output_dir (to write out results).
However it should not attempt to redefine arguments of the same name
by adding them to the parser.

In other words, do NOT do:

    runner.parser.add('--output_dir',...)

# Handling boolean inputs

Note to handle bools intuitively at the command line or in config files,
we recommend importing bool_option and using this as the type when calling
parser.add. For example

from QbiPy.tools.runner import QbiRunner, bool_option

runner = QbiRunner('')
runner.parser.add('--my_flag', type=bool_option)

my_flag will be created as a bool such that:

    - 'yes', 'true', 't', 'y', '1' convert to True

    - 'no', 'false', 'f', 'n', '0' convert to False

These conversions are case insensitive.

This is particularly helpful for being able to set flags that default
to true as false in a command file.

Of course, you are welcome to write your own string to bool function
and pass this to parser.add instead.


'''
import os
import sys
import socket
import getpass
import inspect
import traceback

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import configargparse

#----------------------------------------------
class QbiRunner():
    def __init__(self, description = None):
        '''Initialise QbiRunner class, creating the ArgParser member and setting
        generic input options

        Parameters
        ----------
        description : _type_, optional
            _description_, by default None
        '''
        self.description_ = description
        self.parser = configargparse.ArgParser(description)
        self.set_generic_options()
    
    #---------------------------------------------
    def set_generic_options(self):
        '''Define set of generic options that all function that use the runner
        will inherit
        '''
        #Config file
        self.parser.add('--config', required=False, is_config_file=True,
            type=str, nargs='?', default=None,
            help='Path to config file')

        #Ouput folder
        self.parser.add('--data_dir', type=str, nargs='?', default=None,
            help='Path to the data')

        #Ouput folder
        self.parser.add('--output_dir', type=str, nargs='?', default=None,
            help='Relative path to the output folder')

        #Overwrite flag
        self.parser.add('--overwrite', type=bool_option, nargs='?', default=True, const = True,
            help='Flag to allow overwriting previous output')

        #Archive existing work
        self.parser.add('--archive_dir', type=str, nargs='?', default=None,
            help='Name to archive existing output folder. If set to timestamp, will append a timestamp to current output')
        
        #Name of program log
        self.parser.add('--program_log', type=str, nargs='?', default='_program_log',
            help='Name of the program log, will be appended with timestamp')

        #Name of audit log
        self.parser.add('--audit_dir', type=str, nargs='?', default='audit_logs',
            help='Directory of the audit log, either absolute or relative to the cwd')

        #Name of audit log
        self.parser.add('--audit_log', type=str, nargs='?', default='_audit_log',
            help='Name of the audit log, will be appended with timestamp')

        #Name of config log
        self.parser.add('--config_log', type=str, nargs='?', default='_config_log',
            help='Name of the config log, will be appended with timestamp')
        
        #Program log flag
        self.parser.add('--no_log', type=bool_option, nargs='?', default=False, const = False,
            help='Flag to turn off program logging. Info is printed to stdout')

        #Audit log flag
        self.parser.add('--no_audit', type=bool_option, nargs='?', default=False, const = False,
            help='Flag to turn off audit logging.')

        #Audit log flag
        self.parser.add('--error_exit_code', type=int, nargs='?', default=1, const = False,
            help='Code returned on error.')
        
    #----------------------------------------------
    def parse_args(self, args = None):
        '''Parse args from command-line and/or config file

        Parameters
        ----------
        args : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        '''
        if args is None:
            args = sys.argv[1:]
        return self.parser.parse_args(args)

    #-----------------------------------------------
    def run(self, fun, args = None):
        '''Run the given function fun inside the runner, setting up an output
        directory and program/audit logs

        Parameters
        ----------
        fun : _type_
            _description_
        args : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        '''
        #Parse args (if none set, this will use sys.argv)
        options = self.parse_args(args)
        
        #Set data directory to current working directory if not specified in options
        if options.data_dir is None:
            options.data_dir = os.getcwd()

        #Set up output folder and program log
        if options.output_dir is None:
            options.output_dir = fun.__name__ + '_output'
        options.output_dir = os.path.join(options.data_dir, options.output_dir)
        os.makedirs(options.output_dir, exist_ok = options.overwrite)

        #Get list of files in output dir for archiving
        options.output_list = os.listdir(options.output_dir)

        #Get function args from options namespace
        args = get_function_args(fun, options)

        #If user set no log, then just run the function
        if options.no_log:
            success = run_catch(options, args, fun)
        else:
            success = run_with_logging(options, args, fun)

        #If the function ran without raising an exception, return 0,
        #otherwise, return the value set in options.error_exit_code
        if success:
            exit_code = 0
        else:
            exit_code = options.error_exit_code

        return exit_code

#-----------------------------------------------
def run_catch(options, args, fun):
    '''Trying running fun with input **args. If an exception is raised,
    catch it, print it and return False.

    If the function runs to completion, return true.

    _extended_summary_

    Parameters
    ----------
    options : _type_
        _description_
    args : _type_
        _description_
    fun : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    #Do archiving here, so it get logged correctly
    archive_existing_output(
        options.data_dir, options.archive_dir, options.output_dir, options.output_list)

    try:
        fun(**args)
        success = True
    except Exception: 
        print(traceback.format_exc())
        success = False

    return success

#-----------------------------------------------
def run_with_logging(options, args, fun):
    '''Run the function fun with inputs **args, creating program, config
    and audit logs as specified by the options.

    If a program log is created (ie. options.no_log = False) then
    all stderr and stdout will be redirected to the log.

    fun will be called via run_catch, so any exceptions raised will be
    caught and logged. This function will return True if fun ran to
    completion and False if an exception was caught

    Parameters
    ----------
    options : _type_
        _description_
    args : _type_
        _description_
    fun : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    #Need to set-up logging info

    #Get current datetime
    date_str = datetime.today().strftime('%Y%m%d_%H%M%S')
    
    #Create timestamped paths to log files
    program_log_path = os.path.join(
        options.output_dir, fun.__name__ + options.program_log + date_str + '.txt')
    config_log_path = os.path.join(
        options.output_dir, fun.__name__ + options.config_log + date_str + '.txt')

    #Initialise audit log if not flagged otherwise
    do_audit = not options.no_audit
    if do_audit:
        audit_log_path = os.path.join(
            options.audit_dir, fun.__name__ + options.audit_log + date_str + '.txt')
        initialise_audit_log(audit_log_path, program_log_path)

    #Open program log file
    with open(program_log_path, 'wt') as program_log:

        #Initialise program log and write output config log
        initialise_log(program_log)
        write_config_log(options, config_log_path)
        
        #Run the function within the context of stdout and stderr being
        #redirected to the program log
        with redirect_stderr(program_log), redirect_stdout(program_log):
            success = run_catch(options, args, fun)

        #Finalise the program log
        finalise_log(program_log, success)

    #Finalise the audit log
    if do_audit:
        finalise_audit_log(audit_log_path, success)

    return success

#-----------------------------------------------
def archive_existing_output(data_dir, archive_dir, output_dir, output_list):
    '''Move any existing files and folders inside output_dir to an archive_dir

    Parameters
    ----------
    data_dir : _type_
        _description_
    archive_dir : _type_
        _description_
    output_dir : _type_
        _description_
    output_list : _type_
        _description_
    '''
    if archive_dir is None or not output_list:
        return

    if archive_dir == 'timestamp':
        archive_dir = os.path.join(
            output_dir, 
            datetime.today().strftime('_archive%Y%m%d_%H%M%S'))
    else:
        archive_dir = os.path.join(data_dir, archive_dir)

    print('Output folder exists and archiving is on: '
            f'moving {data_dir} to {archive_dir}')
    os.makedirs(archive_dir, exist_ok=False)
    for obj_name in output_list:
        if not obj_name.lower().startswith('_archive'):
            os.rename(
                os.path.join(output_dir, obj_name), 
                os.path.join(archive_dir, obj_name))
#-----------------------------------------------
def initialise_audit_log(audit_log_path, program_log_path):
    '''Create a new audit log at the specified path

    Parameters
    ----------
    audit_log_path : _type_
        _description_
    program_log_path : _type_
        _description_
    '''
    os.makedirs(os.path.dirname(audit_log_path), exist_ok = True)
    with open(audit_log_path, 'wt') as log:
        initialise_log(log)
        print(f'Program log saved to {program_log_path}', file=log)

#-----------------------------------------------
def finalise_audit_log(audit_log_path, success):
    '''Finalise the audit log

    Parameters
    ----------
    audit_log_path : _type_
        _description_
    success : _type_
        _description_
    '''
    with open(audit_log_path, 'at') as log:
        finalise_log(log, success)

#-----------------------------------------------
def finalise_log(log, success):  
    '''Finalise an open audit/program log with file object log,
    logging the time of completion and whether the function
    exited successfully of raised an exception

    Parameters
    ----------
    log : _type_
        _description_
    success : _type_
        _description_
    '''
    if success:
        print(f'{sys.argv[0]} completed successfully.', file=log)
    else:
        print(f'{sys.argv[0]} exited with errors. Check logs.', file=log)

    date_str = datetime.today().strftime('%Y%m%d %H:%M:%S')
    print(f'Log closed at {date_str}', file=log)

#-----------------------------------------------
def initialise_log(log):
    '''Initialise an open audit/program log with file object log,
    logging the start time, user, host and command-line args

    Parameters
    ----------
    log : _type_
        _description_
    '''
    date_str = datetime.today().strftime('%Y%m%d %H:%M:%S')
    print(f'Log opened at {date_str}', file=log)
    print(f'User: {getpass.getuser()};   Host: {socket.gethostname()}', file=log)
    print(f'Ran in: {os.getcwd()}', file=log)
    command_args = ' '.join(sys.argv)
    print(f'Command args: {command_args}', file=log)

#-----------------------------------------------
def write_config_log(options, config_log_path):
    '''Write the complete set of options (formed from a combination
    of command-line, environment variable, config file and defaults)
    to an output file.

    Parameters
    ----------
    options : _type_
        _description_
    config_log_path : _type_
        _description_
    '''
    with open(config_log_path, 'wt') as log:
        for option, value in options.__dict__.items():
            print(f'{option} = {value}', file = log)

#-----------------------------------------------
def get_function_args(fun, options):
    '''Iterate through attributes in a namepsace object and return attributes
    included in the signature of the given function as a dictionary

    Parameters
    ----------
    fun : function
        function object
    options : namespace
        Simple namepsace object returned from argparse

    Returns
    -------
    args_dict
        dictionary of attribute/values from options where
        an attribute is included iff it is a member of fun's signature
    '''
    #Get list of arg names from the given function signature
    fun_arg_names = list(inspect.signature(fun).parameters)

    #Set-up empty dictionary to store ouput, the loop through the attributes
    #in options and if an attribute is in the function signature, add it to
    #the dict with it's associated value
    args_dict = dict() 
    for arg, value in vars(options).items():
        if arg in fun_arg_names:
            args_dict[arg] = value

    return args_dict

#-----------------------------------------------
def bool_option(v):
    '''Convert various string inputs that might be interpreted as yes/no, true/false to bool.
    If input is already a bool, just return itself
    
    Inputs:
        v : str
            variable to convert to bool
    Notes:
        

    Parameters
    ----------
    v : str
        variable to convert to bool

    Returns
    -------
    bool
        True, 'yes', 'true', 't', 'y', '1' convert to True

        False, 'no', 'false', 'f', 'n', '0' convert to False

        Any other input raises an argparse ArgumentTypeError

    Raises
    ------
    configargparse.ArgumentTypeError
        _description_
    '''
    if isinstance(v, bool):
       return v
    if str(v).lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str(v).lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')