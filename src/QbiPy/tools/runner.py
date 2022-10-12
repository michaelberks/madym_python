'''
'''
import os
import sys
import socket
import getpass
import inspect

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import configargparse

#----------------------------------------------
class QbiRunner():
    def __init__(self, description = None):
        self.description_ = description
        self.parser = configargparse.ArgParser(description)
        self.set_generic_options()
    
    #---------------------------------------------
    def set_generic_options(self):
        '''
        Set-up the generic options for the runner as args in the parser
        '''
        #Config file
        self.parser.add('--config', required=False, is_config_file=True,
            type=str, nargs='?', default=None,
            help='Path to config file')

        #Ouput folder
        self.parser.add('--output_dir', type=str, nargs='?', default=None,
            help='Relative path to the output folder')

        #Overwrite flag
        self.parser.add('--overwrite', type=bool, nargs='?', default=True, const = True,
            help='Flag to allow overwriting previous output')
        
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
        self.parser.add('--no_log', type=bool, nargs='?', default=False, const = False,
            help='Flag to turn off program logging. Info is printed to stdout')

        #Audit log flag
        self.parser.add('--no_audit', type=bool, nargs='?', default=False, const = False,
            help='Flag to turn off audit logging.')
        
    #----------------------------------------------
    def parse_args(self, args = None):
        if args is None:
            args = sys.argv[1:]
        return self.parser.parse_args(args)

    #-----------------------------------------------
    def run(self, fun, args = None):
        '''
        Run the given function fun inside the runner, setting up an output
        directory and program/audit logs
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

        #Get function args from options namespace
        args = get_function_args(fun, options)

        #If user set no log, then just run the function
        if options.no_log:
            fun(**args)
        else:
            run_with_logging(options, args, fun)

#-----------------------------------------------
def run_with_logging(options, args, fun):
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
            fun(**args)

        #Finalise the program log
        finalise_log(program_log)

    #Finalise the audit log
    if do_audit:
        finalise_audit_log(audit_log_path)

#-----------------------------------------------
def initialise_audit_log(audit_log_path, program_log_path):
    os.makedirs(os.path.dirname(audit_log_path), exist_ok = True)
    with open(audit_log_path, 'wt') as log:
        initialise_log(log)
        print(f'Program log saved to {program_log_path}', file=log)

#-----------------------------------------------
def finalise_audit_log(audit_log_path):
    with open(audit_log_path, 'at') as log:
        finalise_log(log)

#-----------------------------------------------
def finalise_log(log):  
    print(f'{sys.argv[0]} completed successfully.', file=log)
    date_str = datetime.today().strftime('%Y%m%d %H:%M:%S')
    print(f'Log closed at {date_str}', file=log)

#-----------------------------------------------
def initialise_log(log):
    date_str = datetime.today().strftime('%Y%m%d %H:%M:%S')
    print(f'Log opened at {date_str}', file=log)
    print(f'User: {getpass.getuser()};   Host: {socket.gethostname()}', file=log)
    print(f'Ran in: {os.getcwd()}', file=log)
    command_args = ' '.join(sys.argv)
    print(f'Command args: {command_args}', file=log)

#-----------------------------------------------
def write_config_log(options, config_log_path):
    with open(config_log_path, 'wt') as log:
        for option, value in options.__dict__.items():
            print(f'{option} = {value}', file = log)

#-----------------------------------------------
def get_function_args(fun, options):
    '''
    Iterate through attributes in a namepsace object and return attributes
    included in the signature of the given function as a dictionary

    Inputs:
        fun: function object

        options: Simple namepsace object returned from argparse

    Outputs:
        args_dict: dictionary of attribute/values from options where
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
        

