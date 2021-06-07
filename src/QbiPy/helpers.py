'''
Some auxilliary functions that provide utilities to the rest of the package
'''

import numpy as np

def check_param_size(**kwargs):
    '''
    For tissue model inputs, we accepts parameter inputs as either
    multi-dimensional arrays or scalar values. If they are arrays they must all be the same size, and are flattened to 1D vectors. If they are scalar
    they are converted to 1-element 1D vectors.

    If any inputs do not match a ValueError is raised

    inputs: keyword list of parameters, either scalars, 1 element arrays or 
        n-d arrays of the same size (but necessarily shape)

    outputs:
        n_vox: size of parameter arrays, 1 iff all parameters are scalars

        params: input parameters reshaped as 1D array
    '''
    n_vox = 0
    for param_name, param in kwargs.items():
        kwargs[param_name] = np.array(param).flatten()
        if kwargs[param_name].size > n_vox:
            n_vox = kwargs[param_name].size

    for param_name, param in kwargs.items():
        print(f'{param_name} size = {param.size}')

        if param.size > 1 and param.size != n_vox:
            raise ValueError(
                f'Error, size of {param_name} ({param.size}) does not match the size of the other parameters ({n_vox})')
    
    return (n_vox, *kwargs.values())
    
def check_param_shape(**kwargs):
    '''
    For parameter conversion, we accepts parameter inputs as either
    multi-dimensional arrays (or array like objects) or scalar values. If they are arrays they must all be the same shape. If they are scalar
    they are converted to np.arrays

    If any inputs do not match a ValueError is raised

    inputs: keyword list of parameters, either scalars or 
        n-d arrays of the same shape

    outputs:
        shape: shape of parameters

        params: input parameters converted to np.arrays
    '''
    shape = None
    
    for param_name, param in kwargs.items():
        kwargs[param_name] = np.array(param)

        if kwargs[param_name].size > 1:          
            if shape is None:
                shape = kwargs[param_name].shape
            elif shape != kwargs[param_name].shape:
                raise ValueError(
                    f'Error, shape of {param_name} ({kwargs[param_name].shape}) does not match the shape of the other parameters ({shape})')

    if shape is None: #all inputs were scalar...
        shape = (1)

    return (shape, *kwargs.values())

    