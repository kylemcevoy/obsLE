import numpy as np

####### Boxcox Transforms and Inverse Transforms.
def boxcox_transform_np(x, offset, lam):
    """Perform boxcox transform on np.ndarrays,
    this function is primarily used in the optimization
    of (lambda, offset) where a single value of lambda
    and offset are used to transform an entire numpy array.
    When (lambda, offset) vary over time, use boxcox_transform().


    Parameters
    ----------
    x: numpy ndarray
        x contains data to transform
    offset: numeric scalar
        offset is for constant shifts to x. If x contains 0s, offset
        should be positive.
    lam: non-negative numeric scalar
        boxcox lambda parameter

    Returns
    -------
    y: numpy ndarray
        y is the Box-Cox transform of x with power parameter lam
        and shift parameter offset. This is given by the formula
        y = np.log(x + offset) if lam == 0 and
        y = ((x + offset)^lam - 1) / lam if lam > 0
    """
    if lam == 0:
        y = np.log(x + offset)
    else:
        y = ((x + offset)**lam - 1) / lam
    return y

def boxcox_transform(x, offset, lam):
    """Performs boxcox transforms on xr.DataArrays.

    Parameters
    ----------
    x : xr.DataArray, dims: (time, lat, lon)
        x should be a DataArray. The time dimension should be a datetime64. Values 
        of x should be positive after the offsets are added. See 
        data_processing.check_target() for more information on the expected 
        input x.

    offset : float64
        The constant value will be added to all the values of x. Should be
        a positive value. After adding the offset all values should be
        positive.

    lam : xr.DataArray, dim: (month, lat, lon)
        The Box-Cox power parameter. The parameter is matched to x using
        (time.month, lat, lon). Only powers greater than 0 are allowed,
        i.e. no log transform.

    Returns
    -------
    y : xr.DataArray, dims: (time, lat, lon)
        y is the Box-Cox transform applied to x with parameters given by offset
        and lam. For non-zero lambda, the Box-Cox transform is defined by:
        y = ((x + offset)^lambda - 1) / lambda
    """
    
    # copy the xarray object, so original xarray is not affected by assignments
    y = x + offset

    lam_ts = lam.loc[{'month': y['time.month']}]

    y = (y**lam_ts - 1) / lam_ts

    return y

def inv_boxcox_transform(y, offset, lam):  
    """Performs inverse Box-Cox transforms on xr.DataArrays.

    Parameters
    ----------
    y : xr.DataArray, dims: (time, lat, lon)
        y should be a DataArray. The time dimension should be a datetime64. 

    offset : float64
        The constant value will be added to all the values of x. Must be
        a positive value. After adding the offset all values should be
        positive.

    lam : xr.DataArray, dim: (month, lat, lon)
        The Box-Cox power parameter. The parameter is matched to y using
        (time.month, lat, lon).

    Returns
    -------
    x : xr.DataArray, dims: (time, lat, lon)
        x is the inverse Box-Cox transform applied to y with parameters given by 
        offset and lam. The Inverse Box-Cox transform for non-zero lambda
        is defined by: x = ((lambda * y + 1)^(1 / lambda) - offset
    """
    
    if not np.isscalar(offset):
        raise ValueError('offset should be a single scalar offset value')

    # this construction matches lambda values with y on the 
    # time axis by month, and creates a matching time coordinate
    lam_ts = lam.loc[{'month': y['time.month']}]

    # ensures that all y values fall within the domain of the
    # inverse box-cox transform (necessary because of resampling)
    y = np.fmax(y, -1 / lam_ts)
    x = (y * lam_ts + 1)**(1 / lam_ts) - offset

    # y values mapped to -1 / lam_ts will be equal to 0
    # before subtracting offset, want to stay non-negative
    # after offset so use np.maximum to do this and propage
    #  np.nans
    x = np.maximum(x, 0)
    return x
    