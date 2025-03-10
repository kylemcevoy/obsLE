import numpy as np
import xarray as xr

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

    offset : xr.DataArray/scalar
        If offset is a DataArray it should have dimensions (month, lat, lon). If it
        is a scalar value, the constant value will be added to all the values of x.
        Otherwise, the offset added to x will be matched to the correct 
        (time.month, lat, lon) of x.

    lam : xr.DataArray, dim: (month, lat, lon)
        The Box-Cox power parameter. The parameter is matched to x using
        (time.month, lat, lon).

    Returns
    -------
    y : xr.DataArray, dims: (time, lat, lon)
        y is the Box-Cox transform applied to x with parameters given by offset
        and lam. The Box-Cox transform is defined by:
            if lambda = 0: y = log(x + offset)
            if lambda != 0: y = ((x + offset)^lambda - 1) / lambda
    """
    
    # copy the xarray object, so original xarray is not affected by assignments
    y = x.copy()
    
    if np.isscalar(offset):
        da_flag = False
        y = y + offset
    elif isinstance(offset, xr.DataArray):
        da_flag = True
    else:
        raise ValueError('offset should be a single scalar offset value'
                         'or a xr.DataArray containing offsets for each location '
                         '+ month')

    #lam was optimized for each month, so perform transform looping over the months 
    for i in range(1, 13):
        month_slice = y.loc[dict(time = (y['time.month'] == i))]
        
        if da_flag:
            month_slice = month_slice + offset.loc[dict(month = i)]
            
        lam_slice = lam.loc[dict(month = i)]

        # the use of fmax here is to suppress divide by 0 warnings at locations 
        # where lambda == 0, so the log transform will be used instead anyway
        lam_slice_trunc = np.fmax(lam_slice, 1e-6)
        month_transform = xr.where(lam_slice == 0,
                               np.log(month_slice), 
                               (month_slice**lam_slice - 1) / lam_slice_trunc)
    
        y.loc[dict(time=(y['time.month'] == i))] = month_transform
        
    return y

def inv_boxcox_transform(y, offset, lam):  
    """Performs inverse Box-Cox transforms on xr.DataArrays.

    Parameters
    ----------
    y : xr.DataArray, dims: (time, lat, lon)
        y should be a DataArray. The time dimension should be a datetime64. 

    offset : xr.DataArray/scalar
        If offset is a DataArray it should have dimensions (month, lat, lon). If it
        is a scalar value, the constant value will be subtracted from all the values 
        of y. Otherwise, the offset subtracted from y will be matched to the correct 
        (time.month, lat, lon) of y.

    lam : xr.DataArray, dim: (month, lat, lon)
        The Box-Cox power parameter. The parameter is matched to y using
        (time.month, lat, lon).

    Returns
    -------
    x : xr.DataArray, dims: (time, lat, lon)
        x is the inverse Box-Cox transform applied to y with parameters given by 
        offset and lam. The Inverse Box-Cox transform is defined by:
            if lambda = 0: x = np.exp(y) - offset
            if lambda != 0: x = ((y * lambda + 1)^(1 / lambda) - offset
    """
    
    # copy the xarray object, so original xarray is not affected by assignments
    x = y.copy()
    
    if np.isscalar(offset):
        da_flag = False
    elif isinstance(offset, xr.DataArray):
        da_flag = True
    else:
        raise ValueError('offset should be a single scalar offset value' +
                         'or a xr.DataArray containing offsets for each location '
                         '+ month')

    #lam was optimized for each month, so perform transform looping over the months 
    for i in range(1, 13):
        month_slice = x.loc[dict(time = (x['time.month'] == i))]

        lam_slice = lam.loc[dict(month = i)]

        # the use of fmin below is to prevent overflow warnings on locations where 
        # a non-log boxcox transform was used resulting in still quite large value. 
        # In these cases, the non-log boxcox-transform will have been applied 
        # instead anyway.
        month_transform = xr.where(lam_slice == 0,
                               np.exp(np.fmin(month_slice, 500)), 
                               (month_slice * lam_slice + 1)**(1 / lam_slice))

        if da_flag:
            month_transform = month_transform - offset.loc[dict(month = i)]
        else:
            month_transform = month_transform - offset

        x.loc[dict(time=(x['time.month'] == i))] = month_transform

    return x