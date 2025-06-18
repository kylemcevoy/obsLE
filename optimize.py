import numpy as np
import xarray as xr

import obsLE.transform
import obsLE.process_data as data_proc

###### Optimization Helpers
def np_to_da(data_np, var_name, coord_dict, nan_mask):
    """Helper function to convert a np.ndarray to a xr.DataArray.

    Parameters
    ----------
    data_np: np.ndarray
        The input array should have final dimension indexing locations with 
        length lat * lon. The array should have the locations that are filled
        with np.nan masked out, resulting in the (lat, lon) coordinates being
        compressed into a single dimension.
    var_name: str
        gives the name to the resulting DataArray.
    coord_dict: dict
        gives a mapping from coord names to values as used in the coord argument
        of the xr.DataArray() constructor.
    nan_mask: np.ndarray, dims: (lat, lon)
        The nan_mask has True for (lat, lon) locations where there are NOT np.nan
        values in the original DataArray that gave rise to data_np. i.e., 
        data_np = da.values[:, nan_mask], where the last two dimensions are 
        (lat, lon).

    Returns
    -------
    da: xr.DataArray
        with dims/coordinates given by coord_dict and values drawn from data_np.
        The locations that were all np.nans in the DataArray, da, that data_np was 
        derived from. coord_dict should match da.coords.
    """
    dim_len = []
    for val in coord_dict.values():
        dim_len.append(val.shape[0])
    da_data = np.empty(dim_len)
    da_data.fill(np.nan)
    da_data[..., nan_mask] = data_np
    da = xr.DataArray(data=da_data, coords=coord_dict, name=var_name)
    return da


### Optimization of profile likelihoods for Box-cox transformations
def optimize_transform(y,
                       X,
                       lambda_values,
                       offset_values,
                       save_path=None):
    """Perform optimization of Box-Cox parameters at each location using maximum
    likelihood over a grid of parameters. The profile likelihood approach is taken.

    Parameters
    ----------
    target_da: xr.DataArray, dims: (time, lat, lon)
        DataArray containing the target variable for regressions/Obs-LE. 
        See data_processing.check_target() for more on expectations about
        the target_da.

    X: pd.DataFrame, dims: (time, #{climate_modes} + 1)
        X is the design matrix for the regression. It contains monthly values for 
        the forcings and the orthogonalized and standardized climate modes. 
        The first column contains an intercept. See 
        data_processing.build_ortho_mode() for additional details.

     lambda_values: np.ndarray, 1D
         Contains the values for lambda to optimize over. lambda is the power
         parameter of the Box-Cox Transform and should be non-negative. See
         transform.boxcox_transform() for more details.

    offset_values: np.ndarray, 1D
        Contains the values for offset to optimize over. Offset gives a linear
        shift of the values in taret_da before the Box-Cox Transform is applied.
        See transform.boxcox_transform() for additional details. In most cases,
        we set offset to a small positive value so that the transform is valid
        for data with some 0s. e.g. offset_values=[1e-6].

    save_path: str
        path to the directory where the output of the optimization should be saved.
        The path should end in /, and the output will be saved at 
        save_path + 'optim_transform_params.nc' and
        save_path + 'optim_transform_loglik.nc'. If None, nothing will be saved.

    Returns
    -------
    param_ds, optim_ds: tuple of xr.Datasets
        param_ds contains variables lam and offset with dimensions (month, lat, lon).
        These DataArrays contain the combination of values in lam_values and
        offset_values that resulted in the maximum log-likelihood over the grid.
    """

    target_da = data_proc.check_target(target_da)

    # Find locations that are always NaN and create a mask for the non-NaN
    # locations.
    xr_nan_mask = ~target_da.isnull().all(dim='time')
    nan_mask = xr_nan_mask.values

    # Expected dimensions are (time x lat x lon)
    target_values = target_da.values[:, xr_nan_mask]
    
    l = target_values.shape[1]
    n = X.shape[0] // 12
    p = X.shape[1]

    target_reshape = target_values.reshape(n, 12, l)
    X_reshape = X.values.reshape(n, 12, p)
    
    lambda_len = len(lambda_values)
    offset_len = len(offset_values)
    # Find the Jacobian determinant for the likelihood function of precip
    boxcox_lik_J = np.zeros((lambda_len, offset_len, 12, l))
    
    for i, lam in enumerate(lambda_values):
        for j, offset in enumerate(offset_values):
            boxcox_lik_J[i, j] = (lam - 1) * np.sum(np.log(target_reshape + offset), axis=0)
    
    # Find the RSS of each regression model after transformation
    RSS_array = np.zeros((lambda_len, offset_len, 12, l))
    
    for i, lam in enumerate(lambda_values):
        for j, offset in enumerate(offset_values):
            transformed_data = obsLE.transform.boxcox_transform_np(target_reshape, offset=offset, lam=lam)
            for m in range(12):
                # y is a n x l matrix so np.linalg.lstsq fits l independent regressions on the covariates
                y = transformed_data[:, m]
                lm_out = np.linalg.lstsq(X_reshape[:, m], y)
                # lstsq returns the RSS as the second element as a (l, ) shaped numpy array
                RSS_array[i, j, m] = lm_out[1]
    
    # MLE estimate of sigma^2 is RSS / n
    sigma2_array = RSS_array / n
    
    # Log likelihood for regression model on transformed y with normal errors + Jacobian of transform
    log_lik_array = (
        -(n / 2) * np.log(2 * np.pi)
        - (n / 2) * np.log(sigma2_array)
        - ((1 / (2 * sigma2_array)) * RSS_array) 
        + boxcox_lik_J)
    
    best_items = np.zeros((2, 12, l), dtype='int')
    ties_counter = 0
    
    if np.isin(1.0, lambda_values):
        no_transform_indx = np.where(lambda_values == 1.0)[0][0]
    else:
        no_transform_indx = np.nan
    
    for m in range(12):
        for l in range(l):
            log_lik_mat = log_lik_array[:, :, m, l]
            max_log_lik = np.max(log_lik_mat)
            best_items_tmp = np.argwhere(log_lik_mat == max_log_lik)
            if best_items_tmp.shape[0] > 1:
                # if there is no transform then all offsets will have the same likelihood
                # since the offset can just be folded into the intercept of the linear model.
                if all(best_items_tmp[:, 0] == no_transform_indx):
                    best_items[:, m, l] = best_items_tmp[0]
                else:
                    ties_counter += 1
                    best_items[:, m, l] = best_items_tmp[0]
            else:
                best_items[:, m, l] = best_items_tmp
    
    if ties_counter > 0:
            print(f'optimization ties: {ties_counter}')
    
    best_lambdas = lambda_values[best_items[0]]
    best_offsets = offset_values[best_items[1]]
    
    coord_dict = {'month': np.arange(1, 13),
                  'lat': target_da['lat'],
                  'lon': target_da['lon']}
    
    coord_dict_llik = {'lam': lambda_values,
                       'offset': offset_values,
                       'month': np.arange(1, 13),
                       'lat': target_da['lat'],
                       'lon': target_da['lon']}
    
    lambda_da = np_to_da(best_lambdas,
                         var_name='lam',
                         coord_dict=coord_dict,
                         nan_mask=nan_mask)
    offset_da = np_to_da(best_offsets,
                         var_name='offset',
                         coord_dict=coord_dict,
                         nan_mask=nan_mask)
    
    param_ds = xr.Dataset({'lam': lambda_da, 'offset': offset_da})
    
    optim_ds = np_to_da(log_lik_array,
                        var_name="loglik",
                        coord_dict=coord_dict_llik,
                        nan_mask=nan_mask)
    optim_ds = optim_ds.to_dataset()
    
    # p already includes intercept, but penalty term needs to include extra
    # param for unknown regression variance
    AIC_penalty = 2 * (p + 1)
    optim_ds['AIC'] = AIC_penalty - 2 * optim_ds['loglik']

    if save_path is not None:
        param_ds.to_netcdf(save_path + 'optim_transform_params.nc')
        optim_ds.to_netcdf(save_path + 'optim_transform_loglik.nc')
        
    return param_ds, optim_ds