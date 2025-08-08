####### Fit monthly regressions.
import numpy as np
import xarray as xr

from . import transform

def fit_linear_models(y, X):
    """Fit OLS models of each individual column (locations) of y regressed against
    the design matrix X. The models are fit separately for each month by 
    subsetting the rows of y and X. A total of 12 * #{locations} independent
    regression models are fit.

    Parameters
    ----------
    y: 2D numpy array
        Matrix containing n observation rows against L columns containing time
        series of output data. L is an index for the locations at which the
        time series are recorded.
    X: 2D numpy array
        Design matrix containing n observation rows against p variable columns
        (including intercept). The columns of X should contain time series
        of monthly data containing only complete years, so that n is
        divisible by 12. X should contain an intercept column.

    Returns
    -------
    tuple(beta, RSS, residuals, fitted_values)

    beta: numpy array of shape (12, p, L)
        Contains the fitted regression coefficients indexed by 
        (month, variable, location). For each regression model fit using 
        y[I{month_j}] ~ X[I{month_j}], where I{month_j} is an indicator for the
        rows coming from month j. If L = 1, the last dimension is dropped.

    RSS: numpy array of shape (12, L)
        Contains the residual sums of squares of each fitted model. If L = 1, the
        last dimension is dropped.
        
    residuals: numpy array of shape (12, n // 12, L)
        Contains the residuals of each fitted model split by month. If L = 1, the
        last dimension is dropped.

    fitted_values: numpy array of shape (12, n // 12, l)
        Contains the fitted values of each regression model split by month. If
        L = 1 the last dimension is dropped.
    """
    
    n = X.shape[0]
    # referred to as p + 1 in documentation
    p = X.shape[1]
    
    if y.ndim > 1:
        L = y.shape[1]
        
        beta = np.zeros((12, p, L))
        residuals = np.zeros((12, n // 12, L))
        fitted_values = np.zeros((12, n // 12, L))
        RSS = np.zeros((12, L))
    else:
        beta = np.zeros((12, p))
        residuals = np.zeros((12, n // 12))
        fitted_values = np.zeros((12, n // 12))
        RSS = np.zeros((12))
        
    I_month = [(np.arange(n) % 12) == i for i in np.arange(12)]

    for i in range(12):
        beta[i, ...], RSS[i], *_ = np.linalg.lstsq(X[I_month[i]],
                                                    y[I_month[i]],
                                                    rcond=None)
        fitted_values[i] = (X[I_month[i]] @ beta[i, ...])
        residuals[i] = y[I_month[i]] - (X[I_month[i]] @ beta[i, ...])

    return beta, RSS, residuals, fitted_values

def build_model_ds(beta,
                   var_names,
                   RSS,
                   residuals,
                   fitted_values,
                   nan_mask,
                   coords):
    """Builds xr.Dataset objects from the regression outputs of fit_linear_models.

    Parameters
    ----------
    beta: np.ndarray, dims: (12, p + 1, l)
        Contains the beta coefficients from the regressions fit by fit_linear_models.

    var_names: list of str
        contains the var_names associated with the p + 1 regression coefficients in
        beta.

    RSS: np.ndarray, dims: (12, l)
        Residual sums of squares from the regression models.

    residuals: np.ndarray, dims: (12, n // 12, l)
        where n is the total number of time points. The residuals of the 
        regression models.

    fitted_values: np.ndarray, dims: (12, n // 12, l)
        The fitted values of the regression models.

    nan_mask: np.ndarray, dims: (lat, lon)
        np.ndarray containing True/False. The Trues mark (lat, lon) locations in t
        he original y (that was converted to y for fit_linear_models) where
        there was actual data, i.e. locations not filled with np.nan.

    coords: dict or xarray coords
        containing the keys ('time', 'lat', 'lon') and the corresponding coordinate
        values.

    Returns
    -------
    beta_ds, lm_out_ds: tuple of xr.Datasets
        beta_ds contains the beta coefficients of the regression model with 
        variables for each of the predictor variables (including intercept)
        and dimensions (month, lat, lon).

        lm_out_ds contains the residuals, fitted_values and RSS of the regression
        model. Residuals and fitted_values have dim (time, lat, lon) and RSS has
        dim. (month, lat, lon).
    """

    time_coord = coords['time']
    lat_coord = coords['lat']
    lon_coord = coords['lon']
    
    # beta is shape (m x p x l), RSS is m x l, residuals/fitted_values are n x m x l
    p = beta.shape[1]
    L = residuals.shape[2]
    lat_len = lat_coord.shape[0]
    lon_len = lon_coord.shape[0]
    time_len = time_coord.shape[0]

    residuals = residuals.reshape((time_len, L))
    fitted_values = fitted_values.reshape((time_len, L))
    
    beta_data = np.empty((12, p, lat_len, lon_len))
    beta_data.fill(np.nan)
    beta_data[:, :, nan_mask] = beta

    RSS_data = np.empty((12, lat_len, lon_len))
    RSS_data.fill(np.nan)
    RSS_data[:, nan_mask] = RSS

    residuals_data = np.empty((time_len, lat_len, lon_len))
    residuals_data.fill(np.nan)
    residuals_data[:, nan_mask] = residuals

    fitted_data = np.empty((time_len, lat_len, lon_len))
    fitted_data.fill(np.nan)
    fitted_data[:, nan_mask] = fitted_values
    
    beta_ds = xr.Dataset(coords={'month': np.arange(1, 13),
                                 'lat': lat_coord,
                                 'lon': lon_coord})

    for i, var in enumerate(var_names):
        beta_ds[var] = (['month', 'lat', 'lon'],
                        beta_data[:, i])

    lm_out_ds = xr.Dataset(data_vars = {'residuals': (['time', 'lat', 'lon'], 
                                                      residuals_data),
                                        'fitted_values': (['time', 'lat', 'lon'], 
                                                          fitted_data),
                                        'RSS': (['month', 'lat', 'lon'], RSS_data)},
                           coords={'month': np.arange(1, 13),
                                   'time': time_coord,
                                   'lat': lat_coord,
                                   'lon': lon_coord})
    
    return beta_ds, lm_out_ds


def fit_optimized_model(y,
                        X,
                        lam,
                        offset,
                        model_mode_list=None,
                        save_path=None):
    """Wrapper function that transforms y by the optimized
    transform. Then fits the linear regression model onto the orthogonalized 
    climate modes.

    Parameters
    ----------
    y: xr.DataArray, dims: (time, lat, lon).
        Contains the target variable for the regression/Obs-LE.

    X: pd.DataFrame, dims: (time, #{climate modes} + 1)
        Contains the orthogonalized climate modes and forcings for the regression.

    lam: xr.DataArray, dims: (month, lat, lon)
        contains the optimized lambda parameter for the Box-Cox transform.
    
    offset: float64
        contains the offset parameter for the Box-Cox transform.

    model_mode_list: list of str or None
        contains the names of the climate modes to fit the model on. Do not include
        intercept, as it will be added to the list by default. If None, all modes 
        from X are used.

    save_path: str or None
        path to the directory to save the output. The directory should end in /.
        The outputs will be saved to save_path + beta.nc and 
        save_path + regression_output.nc, respectively. If None, nothing is saved.

    Returns
    -------
    beta_ds, lm_out_ds: tuple of xr.Datasets.
        beta_ds contains the regression coefficients and lm_out_ds contains the 
        residuals, fitted values, and RSS. See fit_linear_models() and 
        build_model_ds() for additional details.
    """
    
    transformed_target = transform.boxcox_transform(y,
                                                    lam=lam,
                                                    offset=offset)

    nan_mask = ~transformed_target.isnull().all(dim='time')

    # this expects time to be the first dimension
    transformed_values = transformed_target.values[:, nan_mask]

    beta, RSS, residuals, fitted_values = fit_linear_models(y=transformed_values,
                                                            X=X)

    beta_ds, lm_out_ds = build_model_ds(beta=beta,
                                        var_names=X.columns.to_list(),
                                        RSS=RSS,
                                        residuals=residuals,
                                        fitted_values=fitted_values,
                                        nan_mask=nan_mask,
                                        coords=y.coords)

    if save_path is not None:
        beta_ds.to_netcdf(save_path + 'beta.nc')
        lm_out_ds.to_netcdf(save_path + 'regression_output.nc')

    return beta_ds, lm_out_ds