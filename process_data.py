import numpy as np
import pandas as pd
import xarray as xr
import cftime
import warnings

####### Loading and processing climate modes
def process_climate_modes(mode_path,
                          start_year='1920',
                          end_year='2020',
                          save_path=None):
    """Process the climate mode files into a single data object
    containing monthly climate mode values.

    Parameters
    ----------
    mode_path : int
        path to the directory containing the climate mode data
    start_year : str
        four digit string for the first year of data to include
        in the climate mode output. Value should be >= 1920
    end_year : str
        four digit string for last year of data to include
        in the climate mode output. The year should have all 12 months
        of data. Value should be <= 2020.
    save_path : str
        Path to the directory for saving the output. The string should end in / after
        the directory name.
        If None, nothing will be saved. The pd.DataFrame output is converted to an
        xr.DataSet before saving, and is saved at save_path + "climate_mode.nc".

    Returns
    ---------
    mode_df : pd.DataFrame
        DataFrame with columns: [enso, pdo, pna, nao, ao]
        Rows contain monthly values for the modes from January of start_year
        to December of end_year. 
        
        
    """
    
    var_fnames = {'enso': 'nino34.long.anom.nc',
                  'pdo': 'pdo.timeseries.hadisst1-1.nc',
                  'pna_noaa': 'norm.mon.pna.wg.jan1950-current.ascii.table',
                  'pna_20cr': 'pna.20crv2c.long.data',
                  'nao': 'nao_3dp.dat',
                  'ao_noaa': 'monthly.ao.index.b50.current.ascii.table',
                  'ao_20cr': 'ao.20cr.long.data'}

    
    enso_ds = xr.open_dataset(mode_path + var_fnames['enso'])
    enso_pd = (enso_ds.
                 to_dataframe().
                 rename(columns={'value': 'enso'}).
                 astype(np.float64))
    enso_pd = enso_pd['enso']

    pdo = xr.open_dataset(mode_path + var_fnames['pdo'])
    pdo_pd = (pdo.
              astype(np.float64).
              to_dataframe())
    pdo_pd = pdo_pd['pdo']

    #noaa data time range: 1950-2023
    pna_noaa = pd.read_csv(mode_path + var_fnames['pna_noaa'],
                           sep='\\s+',
                           header=None)
    pna_noaa = pna_noaa.rename(columns={0: 'year'})
    pna_noaa = pna_noaa.set_index('year')
    
    #20CR has data from 1920 to 1950
    pna_20cr = pd.read_csv(mode_path + var_fnames['pna_20cr'],
                           engine='python',
                           skiprows=1,
                           skipfooter=5,
                           sep='\\s+',
                           header=None)

    
    pna_20cr = pna_20cr.rename(columns={0: 'year'})
    pna_20cr = pna_20cr.set_index('year').loc[slice('1920', '1949')]
    
    pna_df = pd.concat([pna_20cr, pna_noaa])
    pna_df = pna_df.reset_index()
    
    pna_melt = pd.melt(pna_df,
                       id_vars='year',
                       var_name='month',
                       value_name='pna')
    pna_melt['day'] = 1
    pna_melt['time'] = pd.to_datetime(pna_melt[['year', 'month', 'day']])
    pna_melt = pna_melt.set_index('time').sort_index()
    pna_pd = pna_melt['pna']

    nao = pd.read_csv(mode_path + var_fnames['nao'],
                      engine='python',
                      sep='\\s+', 
                      header=None, 
                      na_values='-99.990',
                      skiprows=4,
                      skipfooter=1)
    nao = nao.rename(columns={0: 'year'})
    nao = nao.drop(columns=13)
    
    nao_pd = pd.melt(nao,
                     id_vars='year',
                     var_name='month',
                     value_name='nao')
    
    nao_pd['day'] = 1
    nao_pd['time'] = pd.to_datetime(nao_pd[['year', 'month', 'day']])
    nao_pd = nao_pd.set_index('time').sort_index()['nao']

    ao_noaa = pd.read_csv(mode_path + var_fnames['ao_noaa'],
                     sep='\\s+',
                     names=['year'] + [*range(1, 13)],
                     skiprows=1)
    
    ao_20cr = pd.read_csv(mode_path + var_fnames['ao_20cr'],
                          engine='python',
                          sep='\\s+',
                          skiprows=1,
                          skipfooter=5,
                          header=None)
    ao_20cr = ao_20cr.rename(columns={0: 'year'})
    ao_20cr = (ao_20cr.
              set_index('year').
              loc[:1949].
              reset_index())
    
    ao = pd.concat([ao_20cr, ao_noaa])
    
    ao_melt = pd.melt(ao,
                      id_vars='year',
                      var_name='month',
                      value_name='ao')
    
    ao_melt['day'] = 1
    ao_melt['time'] = pd.to_datetime(ao_melt[['year', 'month', 'day']])
    ao_melt = ao_melt.set_index('time').sort_index()
    ao_pd = ao_melt['ao']
    
    mode_df = pd.DataFrame({'enso': enso_pd,
                            'pdo': pdo_pd,
                            'pna': pna_pd,
                            'nao': nao_pd,
                            'ao': ao_pd})
    
    mode_df = mode_df.loc[slice(start_year, end_year)]

    if save_path is not None:
        mode_ds = mode_df.to_xarray()
        description = ('dataset containing time series of climate modes for the regression model.'
        + 'Created by the process_climate_modes() function in data_processing.py')
        mode_ds.attrs['description'] = description
        mode_ds.to_netcdf(save_path + 'climate_modes.nc')

    return mode_df

def preprocess_so2(hist_files, future_files, save_path=None):
    
    import regionmask
    import cftime
    from itertools import product

    so2_hist = xr.open_mfdataset(hist_files)
    so2_hist = so2_hist['SO2_em_anthro']
    
    so2_fut = xr.open_mfdataset(future_files)
    so2_fut = so2_fut['SO2_em_anthro']

    so2 = xr.concat([so2_hist, so2_fut], dim='time')
    so2 = so2.sum('sector')
    so2 = so2.sel(time=slice('1920', '2020'))

    lon_coord = so2.lon
    lat_coord = so2.lat
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    US_mask = countries.mask(lon_coord, lat_coord) == 4
    so2_subset = so2.where(US_mask)

    so2_conus = so2_subset.sel(lat=slice(22, 50), lon=slice(-130, -60))

    so2_weighted = so2_conus.weighted(np.cos(np.deg2rad(so2_conus.lat)))

    so2_mean = so2_weighted.mean(dim=['lat', 'lon'])

    # convert from kg m-2 s-1 to g m-2 d-1
    so2_mean = so2_mean * (1000.0 * 60 * 60 * 24)
    
    dates = [cftime.DatetimeNoLeap(year, month, 16) 
             for year, month in product(range(2016, 2020), range(1, 13))]

    
    month_dates = []
    for i in range(1, 13):
        month_dates.append([date for date in dates if date.month == i])

    interp = []
    for month in range(1, 13):
        interp.append((so2_mean.
                       sel(time=so2_mean.time.dt.month == month).
                       interp(time=month_dates[month - 1])))

    interpolated_months = xr.concat(interp, dim='time').sortby('time')

    so2_mean = xr.concat([so2_mean, interpolated_months], dim='time').sortby('time')

    so2_mean = so2_mean.assign_attrs(units='g m-2 d-1', 
                                     description=('CONUS mean anthropogenic SO2 '
                                                  'emissions from input4MIP files. '
                                                  'See preprocess_so2() from '
                                                  'process_data.py'))
    so2_mean = so2_mean.compute()

    if save_path is not None:
        so2_mean.to_netcdf(save_path + 'conus_mean_so2.nc')

    return(so2_mean)

def process_forcings(forcings_path,
                     start_year='1920',
                     end_year='2020',
                     save_path=None):

    var_fnames = {'ico2_log': 'ico2_log.nc',
                 'so2': 'conus_mean_so2.nc'}
    
    ico2_log = xr.open_dataset(forcings_path + var_fnames['ico2_log'],
                              decode_times=False)

    # non-standard calendar + units lead to issues with converting to pd.datetime.
    # so we slice first then replace the index by a datetime64 index that is 
    # compatible with the other Series after the slicing is complete.
    ico2_log['time'] = cftime.num2date(ico2_log['time'].values,
                                       'months since 1600-01-01',
                                       calendar='360_day')

    ico2_log = ico2_log['log(co2)'].sel(time=slice(start_year, end_year))
    
    ico2_log_pd = pd.Series(ico2_log.values, 
                            index=pd.date_range(start=start_year + '-01-01',
                                                end=end_year + '-12-01',
                                                freq='MS'))

    # note that so2 is pre-processed into a data array from input4MIP files.
    so2 = xr.open_dataarray(forcings_path + var_fnames['so2'])

    so2_pd = pd.Series(so2.values,
                      index=pd.date_range(start=start_year + '-01-01',
                                                end=end_year + '-12-01',
                                                freq='MS'))
    
    forcing_df = pd.DataFrame({'ico2_log': ico2_log_pd, 'so2': so2_pd})

    forcing_df.index.rename('time', inplace=True)

    if save_path is not None:
        forcing_ds = forcing_df.to_xarray()
        description = ('dataset containing time series of climate forcings for the'
                      'regression model. Created by processing_forcings() function'
                      'in data_processing.py')
        forcing_ds.attrs['description'] = description
        forcing_ds.to_netcdf(save_path + 'forcings.nc')

    return forcing_df

###### For sequentially orthogonalizing the climate mode variables. ######
def gram_schmidt(mode, ortho_basis=None):
    """Perform a gram-schmidt orthogonalization on the input vector with respect to the given ortho basis

    Parameters
    ----------
    mode: 1D numpy array
        Vector of floats containing a time series of input data.
    ortho_basis: 2D numpy array
        The columns of this numpy array contain the orthogonal basis vectors against which orthogonalize the mode.
        ortho_basis.shape[0] should equal mode.shape[0].

    Returns
    -------
    ortho_mode : numpy array
        1D numpy array containing the orthogonalized mode
    """
    
    if ortho_basis is None:
        ortho_mode = mode
    else:
        inner_products = mode @ ortho_basis
        norm_sq = np.linalg.norm(ortho_basis, axis=0)**2
        proj_weights = inner_products / norm_sq

        ortho_mode = mode - np.sum(proj_weights * ortho_basis, axis=1)

    return ortho_mode     

def find_ortho_basis(mode_array):
    """Sequentially orthogonalize the columns of the input array with respect to all preceding columns.

    Parameters
    ----------
    mode_array: 2D numpy array of floats

    Returns
    -------
    ortho_basis: 2D numpy array
        Array of same dimension as input whose columns are now orthogonalized against each other. 
        The orthogonalization is performed iteratively over the columns in order by applying the gram-schmidt procedure. 
        See gram_schmidt() function for more details.
    """
    
    ortho_basis = mode_array[:, [0]]
    if mode_array.ndim > 1:
        for j in range(1, mode_array.shape[1]):
            ortho_mode = gram_schmidt(mode_array[:, j], ortho_basis)[:, np.newaxis]
            ortho_basis = np.hstack([ortho_basis, ortho_mode])
    return ortho_basis

def orthogonalize_modes(mode_df, mode_list=None, save_path=None):
    """Sequentially orthogonalize a DataFrame of climate modes
        using Gram-Schmidt orthogonalization,

    Parameters
    ----------
    mode_df : pd.DataFrame
        DataFrame containing monthly observations of climate modes
        with a pd.DateTime index. As outputted by process_climate_modes().
    mode_list : list of strings
        list of variable names from mode_df to subset and sequentially orthogonalize.

    Returns
    -------
    ortho_mode_df : pd.DataFrame
        ortho_mode_df will have the same number of rows as mode_df, it will
        have 1 + len(model_mode_list) number of columns with an intercept 
        (all ones) column as the first column. Each column in ortho_mode_df
        is orthogonal to all preceding columns (including the intercept).
    """
    
    if mode_list is None:
        X = mode_df.values
        col_names = ['intercept'] + mode_df.columns.to_list()
    else:
        X = mode_df[mode_list].values
        col_names = ['intercept'] + mode_list
        
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_orth = find_ortho_basis(X)
    
    X_orth_std = X_orth[:, 1:] / np.std(X_orth[:, 1:], axis=0, ddof=1)
    X_orth_std = np.hstack([X_orth[:, [0]], X_orth_std])
    
    ortho_modes_df = pd.DataFrame(X_orth_std,
                                  index=mode_df.index,
                                  columns=col_names)

    if save_path is not None:
        ortho_modes_ds = ortho_modes_df.to_xarray()
        description = ('dataset containing time series of climate modes that'
                       'have been orthogonalized. Created by the '
                       'orthogonalize_modes() function in data_processing.py')
        ortho_modes_ds.attrs['description'] = description
        ortho_modes_ds.to_netcdf(save_path + 'ortho_modes.nc')

    return ortho_modes_df

def build_ortho_mode_df(mode_df,
                        start_year,
                        end_year,
                        mode_list,
                        save_path=None,
                        mode_path=None):
    """Loads the climate mode files processes the data and outputs a pandas
    DataFrame that contains the orthogonal modes.

    Parameters
    ----------
    mode_df: pd.DataFrame
        DataFrame containing the climate modes as columns. The rows should contain
        monthly observations of the climate modes. The index should be a 
        pd.datetime64 index.
        
    start_year : str/int
        four digit string/int for the first year of data to include
        in the climate mode output. Value should be >= 1920
        
    end_year : str/int
        four digit string/int for last year of data to include
        in the climate mode output. Value should be <= 2020.
        
    mode_list : list of str
        the variable names to include in the output DataFrame. 
        
    save_path : str
        Path to the directory for saving the output. The string should end in / after
        the directory name.
        If None, nothing will be saved. The pd.DataFrame output is converted to an
        xr.DataSet before saving, and is saved at save_path + "ortho_mode.nc".
        
    mode_path : str
        Path to the directory containing the climate mode files. Only used if
        mode_df=None. See process_climate_modes() for the expected files. 

    Returns
    -------
    ortho_mode_df : pd.DataFrame
        the columns contain the orthogonalized climate modes, with the first
        column containing an intercept (all ones). The shape of the output is
        (12 * (#years)) by (1 + (# modes)).
    """
    
    if mode_df is None:
        mode_df = process_climate_modes(mode_path=mode_path,
                                        start_year=start_year,
                                        end_year=end_year,
                                        save_path=save_path)

    check_mode(mode_df)

    ortho_mode_df = orthogonalize_modes(mode_df=mode_df,
                                        mode_list=mode_list,
                                        save_path=save_path)
    
    check_ortho_mode(ortho_mode_df)

    return ortho_mode_df

def check_target(target_da):
    """This function does input checking on the target variable xr.DataArray.

    Parameters
    ----------
    target_da : xr.DataArray
        Object containing the target variable for the Obs-LE. target_da should have
        dimensions (time, lat, lon). time should have data type pd.datetime64. 
    
    Returns
    --------
    target_da : xr.DataArray
        This function will rename the dimensions latitude/longitude to lat/lon if
        they are found in the data. It will also use transpose to reorder the
        dimensions to (time, lat, lon).
    """
    
    if not isinstance(target_da, xr.DataArray):
        raise ValueError('target_da is not an xr.DataArray.')
    coords = target_da.coords

    if 'time' not in coords:
        raise ValueError('target_da does not have a time coord. '
                        'time should be datetime64 coord with the same '
                         'months as ortho_mode_df.index.')

    if not pd.api.types.is_datetime64_dtype(target_da.time):
        raise ValueError('target_da.time should be a datetime64 type.')

    mount_counts = np.unique(target_da['time.month'], return_counts=True)[1]

    if any(mount_counts != mount_counts[0]):
        raise ValueError('Only whole years should be included in target_da, '
                        'so each month should have the same number of occurences.')

    if 'lat' not in coords:
        if 'latitude' in coords:
            target_da = target_da.rename({'latitude': 'lat'})
        else:
            raise ValueError('target_da must have a latitude coordinate. ' 
                             'Preferably named lat.')

    if 'lon' not in coords:
        if 'longitude' in coords:
            target_da = target_da.rename({'longitude': 'lon'})
        else:
            raise ValueError('target_da must have a longitude coordinate. ' 
                             'Preferably named lon.')

    target_da = target_da.transpose('time', 'lat', 'lon')

    return target_da

def check_mode(mode_df):
    """Input checking for the DataFrame of climate modes.

    Parameters
    ----------
    mode_df : pd.DataFrame, dims: (time x climate modes)
        mode_df should have one row per month. With a pd.datetime64 index. Only
        whole years should be included. All months are checked to see if they
        have the same number of occurences. The index should be sorted.

    Returns
    -------
    mode_df : pd.DataFrame, dims: (time x climate_modes)
        The returned mode_df object is identical to the input. Errors are raised
        if any of the checks fail.
    """
    if not pd.api.types.is_datetime64_dtype(mode_df.index):
        raise ValueError('mode_df.index should be a datetime64 type.')
    
    mount_counts = np.unique(mode_df.index.month, return_counts=True)[1]

    if any(mount_counts != mount_counts[0]):
        raise ValueError('Only whole years should be included in mode_df, '
                        'so each month should have the same number of occurences.')

    return mode_df

def check_ortho_mode(ortho_mode_df):
    """Input checking for the orthogonalized climate mode DataFrame,

    Parameters
    ----------
    ortho_mode_df : pd.DataFrame, dims: (time x #{climate modes} + 1)
        This DataFrame should have a pd.datetime64 index. The first column is
        expected to be an intercept column, while the other columns contain the
        climate modes that could be used in the Obs-LE after they have been
        sequentially orthogonalized using Gram-Schmidt and standardized. See 
        orthogonalize_climate_modes() and gram_schmidt() for details.

    Returns
    -------
    ortho_mode_df : pd.DataFrame, dims: (time x #{climate modes} + 1)
        If all checks are passed, this will be identical to the input. If the
        columns of ortho_mode_df are not standardized, the output columns will be
        standardized and warning will be raised.
    """
    p = ortho_mode_df.shape[1]
    
    if not pd.api.types.is_datetime64_dtype(ortho_mode_df.index):
        raise ValueError('ortho_mode_df.index should be a datetime64 type.')
    
    mount_counts = np.unique(ortho_mode_df.index.month, return_counts=True)[1]

    if any(mount_counts != mount_counts[0]):
        raise ValueError('Only whole years should be included in mode_df, '
                        'so each month should have the same number of occurences.')

    if not np.allclose(ortho_mode_df.std(ddof=1)[1:], 1.0):
        warnings.warn('the columns of ortho_mode_df are expected to have '
                      'standard deviation 1. Standardizing columns')
        ortho_mode_df = ortho_mode_df / ortho_mode_df.std(ddof=1)

    column_inner_prods = np.transpose(ortho_mode_df.values) @ ortho_mode_df.values

    if not np.allclose(column_inner_prods[np.tril_indices(p, k=-1)], 0):
        raise ValueError('the columns of ortho_mode_df are not orthogonal')
    
    return ortho_mode_df
    