import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import datetime
import os

def convert_to_netcdf4(path, nc3_file: str, nc4_file: str, variable: str) -> None:
    """ Convert ERA5 NETCDF3_64BIT_OFFSET to NETCDF4 format for a single variable """

    data = Dataset(path+nc3_file, 'r', format='NETCDF3_64BIT_OFFSET')
    v = data.variables[variable]
    if 'expver' in data.variables.keys():
        data2 = v[:,0,:,:] # Extracting the variable data for the main run (0), eliminating the expver dimension
    else:
        data2 = v[:,:,:]
    print('Data shape: ', data2.shape)

    with Dataset(path+nc4_file, 'w', format='NETCDF4') as dst:
        dst.setncatts({k: data.getncattr(k) for k in data.ncattrs()})

        for name, dimension in data.dimensions.items():
            if not name == 'expver':
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
                
        for name, var in data.variables.items():
            if name in ['time', 'latitude', 'longitude', 'lon', 'lat']:
                x = dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                dst[name][:] = data[name][:]
                
            if name == variable:
                x = dst.createVariable(name, var.datatype, ('time', 'latitude', 'longitude'))
                dst[name].setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                dst[name][:] = data2[:]
                
        print(f"Converted {nc3_file} to {nc4_file}")

def primary_data_and_missing_values(dataset: Dataset, var: str) -> np.ndarray:
    """ Check for missing values in the primary data and fill them with backup data. """
    if 'expver' in dataset.variables.keys():
        primary_data = dataset.variables[var][:, 0, :, :]
        if len(primary_data[primary_data == -32767]) == 0:
            print(f'No missing values in {var} variable found.')
        else:
            print('Missing values found, filling with backup data...')
            backup_data = dataset.variables[var][:, 1, :, :]
            primary_data[primary_data == -32767] = backup_data[primary_data == -32767]

            print(f'Remaining missing values: ', {len(primary_data[primary_data == -32767])})

    return primary_data

def calculate_wind_speed(dataset: Dataset) -> np.ndarray:
    """ Calculate the wind speed using the u and v components directly from the dataset. """

    u = primary_data_and_missing_values(dataset, 'u')
    v = primary_data_and_missing_values(dataset, 'v')

    # print(u.variables['longitud'], v.variables['longitud'])

    return (u**2 + v**2)**0.5

def save_to_netcdf4(original_file: str, new_file: str, variables: dict) -> None:
    """ Save variables defined in a dictionary to a new netCDF4 file."""

    with Dataset(original_file, 'r') as src, Dataset(new_file, 'w', format='NETCDF4') as dst:
        # Copy attributes from the original file
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        
        # Copy dimensions, except for expver variable
        for name, dimension in src.dimensions.items():
            if not name == 'expver':
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
        
        # Copy all file data
        for name, variable in src.variables.items():
            if name in ['time', 'latitude', 'longitude']:
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)
                    dst[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                    dst[name][:] = src[name][:]

        for name, variable in variables.items():
            x = dst.createVariable(name, variable['datatype'], variable['dimensions'])
            dst[name].setncatts(variable['attributes'])
            dst[name][:] = variable['data']

        print(f"Saved to {new_file}")

def find_files(path,prefix):
    """ Find files in a directory that match a prefix. """
    files = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.startswith(prefix):
                    files.append(entry.name)
    except Exception as e:
        print(f"An error occurred: {e}")

    return files

def create_lagged_features(data, lags):
    temp = pd.DataFrame(index=data.index)
    for col in data.columns:
        for lag in range(1, lags+1):
            temp[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    return temp