# Dissertation_v2
 final version

1. Use **ERA5_API.ipynb** to download the ERA5 SST data. 
  This data file needs to be converted from netcdf3 to netcdf4, for which **Convert_to_netcdf4.ipynb** is provided. In the same notebook you'll fine the code to detrend this data.
  
2. The data for the Precipitation rate has to be downloaded directly from NOAA website, here I provide a link to my GitHub to download this data: https://github.com/cesardt97/Dissertation_v2/tree/main/data

3. After that, you just have to run the following files, in order:
    a. **Run_PCA_Varimax.ipynb**
    b. **RUN_PCMCI.ipynb**
    c. **LASSO_selection.ipynb**
    c. **Encoder - 15comps.ipynb**
    
Automatically the output files will be generated. 
If an error occurs, make sure you have the following folders created:

1) ./data/
2) ./plots/
3) ./runs/
4) ./runs/encoder_results/
5) ./runs/pcmci_results/
6) ./runs/pcmci_results/test
7) ./runs/train/


