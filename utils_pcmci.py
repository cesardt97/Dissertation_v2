from tigramite.pcmci import PCMCI
import itertools
import pickle
import pandas as pd
import numpy as np

def split(container, count):
    """
    Simple function splitting a range of selected variables (or range(N)) 
    into equal length chunks. Order is not preserved.
    """
    return [container[_i::count] for _i in range(count)]


def run_pc_stable_parallel(j, dataframe, cond_ind_test, verbosity, params):

    """Wrapper around PCMCI.run_pc_stable estimating the parents for a single 
    variable j.

    Parameters
    ----------
    j : int
        Variable index.

    Returns
    -------
    j, pcmci_of_j, parents_of_j : tuple
        Variable index, PCMCI object, and parents of j
    """

    # CondIndTest is initialized globally below
    # Further parameters of PCMCI as described in the documentation can be
    # supplied here:
    pcmci_of_j = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=verbosity)

    # Run PC condition-selection algorithm. Also here further parameters can be
    # specified:
    parents_of_j = pcmci_of_j.run_pc_stable(
        link_assumptions=params['link_assumptions'][j], ########## CHECK THIS ONE selected_variables=[j], selected_links=params['selected_links'][j],
        tau_min=params['tau_min'],
        tau_max=params['tau_max'],
        pc_alpha=params['pc_alpha'],
    )

    # We return also the PCMCI object because it may contain pre-computed 
    # results can be re-used in the MCI step (such as residuals or null
    # distributions)
    return j, pcmci_of_j, parents_of_j


def run_mci_parallel(j, pcmci_of_j, all_parents, params):
    """Wrapper around PCMCI.run_mci step.


    Parameters
    ----------
    j : int
        Variable index.

    pcmci_of_j : object
        PCMCI object for variable j. This may contain pre-computed results 
        (such as residuals or null distributions).

    all_parents : dict
        Dictionary of parents for all variables. Needed for MCI independence
        tests.

    Returns
    -------
    j, results_in_j : tuple
        Variable index and results dictionary containing val_matrix, p_matrix,
        and optionally conf_matrix with non-zero entries only for
        matrix[:,j,:].
    """
    
    results_in_j = pcmci_of_j.run_mci(
        link_assumptions=params['link_assumptions'][j], # selected_links=params['selected_links']
        tau_min=params['tau_min'],
        tau_max=params['tau_max'],
        parents=all_parents,
        max_conds_px=params['max_conds_px'],
    )

    return j, results_in_j


def load_varimax_data(variables, seasons_mask, model_name, n_comps, mask):
    """ Load all the results from PCA-VARIMAX and concatanate by columns the n 
    principal components for the different variables and seasons specified, 
    as well as the mask for the time series."""

    data_dict = {}
    for var, months in itertools.product(variables, seasons_mask):
        # print(model_name[var], seasons_mask[months])
        
        # create file name, load data and save it to a dictionary
        file_name = './runs/train/train_varimax_%s_3dm_comps-%d_months-%s.bin' % (model_name[var], n_comps[var], seasons_mask[months]) # (model_name[0], n_comps, months)
        datadict = pickle.load(open(file_name, 'rb'))
        
        if mask == 'unmasked':
            ts_to_use = 'ts_unmasked'

            data_dict[f'{var}_{months}'] = {'results': datadict['results'], 
                                            'time_mask': datadict['results']['time_mask'], 
                                            'dateseries': datadict['results']['time'][:]}
        elif mask == 'masked':
            ts_to_use = 'ts_masked'

            data_dict[f'{var}_{months}'] = {'results': datadict['results'], 
                                            'time_mask': datadict['results']['time_mask'][~datadict['results']['time_mask']][1::3], 
                                            'dateseries': datadict['results']['time'][~datadict['results']['time_mask']][:]}
            
    
    # Check if the 'dateseries' and 'time_mask' arrays are equal to confirm the time series can be concatenated
    check = []
    for season in seasons_mask.keys():
        for serie in ['dateseries', 'time_mask']:
            if serie == 'dateseries':
                all_equal = (data_dict[variables[0]+'_'+season][serie].shape == data_dict[variables[1]+'_'+season][serie].shape) # and \
                #(data_dict[variables[1]+'_'+season][serie].shape == data_dict[variables[2]+'_'+season][serie].shape)    
            else:
                all_equal = (data_dict[variables[0]+'_'+season][serie] == data_dict[variables[1]+'_'+season][serie]).all() #and \
                #(data_dict[variables[1]+'_'+season][serie] == data_dict[variables[2]+'_'+season][serie]).all()
                    

            check.append(all_equal)
            if not all_equal:
                print('%s in season %s not equal' % (serie, season))
                break

    # Create lists of the data and mask to concatenate
    if all(check):
        concat_list = [
            data_dict[f'{var}_{months}']['results'][ts_to_use]
            for months in seasons_mask.keys()
            for var in variables
        ]

        concat_mask_list = []
        for months in seasons_mask.keys():
            for var in variables:
                T, N = data_dict[f'{var}_{months}']['results'][ts_to_use].shape

                temp_time_mask = data_dict[f'{var}_{months}']['time_mask']
                concat_mask_list.append(np.repeat(temp_time_mask.reshape(T, 1), N,  axis=1))

        # Ensure dimensions match before concatenation
        concat_shapes = [arr.shape[0] for arr in concat_list]
        if len(set(concat_shapes)) != 1:
            raise ValueError("Arrays to concatenate must have the same shape. Found shapes: {}".format(concat_shapes))
        concatenated_data = np.ma.concatenate(concat_list, axis=1)
        concatenated_mask_data = np.ma.concatenate(concat_mask_list, axis=1)

    # If the series are not equal for the same season and for different variables, return None
    else:
        concatenated_data = None
        concatanated_mask_data = None

    return data_dict, concatenated_data, concatenated_mask_data

##### BACKUP #####
# def load_varimax_data(variables, seasons_mask, model_name, n_comps, mask):
#     """ Load all the results from PCA-VARIMAX and concatanate by columns the n 
#     principal components for the different variables and seasons specified, 
#     as well as the mask for the time series."""

#     data_dict = {}
#     for var, months in itertools.product(variables, seasons_mask):
#         # print(model_name[var], seasons_mask[months])
        
#         # create file name, load data and save it to a dictionary
#         file_name = './runs/train/train_varimax_%s_3dm_comps-%d_months-%s.bin' % (model_name[var], n_comps, seasons_mask[months]) # (model_name[0], n_comps, months)
#         datadict = pickle.load(open(file_name, 'rb'))
        
#         if mask == 'unmasked':
#             ts_to_use = 'ts_unmasked'

#             data_dict[f'{var}_{months}'] = {'results': datadict['results'], 
#                                             'time_mask': datadict['results']['time_mask'], 
#                                             'dateseries': datadict['results']['time'][:]}
#         elif mask == 'masked':
#             ts_to_use = 'ts_masked'

#             data_dict[f'{var}_{months}'] = {'results': datadict['results'], 
#                                             'time_mask': datadict['results']['time_mask'][~datadict['results']['time_mask']][1::3], 
#                                             'dateseries': datadict['results']['time'][~datadict['results']['time_mask']][:]}
            
    
#     # Check if the 'dateseries' and 'time_mask' arrays are equal to confirm the time series can be concatenated
#     check = []
#     for season in seasons_mask.keys():
#         for serie in ['dateseries', 'time_mask']:
#             if serie == 'dateseries':
#                 all_equal = (data_dict[variables[0]+'_'+season][serie].shape == data_dict[variables[1]+'_'+season][serie].shape) # and \
#                 #(data_dict[variables[1]+'_'+season][serie].shape == data_dict[variables[2]+'_'+season][serie].shape)    
#             else:
#                 all_equal = (data_dict[variables[0]+'_'+season][serie] == data_dict[variables[1]+'_'+season][serie]).all() #and \
#                 #(data_dict[variables[1]+'_'+season][serie] == data_dict[variables[2]+'_'+season][serie]).all()
                    

#             check.append(all_equal)
#             if not all_equal:
#                 print('%s in season %s not equal' % (serie, season))
#                 break

#     # Create lists of the data and mask to concatenate
#     if all(check):
#         concat_list = [
#             data_dict[f'{var}_{months}']['results'][ts_to_use]
#             for months in seasons_mask.keys()
#             for var in variables
#         ]

#         concat_mask_list = []
#         for months in seasons_mask.keys():
#             for var in variables:
#                 T, N = data_dict[f'{var}_{months}']['results'][ts_to_use].shape

#                 temp_time_mask = data_dict[f'{var}_{months}']['time_mask']
#                 concat_mask_list.append(np.repeat(temp_time_mask.reshape(T, 1), N,  axis=1))

#         # Ensure dimensions match before concatenation
#         concat_shapes = [arr.shape for arr in concat_list]
#         if len(set(concat_shapes)) != 1:
#             raise ValueError("Arrays to concatenate must have the same shape. Found shapes: {}".format(concat_shapes))
#         concatenated_data = np.ma.concatenate(concat_list, axis=1)
#         concatenated_mask_data = np.ma.concatenate(concat_mask_list, axis=1)

#     # If the series are not equal for the same season and for different variables, return None
#     else:
#         concatenated_data = None
#         concatanated_mask_data = None

#     return data_dict, concatenated_data, concatenated_mask_data