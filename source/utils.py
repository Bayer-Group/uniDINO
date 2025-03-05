import os
import pandas as pd
import string
import torch
import logging

def ensure_dir(directory:str):
    """
    Create the directory tree in the input path if it does not exist. Input could point to a directory or a file
    """
    is_file = '.' in os.path.split(directory)[-1]
    if is_file:
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def well_map_384(inverted:bool=False) -> dict:
    """
    Maps well names (e.g A01) to well locations (row, column) for a 384 well plate
    """
    Nrows, Ncols = 16, 24
    row_nums = range(1,Nrows+1)
    row_lets = string.ascii_uppercase[0:Nrows]
    well_map = dict()
    for row_num, row_let in zip(row_nums, row_lets) :
        for col in range(1,Ncols+1):
            col_str = ("0"+str(col))[-2:] # add zero to 1digit numbers
            well_map[ (row_num,col) ] = row_let+col_str
    if inverted:
        well_map = dict( zip(well_map.values(), well_map.keys()))
    return well_map

def get_filename_columns(df:pd.DataFrame) ->list:
    """
    Extracts all collumns in input dataframe matching the patern "FileName_"
    :param check_CP_channels: Checks that the channel naming and order match those from Cell Painting
    """
    filename_cols = [c for c in df if 'FileName_' in c and c != 'FileName_Merged']
    return filename_cols

def save_model(filename, model, optimizer=None):
    if optimizer:
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'state_dict': model.state_dict(),
        }
    torch.save(state, filename)


def load_model( filename , model_class ):
    state = torch.load(filename)
    model_class.load_state_dict(state['state_dict'])
    if not 'optimizer' in state.keys():
         state['optimizer'] = None
    return model_class, state['optimizer']


def convert_compartment_format_to_list(compartments):
    """Converts compartment to a list.
    Parameters
    ----------
    compartments : list of str or str
        Cell Painting compartment(s).
    Returns
    -------
    compartments : list of str
        List of Cell Painting compartments.
    """

    if isinstance(compartments, list):
        compartments = [x.lower() for x in compartments]
    elif isinstance(compartments, str):
        compartments = [compartments.lower()]

    return compartments

def get_feature_cols(df, feature_type="standard"):
    """Splits columns of input dataframe into columns contining metadata and columns containing morphological profile features
    :param df: input data frame
    :param features_type:
        "standard" for features named as emb1, emb2 ..
        "CellProfiler": for cell-centric Cell Profiler features names as Cells_ , Nuclei_ , Cytoplasm_
    :return : feature_columns , info_columns
    """
    if feature_type.lower() =="cellprofiler":
        feature_cols = [ c for c in df.columns if (c.startswith("Cells_")|c.startswith("Nuclei_")|c.startswith("Cytoplasm_")) & ("metadata" not in c.lower()) ]
    elif feature_type=="standard":
        feature_cols = [ c for c in df.columns if c.startswith("emb")]
    else:
        raise NotImplementedError("Feature Type not implemented. Options: CellProfiler, standard")
    info_cols =  list( set(df.columns).difference(set(feature_cols)) )
    return feature_cols, info_cols


def get_feature_data(df, feature_type='cellprofiler'):
    feature_cols, _ = get_feature_cols(df, feature_type)
    return df[feature_cols]

def get_dfs(path):
    """
    Get the dataframes from path. Fixes prefixes to work on all available servers.
    Returns the train df and the val df
    """
    f = os.path.join(path)
    data_df = pd.read_csv(f)
    # Take 80% of the data for training and 20% for validation
    n_train = int(0.8 * data_df.shape[0])
    # image field of views are already shuffled,
    # therefore simply take the first n_train for training
    train_df, val_df = data_df.iloc[:n_train,:].reset_index(drop=True), data_df.iloc[n_train:,:].reset_index(drop=True)
    return train_df, val_df

def log_and_print(msg):
    logging.info(msg)
    print(msg)

def assert_(condition, message="Assertion failed"):
    if not condition:
        raise AssertionError(message)