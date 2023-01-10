# Author: Brenton Graham
# Description: Common functions used in ML scripts
# Last updated: 01/09/2023


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


def xy_split(df, target: str, to_numpy: bool):
    '''
    Function to split df into features (x) and target (y)
    
    Input
    - target: string corresponding to the target column name
    - to_numpy: boolean flag that can be used to convert x and y outputs to numpy arrays
    '''
    
    # Split data frame
    x = df.loc[:, df.columns != target]
    y = df.loc[:, target]
    
    # Return x, y in specified format
    return (x.to_numpy(), y.to_numpy()) if to_numpy else (x, y)


def flatten_list(nested_list):
    '''
    Function to flatten a list of lists into one list
    e.g. [[1, 0, 0], [0, 0, 1]] -> [1, 0, 0, 0, 0, 1]
    Source: https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    '''
    
    return [item for sublist in nested_list for item in sublist]


def numpy_to_DF(numpyMatrix, columns):
    '''
    Function to convert numpy matrix to data frame
    
    Input
    - numpyMatrix
    - columns: column names, usually extracted from original df (e.g. columns = x_df.columns)
    '''
    
    return pd.DataFrame(numpyMatrix, columns=columns)


def getClfStats(y_true, y_pred, y_prob):
    '''
    Function to evaluate classifier metrics, including AUC, MCC, Precision, Recall, and F1
    '''
    
    statsDict = {}
    statsDict["AUC"] = round(roc_auc_score(y_true, y_prob, average='weighted'), 3)
    statsDict["MCC"] = round(matthews_corrcoef(y_true, y_pred), 3) 
    statsDict["Precision"] = round(precision_score(y_true, y_pred, average='weighted'), 3) 
    statsDict["Recall"] = round(recall_score(y_true, y_pred, average='weighted'), 3)  
    statsDict["F1"] = round(f1_score(y_true, y_pred, average='weighted'), 3) 
    return statsDict


def encodeCategories(x):
    '''
    Function to encode categorical features
    Features with two categories will be encoded as binary
    Features with more than two categories will be one-hot-encoded
    '''
    # Copy df 
    df = x.copy()
    categorialColumnNames_list = df.select_dtypes(include=['object']).columns.values.tolist()
    featureCategoryCount_dict = {feature: len(set(df[feature])) for feature in categorialColumnNames_list}
    
    # Display message
    print('Encoding categorical features...\n')
    
    # Encode feature based on number of categories
    for feature, categoryCount in featureCategoryCount_dict.items():

        # Convert binary feature to binary 0, 1
        if categoryCount == 2:

            # Encode column
            feat0, feat1 = tuple(set(df[feature]))
            df[feature] = df[feature].map({feat0: 0, feat1: 1})

            # Display which category is encoded 0 and 1
            print(feature + ' encoded as binary')
            print(f'   0: {feat0}')
            print(f'   1: {feat1}\n')

        # One-hot-encode feature with more than 2 categories
        elif categoryCount > 2:

            # Code adapted from Antonio Perez, PhD
            # Encode column
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            encoder.fit(df[feature].to_numpy().reshape(-1, 1))
            encodedData = encoder.transform(df[feature].to_numpy().reshape(-1, 1))

            # Copy to data frame
            for i in range(encodedData.shape[1]):
                if type(encoder.categories_[0][i]) == str:
                    df[f'{feature}_{encoder.categories_[0][i]}'] = encodedData[:, i]

            # Drop original column
            df.pop(feature)

            # Display message
            print(f'{feature} one-hot-encoded\n')

        # Safe gaurd against features with only one category
        else:
            pass
        
    return df

