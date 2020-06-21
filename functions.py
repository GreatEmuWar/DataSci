import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def separate_dtypes(data):

    catg_data = data.select_dtypes(include=['object'])
    numr_data = data.select_dtypes(include=['int64','float64'])

    return [catg_data, numr_data]



def encodeCategoricals(catg_df, replace=False, replace_df=None, fillna=False, value=None):

    if fillna == False:
        
        master_keys = {}
        
        for column in set(catg_df.columns):
            
            code = {}
            cols_unq = list(catg_df['column'].unique())
            
            i = 0
            
            for x in cols_unq:
                
                code.update({str(x):i})
                i += 1
            
            master_keys.update({str(column):code})
            
        for column in set(catg_df.columns):
            
            catg_df[str(column)].replace(master_keys.get(str(column)), inplace=True)
            
        if replace == False:
            
            return [catg_df, master_keys]
        
        else:
            
            for column in list(replace_df.columns):
                
                if column in set(master_keys.keys()):
                    
                    replace_df[str(column)] = catg_df[str(column)]
                
                else:
                    
                    continue
            
            return [replace_df, master_keys]
    else:
        
        catg_df.fillna(value=value, inplace=True)
        
        master_keys = {}
        
        for column in set(catg_df.columns):
            
            code = {}
            cols_unq = list(catg_df[str(column)].unique())
            
            i = 0
            
            for x in cols_unq:
                
                code.update({str(x):i})
                i += 1
            
            master_keys.update({str(column):code})
            
        for column in set(catg_df.columns):
            
            catg_df[str(column)].replace(master_keys.get(str(column)), inplace=True)
            
        if replace == False:
            
            return [catg_df, master_keys]
        
        else:
            
            for column in list(replace_df.columns):
                
                if column in set(master_keys.keys()):
                    
                    replace_df[str(column)] = catg_df[str(column)]
                
                else:
                    
                    continue
            
            return [replace_df, master_keys]

def imputeNumericalNaN(numerical_cols, replace=False, repl_df=''): 
    
    if replace == False:
        
        
        
        for column in set(numerical_cols.columns):
            
            mean = numerical_cols[str(column)].mean()
            numerical_cols[str(column)].fillna(value=mean, inplace=True)
            
        return numerical_cols          
        
    else:
        
        for column in set(numerical_cols.columns):
            
            mean = numerical_cols[str(column)].mean()
            numerical_cols[str(column)].fillna(value=mean, inplace=True)
        
        for column in set(repl_df.columns):
            
            if column in set(numerical_cols.columns):
                repl_df[str(column)] = numerical_cols[str(column)]
            else:
                continue

        return repl_df


def trainModels(x, y, random=0, model=None):

    training_x, value_x, training_y, value_y = train_test_split(x, y, random_state=random)
    model.fit(training_x, training_y)
    predictions = model.predict(value_x)
    score = mean_absolute_error(value_y,predictions)
    return_obj = {
        
        "train_x" : training_x,
        "train_y" : training_y,
        "value_x" : value_x,
        "value_y" : value_y,
        "predictions" : predictions,
        "model" : model,
        "score" : score
        
    }
    
    return return_obj