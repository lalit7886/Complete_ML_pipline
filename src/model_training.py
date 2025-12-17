import os
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir='log'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model building')
logger.setLevel("DEBUG")

console_handeler=logging.StreamHandler()
console_handeler.setLevel("DEBUG")

log_path=os.path.join(log_dir,"model_building.log")
file_handeler=logging.FileHandler(log_path)
file_handeler.setLevel("DEBUG")

formater=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")
console_handeler.setFormatter(formater)
file_handeler.setFormatter(formater)

logger.addHandler(console_handeler)
logger.addHandler(file_handeler)


def load_params(params_path:str)->dict:
    '''
    Docstring for load_params
    
    :param params_path: path of your parameter yaml file
    :type params_path: str
    :return: dictionary of parameter of this python file
    :rtype: dict
    '''
    try:
        with open(params_path , 'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameter are succesfully loaded")
        return params
    
    except FileNotFoundError as e:
        logger.error("File not found error has occured as %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error has occured as %s",e)
        raise


def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s of shape %s",file_path,df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file. : %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("File is not found %s",e)
        raise
        
    except Exception as e:
        logger.error("Unexpected error occured")
        raise
    
def train_model(x_train:np.ndarray,y_train:np.ndarray,param:dict)->RandomForestClassifier:
    '''
    Docstring for train_model
    
    :param x_train: Features of training data
    :type x_train: np.ndarray
    :param y_train: Target value
    :type y_train: np.ndarray
    :param param: parameter for Randomforest classifier
    :type param: dict
    :return: Description
    :rtype: RandomForestClassifier
    '''
    
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The train column and targer column must have same shape")
        logger.debug("initializing RandomForestClassifier with parameters %s",param)
        clf=RandomForestClassifier(n_estimators=param["n_estimators"],random_state=param["random_state"])
        logger.debug("model training started with %d samples", x_train.shape[0])
        clf.fit(x_train,y_train)
        logger.debug('Model training completed')
        return clf
    except ValueError as e:
        logger.debug("value error has occured in train model as :%s",e)
        raise
    except Exception as e:
        logger.debug("unwxpected error has occured as %s",e)
        raise
        
def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open (file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug("file save to path :%s",file_path)
    
    except FileNotFoundError as e:
        logger.debug("File not found : %s",e)
        raise
    except Exception as e:
        logger.debug("Unexpected error occured : %s",e)
        raise
    
def main():
    try:
        param=load_params("./params.yaml")["model_training"]
        train_data=load_data('./data/processed/train_tfid.csv')
        X_train=train_data.iloc[:,:-1].values #.values convert the selected part into numpy array
        y_train=train_data.iloc[:,-1].values
        
        clf=train_model(x_train=X_train,y_train=y_train,param=param)
        model_save_path='model/model.pkl'
        save_model(clf,model_save_path)
        
    except Exception as e:
        logger.error('Failed to complete model building %s',e)
        print(f"Error:{e}")
        
if __name__=='__main__':
    main()
        
        
        
        
