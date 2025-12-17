import os
import pandas as pd
import json
import logging
import pickle
import yaml
from dvclive import Live

import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

log_dir='log'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Evaluation_logs")
logger.setLevel("DEBUG")

console_handeller=logging.StreamHandler()
console_handeller.setLevel("DEBUG")

file_log_path=os.path.join(log_dir,"Model_evaluation.log")
file_handeller=logging.FileHandler(file_log_path)
file_handeller.setLevel("DEBUG")

formatter=logging.Formatter(" %(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handeller.setFormatter(formatter)
console_handeller.setFormatter(formatter)

logger.addHandler(file_handeller)
logger.addHandler(console_handeller)

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


def load_model(file_path:str):
    '''
    Docstring for load_model
    
    :param file_path: path of the model
    :type file_path: str
    '''
    
    try:
        with open (file_path,'rb') as file:
            model=pickle.load(file)
            
        return model
        logger.debug("Model loaded succesfully from: %s",file_path)
    
    except FileNotFoundError as e:
        logger.error("File is not found: %s",e)
        raise
        
    except Exception as e:
        logger.error("Unwanted error is occured")
        raise

def load_data(file_path:str)->pd.DataFrame:
    '''
    Docstring for load_data
    
    :param file_path: path of the data i.e test data
    :type file_path: str
    :return: dataframe of test data
    :rtype: DataFrame
    '''
    try:
        df=pd.read_csv(file_path)
        logger.debug("Sucessfully loaded data from %s", file_path)
        return df
    except FileNotFoundError as e:
        logger.error("file to be read not founnd  %s",e)
    
    except Exception as e:
        logger.error("Unexpected error has occured %s",e)
        
def evaluate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
    '''
    Docstring for evaluate_model
    
    :param clf: Model whose metrices need to be calculated
    :param X_test: Test features
    :type X_test: np.ndarray
    :param y_test: Target columns
    :type y_test: np.ndarray
    :return: Metricis in dictionary format
    :rtype: dict
    '''
    try:
    
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1]
        
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred)
        
        metrices_dict={
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "auc":auc
        }
        logger.debug("Metrices are succefully calculated")
        return metrices_dict
       
    except Exception as e:
        logger.error("Unexpected error has occured as %s",e) 
        
def save_metrices(metrices:dict,file_path:str)->None:
    '''
    Docstring for save_metrices
    
    :param metrices: Metrices to save
    :type metrices: dict
    :param file_path: path of the file where the metrices must be saved
    :type file_path: str
    '''
    
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'w') as file:
            json.dump(metrices,file,indent=4)
            
    except FileNotFoundError as e:
        logger.error("File you want to write doesnt exist as: %s",e)
        
    except Exception as e:
        logger.error("Unexpected errro has occured as %s",e)
        

def main():
    try:
        params=load_params("./params.yaml")
        clf=load_model("./model/model.pkl")
        test_data=load_data("./data/processed/test_tifd.csv")
        X_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        metrices=evaluate_model(clf=clf,X_test=X_test,y_test=y_test)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("Accuracy", metrices["accuracy"])
            live.log_metric("Precision", metrices["precision"])
            live.log_metric("Recall", metrices["recall"])
            live.log_metric("auc",metrices["auc"])
            live.log_params(params)
        save_metrices(metrices=metrices, file_path="./report/metrices.json")
        logger.debug("Metrices are sucessfully saved")
        
    except Exception as e:
        logger.error("unexpected erro has occured as %s ",e)
        
if __name__=="__main__":
    main()