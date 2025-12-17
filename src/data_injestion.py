import pandas as pd
import logging
import os
import yaml
from sklearn.model_selection import train_test_split
# Ensuring the log directories exist
log_dir= 'log'
os.makedirs(log_dir, exist_ok=True)

# Logging confuguration
logger=logging.getLogger('data_injestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir, "data_injestion.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    '''
    Docstring for load_params
    
    :param params_path: path of parameter.yaml
    '''
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
            logger.debug("Parameter loaded sucessfully")
            return params
    except FileNotFoundError as e:
        logger.debug("File not found error has occured as %s",e)
        raise
    except Exception as e:
        logger.debug("Unexpected error has occured as %s",e)
        raise

def load_data(data_url:str)->pd.DataFrame:
    '''load dataset from given URL and retruns it'''
    try:
        df=pd.read_csv(data_url)
        logger.debug("Data loaded successfully")
        return df
        
    except pd.errors.ParserError as e:
        logger.error("failed to parsethe csv file: %s",e)
        raise
    
    except Exception as e:
        logger.error("unexpected error happened %s", e)
        raise
        

def pre_process_data(df:pd.DataFrame)->pd.DataFrame:
    '''Performs initial level data preprocessing. Removes three coumns from data '''
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("data preprocessing completed")
        return df
    
    except KeyError as e:
        logger.error("Missing columns: %s",e)
        raise
    
    except Exception as e:
        logger.error("unexpected error happened %s",e)
        raise
    
def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame,data_path:str)->None:
    '''Saves a train data and test data separetly to directory by data path/raw'''
    try:
        raw_data_path=os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("Train and Test data saved to %s", raw_data_path)
        
    except Exception as e:
        logger.error("Unexpected error occured %s",e)
        raise
    
    
def main():
    try:
        params=load_params('./params.yaml')
        test_size= params['data_injestion']['test_size']
        data_path="https://raw.githubusercontent.com/lalit7886/Data_set/refs/heads/main/spam.csv"
        df=load_data(data_url=data_path)
        final_df=pre_process_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='data')
        
    except Exception as e:
            logger.error('failed to complete injestion %s', e)
            print(f"Error: {e}")
            
if __name__ == "__main__":
            main()