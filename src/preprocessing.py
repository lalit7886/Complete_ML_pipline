import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt_tab',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)
stopwords=set(stopwords.words('english'))

# Making log directories
log_dir="log"
os.makedirs(log_dir,exist_ok=True)

logger= logging.getLogger("Preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"Datapreprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    '''This function converts the sentence into lowercase, remove non-alphanumeric, remove puntuations
    and stopword, and returns new  preprocessed sentence '''
    try:
        ps=PorterStemmer()
        text=text.lower()
        text=nltk.word_tokenize(text)
        
        text=[word for word in text if word.isalnum()]
        text=[word for word in text if word not in stopwords and word not in string.punctuation]
        
        text = [ps.stem(word) for word in text]
        return ' '.join(text)
    except Exception as e:
        logger.error("Unexpected error has occured during text transformation %s",e)
        raise
    

def preprocess_df(df, text_columns='text',target_columns='target'):
    '''This function apply label encoding in target column, removes duplicate data and apply 
    transformation in text columns'''
    try:
        logger.debug("starting preprocessing of dataframe")
        
        encoder=LabelEncoder()
        df[target_columns]=encoder.fit_transform(df[target_columns])
        logger.debug('Target column encoded')
        
        df=df.drop_duplicates(keep='first')
        logger.debug("Duplicate data are removed")
        
        df.loc[:,text_columns]=df[text_columns].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error("Column not found : %s", e)
        raise
        
    except Exception as e:
        logger.error("Unexpected error occured during normalization %s",e)
        raise

def main(text_column="text", target_columns="target"):
    try:
        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        
        train_processed_data=preprocess_df(train_data,text_columns=text_column,target_columns=target_columns)
        test_processed_data=preprocess_df(test_data,text_columns=text_column,target_columns=target_columns)
        
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
        
        logger.debug("Train processed and test processed data are saved sucessfully at %s", data_path)
        
    except FileNotFoundError as e:
        logger.error("The file is not found: %s",e)
        
    except Exception as e:
        logger.error("The unwanted error has occured %s",e)
        
if __name__=="__main__":
    main()