import sys
sys.path.append("../")

import os

from globals.globals import *
from torchtext import * # imports vocab

from torchtext import data as dta
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader,Dataset


import pandas as pd
import sqlite3 as sq
import sys


# Ref:- https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
class TextData(Dataset):
    def __init__(self,datapath):
        logger.debug('Inside TextData')
        
        self.sqldata = self.load_sql_data(datapath) 
        
        self.vocab   = vocab.build_vocab_from_iterator(self.yield_text(self.sqldata),specials=['<unk>'],)
        self.vocab.set_default_index(0)
        self.label_map = {
            "positive":1,
            "negative":0
        }
    
    def yield_text(self,data):
        assert isinstance(data,pd.DataFrame), "data should be object of pandas data frame"
        
        for sentence in data.CleanedText.values:
            yield sentence.split()
            
    def load_sql_data(self,datapath):
        """
            Loads data from sql
            
            Prams:
                datapath: path to sqlite file
        """
        data = None
        
        con = sq.connect(datapath)
        data = pd.read_sql_query("SELECT * FROM Reviews",con)
        
        return data
    
    def proper_df(self,data):
        return isinstance(data,pd.DataFrame)
    
    def __len__(self):
        if self.proper_df(self.sqldata):
            logger.info('proper data')
            return len(self.sqldata)
        else:
            logger.info('imporper proper data')

            return len([])
    
    def _map_label(self,label):
        return self.label_map.get(label)
    def __getitem__(self,idx):
        # logger.debug('Inside get item')
        
        ## Loading raw text from the dataframe
        raw_text  = self.sqldata.CleanedText.iloc[idx]
        raw_text  = ' '.join(raw_text.split()[:500])
        raw_label = self.sqldata.Score[idx]
        
        ## maping the label to it respective index
        label = self._map_label(raw_label)
        embded_x = self.vocab.lookup_indices(raw_text.split())
        
        # logger.debug(f'len of raw_text : {len(raw_text)}')
        embded_x = th.tensor(embded_x)
        
        
        ## Label should be 0 or 1
        assert label in self.label_map.values(), f"Label should in {self.label_map.values()}"
        
        # logger.debug(f'returning label {label}')
        return embded_x,label
    
def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = _text #torch.tensor(text_transform(_text))
        text_list.append(processed_text)
    # logger.debug(f'The label list : {label_list}')
    x = pad_sequence(text_list, padding_value=0.0,batch_first=True)
    y = th.tensor(label_list)
    logger.debug(f'Shape of padded sequence : {x.shape}')
    return x,y


datapath = "data/raw/data.sqlite"
dataloader = TextData(datapath)

validation_ratio = 0.2
test_ratio = 0.2
BATCH_SIZE = 32

# positive_len = len(dataloader.sqldata[dataloader.sqldata.Score == 'positive'])
# negative_len = len(dataloader.sqldata[dataloader.sqldata.Score == 'negative'])
# other_len    = len(dataloader.sqldata[~dataloader.sqldata.Score.isin(['negative','positive'])])
total_len    = dataloader.sqldata.shape[0]



logger.info(f'Total number of data points : {total_len}')
# logger.info(f'Total number of positive data points : {positive_len} - {round(positive_len/total_len,2) * 100} %')
# logger.info(f'Total number of negative data points : {negative_len} - {round(negative_len/total_len,2) * 100} %')
# logger.info(f'Total number of other data points : {other_len} - {round(other_len/total_len,2) * 100} %')

test_range = int(total_len * test_ratio)
validation_range = test_range + int(total_len * validation_ratio)

total_inds = [i for i in range(total_len)]

test_inds = total_inds[:test_range]
validate_inds = total_inds[test_range:validation_range]
train_inds = total_inds[validation_range:]

test_range,validation_range

logger.info(f'Total number of train data points {len(train_inds)}')
logger.info(f'Total number of validation data points {len(validate_inds)}')
logger.info(f'Total number of test data points {len(test_inds)}')

train_sampler = SubsetRandomSampler(train_inds)
validation_sampler = SubsetRandomSampler(validate_inds)
test_sampler = SubsetRandomSampler(test_inds)


train_data = DataLoader(dataloader,batch_size=BATCH_SIZE,sampler=train_sampler,collate_fn=collate_batch)
validation_data = DataLoader(dataloader,batch_size=BATCH_SIZE,sampler=validation_sampler,collate_fn=collate_batch)
test_data = DataLoader(dataloader,batch_size=BATCH_SIZE,sampler=test_sampler,collate_fn=collate_batch)

if __name__ == "__main__":
    dataloader = TextData(datapath)
