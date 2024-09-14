import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

def load_csv_files(file_path):
    df = pd.read_csv(file_path)  
    return df

def split_data(file_path, test_size=0.30, random_state=42):
    df=load_csv_files(file_path)
    X=df['lemma'] 
    y=df['antonyms']
    train_data, test_data = train_test_split(X,y, test_size=test_size, random_state=random_state)
    return train_data, test_data


class CSVDataset(Dataset):
    def __init__(self,dataframe, tokenizer, max_len=150):
        self.dataframe=dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['lemma'])
        output = self.dataframe.iloc[idx]['antonyms']
        encoding_input= self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding_output= self.tokenizer(
            output,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        
        input_ids = encoding_input['input_ids'].squeeze()  
        attention_mask = encoding_input['attention_mask'].squeeze()
        targets_ids=encoding_output['input_ids'].squeeze()  
        
        return input_ids, attention_mask, targets_ids
    


def create_data_loaders(train_data, test_data, tokenizer, batch_size):
    train_dataset = CSVDataset(train_data, tokenizer)
    test_dataset = CSVDataset(test_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_data(file_path, tokenizer, batch_size=32):
    train_data, test_data = split_data(file_path)
    train_loader, test_loader = create_data_loaders(train_data, test_data, tokenizer, batch_size)
    return train_data, test_data, train_loader, test_loader





