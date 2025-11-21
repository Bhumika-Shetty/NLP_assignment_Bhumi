import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.data_folder = data_folder
        
        # Initialize T5 tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Process the data
        self.data = self.process_data(data_folder, split, self.tokenizer)
        
        print(f"Loaded {len(self.data)} examples for {split} split")

    def process_data(self, data_folder, split, tokenizer):
        '''
        Process the natural language and SQL data for training/evaluation
        '''
        # Load natural language queries
        nl_file = os.path.join(data_folder, f"{split}.nl")
        sql_file = os.path.join(data_folder, f"{split}.sql")
        
        # Read natural language queries
        with open(nl_file, 'r', encoding='utf-8') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        # Read SQL queries (not available for test split)
        sql_queries = []
        if split != 'test':
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_queries = [line.strip() for line in f.readlines()]
        
        # Prepare data
        processed_data = []
        for i, nl_query in enumerate(nl_queries):
            data_item = {
                'nl_query': nl_query,
                'sql_query': sql_queries[i] if i < len(sql_queries) else None
            }
            processed_data.append(data_item)
        
        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize natural language input (for encoder)
        nl_query = "translate English to SQL: " + item['nl_query']  # T5 task prefix
        encoder_tokens = self.tokenizer(nl_query, truncation=True, max_length=512, 
                                      padding=False, return_tensors=None)
        
        result = {
            'encoder_input_ids': encoder_tokens['input_ids'],
            'encoder_attention_mask': encoder_tokens['attention_mask']
        }
        
        # For training/dev: include SQL target
        if self.split != 'test' and item['sql_query'] is not None:
            # Tokenize SQL target (for decoder)
            sql_query = item['sql_query']
            decoder_tokens = self.tokenizer(sql_query, truncation=True, max_length=512,
                                          padding=False, return_tensors=None)
            
            # Add decoder inputs (shifted right for teacher forcing)
            # T5 uses pad token as the start token
            decoder_input_ids = [self.tokenizer.pad_token_id] + decoder_tokens['input_ids'][:-1]
            decoder_target_ids = decoder_tokens['input_ids']
            
            result.update({
                'decoder_input_ids': decoder_input_ids,
                'decoder_target_ids': decoder_target_ids
            })
        
        return result

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract components from batch
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_masks = [item['encoder_attention_mask'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    decoder_target_ids = [item['decoder_target_ids'] for item in batch]
    
    # Convert to tensors and pad sequences
    encoder_ids = pad_sequence([torch.tensor(seq) for seq in encoder_input_ids], 
                              batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(seq) for seq in encoder_attention_masks], 
                               batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence([torch.tensor(seq) for seq in decoder_input_ids], 
                                 batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([torch.tensor(seq) for seq in decoder_target_ids], 
                                  batch_first=True, padding_value=PAD_IDX)
    
    # Initial decoder input is just the first token of decoder inputs
    initial_decoder_inputs = decoder_inputs[:, :1]  # Shape: [batch_size, 1]
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Extract encoder components from batch
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    encoder_attention_masks = [item['encoder_attention_mask'] for item in batch]
    
    # Convert to tensors and pad sequences
    encoder_ids = pad_sequence([torch.tensor(seq) for seq in encoder_input_ids], 
                              batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([torch.tensor(seq) for seq in encoder_attention_masks], 
                               batch_first=True, padding_value=0)
    
    # For test, we start with pad token as initial decoder input
    batch_size = len(batch)
    initial_decoder_inputs = torch.full((batch_size, 1), PAD_IDX, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x