import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import copy
import json
import matplotlib.pyplot as plt
from models import *
# from configs import cfg
import pandas as pd
from nltk.translate import bleu_score

DATA_SET_DIR = '/datasets/cs190f-public/BeerAdvocateDataset/'

START_CHAR = 'BOS'
STOP_CHAR = 'EOS'

CHAR_MAP_FILE = 'char_map.json' 
with open(CHAR_MAP_FILE, 'r') as cmap_file:
    CHAR_MAP = json.load(cmap_file)

def char_to_one_hot(c, char_map=CHAR_MAP):
    char_vec = [0. for i in range(len(char_map))]
    char_vec[char_map[c]] = 1.
    return char_vec

SOS_VEC = char_to_one_hot(START_CHAR)
EOS_VEC = char_to_one_hot(STOP_CHAR)

def load_data(fname):
    df = pd.read_csv(fname)
    return df

def process_train_data(data):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).
    data = data[['beer/style','review/overall','review/text']]
    # For every review
    for index, row in data.iterrows():
        review = []
        for c in str(row['review/text']):
            review.append(char_to_one_hot(c))
        yield review

    
def train_valid_split(data, labels):
    # TODO: Takes in train data and labels as numpy array (or a torch Tensor/ Variable) and
    # splits it into training and validation data.
    raise NotImplementedError
    
    
def process_test_data(data):
    raise NotImplementedError
    
def pad_data(orig_data):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.
    pad_data = []
    longest = 0
    for rev in orig_data:
        longest = max(len(rev), longest)

    for rev in orig_data:
        pad = [copy.deepcopy(EOS_VEC) for i in range(longest-len(rev))]
        pad_data.append(rev.extend(pad))

    return pad_data

    

def train(model, X_train, y_train, X_valid, y_valid, cfg):
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    learing_rate = cfg['learning_rate']
    reg_const = cfg['L2_penalty']]
    num_mini_batches = int(len(X_train)/batch_size)

    val_loss = []
    for epoch in range(epochs):
        total_train_loss = 0.
        print('Starting epoch: %d' % epoch)
        for i in range(0, len(X_train), batch_size):
            end_pos = i+batch_size
            batch_vec = X_train[i:end_pos]
            batch_targ = y_train[i:end_pos]



        # validate the model

        # print statistics
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    raise NotImplementedError
    
    
def save_to_file(outputs, fname):
    with open(fname, 'w') as out_file:
        out_file.write(outputs)
    

if __name__ == "__main__":
    train_data_fname = DATA_SET_DIR+'BeerAdvocate_Train.csv'
    test_data_fname = DATA_SET_DIR+'BeerAdvocate_Test.csv'
    out_fname = 'out.txt'
    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    train_data, train_labels = process_train_data(train_data) # Converting DataFrame to numpy array
    X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels) # Splitting the train data into train-valid data
    X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    
    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)
    
    train(model, X_train, y_train, X_valid, y_valid, cfg) # Train the model
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    save_to_file(outputs, out_fname) # Save the generated outputs to a file
