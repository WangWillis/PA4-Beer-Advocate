import time
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
from configs import cfg
import pandas as pd
from nltk.translate import bleu_score

import gc

DATA_SET_DIR = '/datasets/cs190f-public/BeerAdvocateDataset/'

START_CHAR = 'BOS'
STOP_CHAR = 'EOS'

CHAR_MAP_FILE = 'char_map.json' 
with open(CHAR_MAP_FILE, 'r') as cmap_file:
    CHAR_MAP = json.load(cmap_file)

BEER_STYLE_MAP_FILE = 'beer_styles.json' 
with open(BEER_STYLE_MAP_FILE, 'r') as bmap_file:
    BEER_MAP = json.load(bmap_file)

def dict_to_one_hot(c, char_map=CHAR_MAP):
    char_vec = [0 for i in range(len(char_map))]
    char_vec[char_map[c]] = 1
    return char_vec

def gen_one_hots(char_map=CHAR_MAP):
    one_hot_dict = {}
    for key in char_map:
        one_hot_dict[key] = dict_to_one_hot(key, char_map)
    return one_hot_dict

ONE_HOT_DICT = gen_one_hots()
BEER_STYLE_DICT = gen_one_hots(BEER_MAP)

SOS_VEC = ONE_HOT_DICT[START_CHAR]
EOS_VEC = ONE_HOT_DICT[STOP_CHAR]

def load_data(fname):
    df = pd.read_csv(fname)
    return df

def process_train_data(data):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).
    data = data[['beer/style','review/overall','review/text']]
    feats = []
    targs = []
    # For every review
    for index, row in data.iterrows():
        beer_style_vec = BEER_STYLE_DICT[row['beer/style']]
        r_score = [float(row['review/overall'])]
        start_feat = SOS_VEC+beer_style_vec+r_score

        feat = [start_feat]
        targ = []
        for c in str(row['review/text']):
            if (c in ONE_HOT_DICT):
                feat_vec = ONE_HOT_DICT[c]+beer_style_vec+r_score

                feat.append(feat_vec)
                targ.append(CHAR_MAP[c])
                # del feat_vec

        targ.append(CHAR_MAP[STOP_CHAR])
        feats.append(feat)
        targs.append(targ)

        # del beer_style_vec, r_score, feat, targ, start_feat
    # del data

    return feats, targs

    
def train_valid_split(data, split_perc=0.80):
    split_idx = int(split_perc*data.shape[0])
    data = data.sample(frac=1).reset_index(drop=True)
    return data[:split_idx], data[split_idx:] 
    
def process_test_data(data):
    data = data[['beer/style','review/overall']]
    feats = []
    # # For every review
    # for index, row in data.iterrows():

        metadata = []
        metadata.extend(BEER_STYLE_DICT[row['beer/style']])
        metadata.append(float(row['review/overall']))

    #     review = [SOS_VEC + metadata]

    #     feats.append(review)
    return feats

EOS_PAD = EOS_VEC+[0 for i in range(len(BEER_STYLE_DICT))]+[0]
MAX_TRAIN_LEN = 10000
def pad_data(orig_data, targ=False):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.
    eos_vec = EOS_PAD
    if (targ):
        eos_vec = CHAR_MAP[STOP_CHAR]
    pad_arr = [eos_vec for i in range(MAX_TRAIN_LEN)]

    longest = 0
    for rev in orig_data:
        longest = max(len(rev), longest)

    for rev in orig_data:
        rev.extend(pad_arr[:longest-len(rev)])
    tensor = torch.tensor(orig_data)
    # del orig_data, pad_arr, longest, eos_vec
    return tensor

CHECK_SIZE = 10
VOCAB_SIZE = len(ONE_HOT_DICT)
def train(model, train_data, val_data, cfg):
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    learning_rate = cfg['learning_rate']
    reg_const = cfg['L2_penalty']

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    # del learning_rate, reg_const
    loss_func = nn.CrossEntropyLoss()

    val_loss = []
    for epoch in range(epochs):
        total_train_loss = 0.
        print('Starting epoch: %d' % epoch)
        for i in range(0, len(train_data), batch_size):
            mini_batch = int(i/batch_size)
            end_pos = i+batch_size

            batch_data = train_data[i:end_pos]
            # del end_pos
            # so it does not have to pad every time in future epochs
            batch_vec, batch_targ = process_train_data(batch_data)
            batch_vec = pad_data(batch_vec)
            batch_targ = pad_data(batch_targ, targ=True)

            opt.zero_grad()

            out = model(batch_vec, train=True, init_hidden=True)
            flat_out = out.contiguous().view(-1,VOCAB_SIZE).cuda()
            flat_targ = batch_targ.contiguous().view(-1).cuda()
            loss = loss_func(flat_out, flat_targ)
            total_train_loss += float(loss)
            
            loss.backward()
            opt.step()

            # del loss, out, batch_vec, batch_targ, flat_out, flat_targ, batch_data

            if (mini_batch % CHECK_SIZE == 0 and mini_batch != 0):
                # validate the model
                tot_val_loss = 0.
                for j in range(0, len(val_data), batch_size):
                    end_pos = j+batch_size
                    batch_data = val_data[j:end_pos]
                    # del end_pos

                    batch_vec, batch_targ = process_train_data(batch_data)
                    batch_vec = pad_data(batch_vec)
                    batch_targ = pad_data(batch_targ, targ=True)

                    v_out = model(batch_vec, train=True, init_hidden=True)
                    flat_out = v_out.contiguous().view(-1,VOCAB_SIZE).cuda()
                    flat_targ = batch_targ.contiguous().view(-1).cuda()
                    v_loss = loss_func(flat_out, flat_targ)
                    tot_val_loss += float(v_loss)

                    # del batch_vec, batch_targ, v_out, flat_out, v_loss, flat_targ, batch_data

                avg_val_loss = tot_val_loss/(len(val_data)-(len(val_data)%batch_size))
                avg_train_loss = total_train_loss/CHECK_SIZE

                total_train_loss = 0.

                val_loss.append(avg_val_loss)

                # print statistics
                print('Epoch %d, Mini_Batch %d' % (epoch, mini_batch))
                print('Average Train Loss: %.8f, Validation Loss: %.8f' % (avg_train_loss, avg_val_loss))

                # del  avg_val_loss, avg_train_loss

            # del mini_batch
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    raise NotImplementedError
    
    
def save_to_file(outputs, fname):
    with open(fname, 'w') as out_file:
        out_file.write(outputs)

if cfg['cuda']:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DATA_PERC = 0.05
if __name__ == "__main__":
    train_data_fname = DATA_SET_DIR+'BeerAdvocate_Train.csv'
    test_data_fname = DATA_SET_DIR+'BeerAdvocate_Test.csv'
    out_fname = 'out.txt'
    
    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    train_data = train_data.sample(frac=DATA_PERC).reset_index(drop=True)
    test_data = test_data.sample(frac=0.08*DATA_PERC).reset_index(drop=True)

    print('Train Size: %d, Test Size: %d' % (train_data.shape[0], test_data.shape[0]))

    train_data,  val_data = train_valid_split(train_data) # Splitting the train data into train-valid data
    # X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    
    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    model.to(DEVICE)
    
    train(model, train_data,  val_data, cfg) # Train the model
    outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    save_to_file(outputs, out_fname) # Save the generated outputs to a file
