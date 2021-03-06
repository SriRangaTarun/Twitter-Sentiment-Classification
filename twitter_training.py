# Use BF16 with PyTorch XLA

!export XLA_USE_BF16=1

# Import necessary libraries

import time
import colored
import argparse
import numpy as np
import pandas as pd
from colored import fg, bg, attr

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from transformers import RobertaModel, RobertaTokenizer

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences as pad

# Define hyperparameters and paths

parser = argparse.ArgumentParser()
parser.add_argument('test_data_path')
parser.add_argument('train_data_path')

args = parser.parse_args()
test_data_path = args.test_data_path
train_data_path = args.train_data_path

EPOCHS = 8
SPLIT = 0.8
MAXLEN = 48
DROP_RATE = 0.225

OUTPUT_UNITS = 3
BATCH_SIZE = 384
LR = (4e-5, 1e-2)
ROBERTA_UNITS = 768
VAL_BATCH_SIZE = 384
MODEL_SAVE_PATH = 'sentiment_model.pt'

# Load data and set random seeds

np.random.seed(42)
torch.manual_seed(42)
test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)

# Define PyTorch dataset to input data to roBERTa

class TweetDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.text = data.text
        self.tokenizer = tokenizer
        self.sentiment = data.sentiment
        self.sentiment_dict = {"positive": 0, "neutral": 1, "negative": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pg, tg = 'post', 'post'
        tweet = str(self.text[i]).strip()
        tweet_ids = self.tokenizer.encode(tweet)

        attention_mask_idx = len(tweet_ids) - 1
        if 0 not in tweet_ids: tweet_ids = 0 + tweet_ids
        tweet_ids = pad([tweet_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)

        attention_mask = np.zeros(MAXLEN)
        attention_mask[1:attention_mask_idx] = 1
        attention_mask = attention_mask.reshape((1, -1))
        if 2 not in tweet_ids: tweet_ids[-1], attention_mask[-1] = 2, 0
            
        sentiment = [self.sentiment_dict[self.sentiment[i]]]
        sentiment = torch.FloatTensor(to_categorical(sentiment, num_classes=3))
        return sentiment, torch.LongTensor(tweet_ids), torch.LongTensor(attention_mask)
    
# Define roBERTa-base model with dropout and dense head
        
class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(DROP_RATE)
        self.roberta = RobertaModel.from_pretrained(model)
        self.dense = nn.Linear(ROBERTA_UNITS, OUTPUT_UNITS)
        
    def forward(self, inp, att):
        inp = inp.view(-1, MAXLEN)
        _, self.feat = self.roberta(inp, att)
        return self.softmax(self.dense(self.drop(self.feat)))
    
# Shuffle train data and split into train and val sets (80/20)
        
train_df = shuffle(train_df)
split = np.int32(SPLIT*len(train_df))
val_df, train_df = train_df[split:], train_df[:split]

# Define tokenizer

model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model)

# Define cross entropy loss function and accuracy for training and evaluation

def cel(inp, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(inp, labels)

def accuracy(inp, target):
    inp_ind = inp.max(axis=1).indices
    target_ind = target.max(axis=1).indices
    return (inp_ind == target_ind).float().sum(axis=0)/len(inp_ind)

# Define function to print metrics during training
    
def print_metric(data, batch, epoch, start, end, metric, typ):

    cs = [211, 212, 213]
    time = np.round(end - start, 1)
    time = "Time: %s{}%s s".format(time)
    fonts = [(fg(c), attr('reset')) for c in cs]
    if typ == "Train": pre = "BATCH %s" + str(batch-1) + "%s  "
    if typ == "Val": pre = "\nEPOCH %s" + str(epoch+1) + "%s  "

    print(pre % fonts[0] , end='')
    t = typ, metric, "%s", data, "%s"
    print("{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])
    
# Train model on TPU using PyTorch XLA
    
global val_losses; global train_losses
global val_accuracies; global train_accuracies

def train_fn():
    size = 1
    torch.manual_seed(42)
    train_df = pd.read_csv(train_data_path)

    train_df = shuffle(train_df)
    split = np.int32(SPLIT*len(train_df))
    val_df, train_df = train_df[split:], train_df[:split]

    val_df = val_df.reset_index(drop=True)
    val_set = TweetDataset(val_df, tokenizer)
    val_sampler = DistributedSampler(val_set, num_replicas=8,
                                     rank=xm.get_ordinal(), shuffle=True)

    train_df = train_df.reset_index(drop=True)
    train_set = TweetDataset(train_df, tokenizer)
    train_sampler = DistributedSampler(train_set, num_replicas=8,
                                       rank=xm.get_ordinal(), shuffle=True)
    
    val_loader = DataLoader(val_set, VAL_BATCH_SIZE,
                            sampler=val_sampler, num_workers=0, drop_last=True)

    train_loader = DataLoader(train_set, BATCH_SIZE,
                              sampler=train_sampler, num_workers=0, drop_last=True)

    device = xm.xla_device()
    network = Roberta().to(device)
    optimizer = Adam([{'params': network.dense.parameters(), 'lr': LR[1]*size},
                      {'params': network.roberta.parameters(), 'lr': LR[0]*size}])

    val_losses, val_accuracies = [], []
    train_losses, train_accuracies = [], []
    
    start = time.time()
    xm.master_print("STARTING TRAINING ...\n")

    for epoch in range(EPOCHS):

        batch = 1
        network.train()
        fonts = (fg(48), attr('reset'))
        xm.master_print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)

        val_parallel = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
        train_parallel = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        
        for train_batch in train_parallel:
            train_targ, train_in, train_att = train_batch
            
            network = network.to(device)
            train_in = train_in.to(device)
            train_att = train_att.to(device)
            train_targ = train_targ.to(device)

            train_preds = network.forward(train_in, train_att)
            train_loss = cel(train_preds, train_targ.squeeze(dim=1))/len(train_in)
            train_accuracy = accuracy(train_preds, train_targ.squeeze(dim=1))/len(train_in)

            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            
            end = time.time()
            batch = batch + 1
            acc = np.round(train_accuracy.item(), 3)
            print_metric(acc, batch, None, start, end, metric="acc", typ="Train")

        val_loss, val_accuracy, val_points = 0, 0, 0

        network.eval()
        with torch.no_grad():
            for val_batch in val_parallel:
                targ, val_in, val_att = val_batch

                targ = targ.to(device)
                val_in = val_in.to(device)
                val_att = val_att.to(device)
                network = network.to(device)
            
                val_points += len(targ)
                pred = network.forward(val_in, val_att)
                val_loss += cel(pred, targ.squeeze(dim=1)).item()
                val_accuracy += accuracy(pred, targ.squeeze(dim=1)).item()
        
        end = time.time()
        val_loss /= val_points
        val_accuracy /= val_points
        acc = xm.mesh_reduce('acc', val_accuracy, lambda x: sum(x)/len(x))
        print_metric(np.round(acc, 3), None, epoch, start, end, metric="acc", typ="Val")
    
        xm.master_print("")
        val_losses.append(val_loss); train_losses.append(train_loss.item())
        val_accuracies.append(val_accuracy); train_accuracies.append(train_accuracy.item())

    xm.master_print("ENDING TRAINING ...")
    xm.save(network.state_dict(), MODEL_SAVE_PATH); del network; gc.collect()

    metric_names = ['val_loss_', 'train_loss_', 'val_acc_', 'train_acc_']
    metric_lists = [val_losses, train_losses, val_accuracies, train_accuracies]
    
    for i, metric_list in enumerate(metric_lists):
        for j, metric_value in enumerate(metric_list):
            torch.save(metric_value, metric_names[i] + str(j) + '.pt')

FLAGS = {}
def _mp_fn(rank, flags): train_fn()
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

# Check testing performance

test_set = TweetDataset(test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=VAL_BATCH_SIZE)

device = xm.xla_device()
test_set = TweetDataset(test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=VAL_BATCH_SIZE)

network.eval()
test_accuracy = 0
with torch.no_grad():
    for test_batch in tqdm(test_loader):
        targ, test_in, test_att = test_batch

        network = network.to(device)
        test_in = test_in.to(device)
        test_att = test_att.to(device)

        targ = targ.squeeze(dim=1)
        pred = network.forward(test_in, test_att)
        test_accuracy = accuracy(pred, targ.to(device))*len(pred)

test_accuracy /= len(test_set)
fonts = (fg(212), attr('reset'))
acc = np.round(test_accuracy.item()*100, 2)
print("{}: {}{}{}".format("Test acc", "%s", str(acc), "%s") % fonts + " %")

# Predict sentiment for random sentences to check performance

def predict_sentiment(tweet):
    pg, tg = 'post', 'post'
    tweet_ids = tokenizer.encode(tweet.strip())
    sent = {0: 'positive', 1: 'neutral', 2: 'negative'}

    att_mask_idx = len(tweet_ids) - 1
    if 0 not in tweet_ids: tweet_ids = 0 + tweet_ids
    tweet_ids = pad([tweet_ids], maxlen=MAXLEN, value=1, padding=pg, truncating=tg)

    att_mask = np.zeros(MAXLEN)
    att_mask[1:att_mask_idx] = 1
    att_mask = att_mask.reshape((1, -1))
    if 2 not in tweet_ids: tweet_ids[-1], att_mask[-1] = 2, 0
    tweet_ids, att_mask = torch.LongTensor(tweet_ids), torch.LongTensor(att_mask)
    return sent[np.argmax(network.forward(tweet_ids.to(device), att_mask.to(device)).detach().cpu().numpy())]

predict_sentiment("It was okay I guess?")
predict_sentiment("I want to hide omg !!!")
predict_sentiment("I am feeling great today ...")

# Move model to CPU and save it to 'sentiment_model.pt'

network = network.cpu()
torch.save(network.state_dict(), MODEL_SAVE_PATH)
