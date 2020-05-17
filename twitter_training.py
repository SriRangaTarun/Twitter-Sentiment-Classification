!export XLA_USE_BF16=1

import time
import colored
import numpy as np
import pandas as pd
from colored import fg, bg, attr

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

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

np.random.seed(42)
torch.manual_seed(42)
test_df = pd.read_csv(test_data_path)
train_df = pd.read_csv(train_data_path)

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
        
train_df = shuffle(train_df)
split = np.int32(SPLIT*len(train_df))
val_df, train_df = train_df[split:], train_df[:split]

model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model)

def cel(inp, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(inp, labels)

def accuracy(inp, target):
    inp_ind = inp.max(axis=1).indices
    target_ind = target.max(axis=1).indices
    return (inp_ind == target_ind).float().sum(axis=0)/len(inp_ind)
    
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
    
device = xm.xla_device()

val_df = val_df.reset_index(drop=True)
val_set = TweetDataset(val_df, tokenizer)
val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE)

train_df = train_df.reset_index(drop=True)
train_set = TweetDataset(train_df, tokenizer)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
network = Roberta().to(device)
optimizer = Adam([{'params': network.dense.parameters(), 'lr': LR[1]},
                  {'params': network.roberta.parameters(), 'lr': LR[0]}])

val_losses, val_accuracies = [], []
train_losses, train_accuracies = [], []
    
start = time.time()
print("STARTING TRAINING ...\n")

for epoch in range(EPOCHS):

    batch = 1
    network.train()
    fonts = (fg(48), attr('reset'))
    print(("EPOCH %s" + str(epoch+1) + "%s") % fonts)
        
    for train_batch in train_loader:
        train_targ, train_in, train_att = train_batch
            
        network = network.to(device)
        train_in = train_in.to(device)
        train_att = train_att.to(device)
        train_targ = train_targ.to(device)

        train_preds = network.forward(train_in, train_att)
        train_loss = cel(train_preds, train_targ.squeeze(dim=1))
        train_accuracy = accuracy(train_preds, train_targ.squeeze(dim=1))

        optimizer.zero_grad()
        train_loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
            
        end = time.time()
        batch = batch + 1
        acc = np.round(train_accuracy.item(), 3)
        print_metric(acc, batch, None, start, end, metric="acc", typ="Train")

    network.eval()
    val_loss, val_accuracy = 0, 0
    for val_batch in tqdm(val_loader):
        targ, val_in, val_att = val_batch

        with torch.no_grad():
            targ = targ.to(device)
            val_in = val_in.to(device)
            val_att = val_att.to(device)
            network = network.to(device)

            pred = network.forward(val_in, val_att)
            val_loss += cel(pred, targ.squeeze(dim=1)).item()*len(pred)
            val_accuracy += accuracy(pred, targ.squeeze(dim=1)).item()*len(pred)
        
    end = time.time()
    val_loss /= len(val_set)
    val_accuracy /= len(val_set)
    acc = np.round(val_accuracy, 3)
    print_metric(acc, None, epoch, start, end, metric="acc", typ="Val")
    
    print("")
    val_losses.append(val_loss); train_losses.append(train_loss)
    val_accuracies.append(val_accuracy); train_accuracies.append(train_accuracy)
    
print("ENDING TRAINING ...")

test_set = TweetDataset(test_df, tokenizer)
test_loader = DataLoader(test_set, batch_size=VAL_BATCH_SIZE)

network.eval()
for test_batch in tqdm(test_loader):
    test_preds, test_targs = [], []
    targ, test_in, test_att = test_batch

    with torch.no_grad():
        network = network.to(device)
        test_in = test_in.to(device)
        test_att = test_att.to(device)
        pred = network.forward(test_in, test_att)
        test_preds.append(pred); test_targs.append(targ)

test_preds = torch.cat(test_preds, axis=0)
test_targs = torch.cat(test_targs, axis=0)
test_accuracy = accuracy(test_preds, test_targs.squeeze(dim=1).to(device))

predict_sentiment("It was okay I guess?")
predict_sentiment("I want to hide omg !!!")
predict_sentiment("I am feeling great today ...")

network = network.cpu()
torch.save(network.state_dict(), MODEL_SAVE_PATH)
