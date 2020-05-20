# Import necessary libraries

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import LongTensor
from transformers import RobertaModel, RobertaTokenizer
from keras.preprocessing.sequence import pad_sequences as pad

# Define hyperparameters and paths

MAXLEN = 48
OUTPUT_UNITS = 3
LR = (4e-5, 1e-2)
DROP_RATE = 0.225
ROBERTA_UNITS = 768

# Load trained model for inference from path

model = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model)

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
    
network = Roberta()
parser = argparse.ArgumentParser()

parser.add_argument('train_model_path')
args = parser.parse_args(); path = args.train_model_path
network.load_state_dict(torch.load(path + 'sentiment_model.pt'))

# Move model to GPU and set it to evaluation mode

network = network.cuda().eval()

# Create inference function to predict sentiment

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
    tweet_ids, att_mask = LongTensor(tweet_ids).cuda(), LongTensor(att_mask).cuda()
    return sent[np.argmax(network.forward(tweet_ids, att_mask).detach().cpu().numpy())]

# Predict sentiment on random tweet

parser.add_argument('tweet')
args = parser.parse_args()
print(predict_sentiment(args.tweet))
