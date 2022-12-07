import torch

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer


# needed resources

class RNN(nn.Module):
    def __init__(self, embeddings, LSTM_dim, n_layers, bidirectional):
        super().__init__()
        
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.load_state_dict({'weight': embeddings})
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embeddings.shape[1], LSTM_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(LSTM_dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_x):
        # input_x is expected to be of size (example length, batch size) however the axes are
        # flipped so we permute them to be the correct size.
        embedded = self.embedding(input_x.permute(1,0)) 
        # embedded size = (example length, batch size, embedding dimensions)
        output, (hidden, cell) = self.lstm(embedded)
        # hidden size = (number of layers * number of directions, batch size, number of hidden units)
        output = self.dropout(hidden[-1])
        # output size = (batch size, number of hidden units)
        output = self.fc(output)
        # output size = (batch size, 1)
        output = self.sigmoid(output)
        
        return output

class CustomImageDataset(Dataset):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        example = torch.IntTensor(self.examples[idx])
        label = self.labels[idx]
        
        return example, label

def prep_text(text):
    tweet_tokenizer = TweetTokenizer()

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    new_text=''
    text = "i don't. feel good"
    text = ' '.join([x.replace("'", '') for x in tweet_tokenizer.tokenize(text)])
    text = tokenizer.tokenize(text)

    for word in text:
        new_text += " "+lemmatizer.lemmatize(word)
    return new_text
    
def string_to_token(text):
    text_vect = []
    for word in text.split():
        if word in word_to_idx:
            text_vect.append(word_to_idx[word])
    return text_vect
idx_to_word = torch.load("API/API/idx_to_word.list")
word_to_idx = torch.load("API/API/word_to_idx.list")
model = torch.load("API/API/depression2.model")

def predict_depression(text):
    text = text.lower()
    text = prep_text(text)
    text_vect = string_to_token(text)
    print(text_vect)
    text = np.pad(text_vect, (200 - len(text_vect), 0), 'constant') #x_train
    data = []
    data.append([k for k in text]) #x_train_data
    data = np.array(data) #x_train_data
    instance = DataLoader(CustomImageDataset(data,[1])) #dataset into x_train_dataloader
    with torch.no_grad():
        for messages, labels in instance:
            messages = torch.tensor(messages).to('cpu')
            prediction = model(messages)
    return prediction
