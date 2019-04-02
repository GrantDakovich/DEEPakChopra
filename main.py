import numpy as np
import torch
import pandas as pd
import os
import tensorflow as tf
import csv

ifile = open("cleanedTweets.csv", "r")  # input file containing cleaned Chopra tweets
reader = csv.reader(ifile)  # object to read csv ifile

tweets = []  # array to store tweets

# read in tweets, save to array for further cleaning
for row in reader:
    tweets.append(row)

ifile.close()  # tweets read in, no longer need input file




########## BEGIN CharRNN ##########

# Character mapping to integers
with open("cleanedTweets.csv", 'r', encoding='utf-8') as f:tweets=f.read()
tweetChars = []
tweetChars = set(tweets)
char2int = {ch:i for i,ch in enumerate(tweetChars)}
int2char = dict(enumerate(tweetChars))
text_ints = np.array([char2int[ch] for ch in tweets],dtype=np.int32)

print(text_ints)


# Function for splitting data
def split_data(sequence, batch_size, num_steps):
    total_length = batch_size * num_steps
    num_batches = int(len(sequence) / total_length)
    if num_batches*total_length + 1 > len(sequence):
        num_batches = num_batches - 1
    # Cut down character stream to length of a batch
    inputs = sequence[0: num_batches * total_length]
    output = sequence[1: num_batches * total_length + 1]
    # Split input & output:
    split_input = np.split(inputs, batch_size)
    split_output = np.split(output, batch_size)
    # Combine the batches
    inputs = np.stack(split_input)
    output = np.stack(split_output)
    return inputs, output

def create_batch_generator(data_x, data_y, num_steps):
    batch_size, total_length = data_x.shape
    num_batches = int(total_length/num_steps)
    for b in range(num_batches):
        yield (data_x[:, b*num_steps:(b+1)*num_steps],
               data_y[:, b*num_steps:(b+1)*num_steps])

