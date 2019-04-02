import numpy as np
import torch
import pandas as pd
import csv

########## BEGIN CharRNN ##########
# Character mapping to integers
with open("cleanedTweets.csv", 'r', encoding='utf-8') as f:tweets=f.read()
tweetChars = []
tweetChars = set(tweets)
char2int = {ch:i for i,ch in enumerate(tweetChars)}
int2char = dict(enumerate(tweetChars))
text_ints = np.array([char2int[ch] for ch in tweets],dtype=np.int32)

print(text_ints)

# Splitting data
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
