import numpy as np
import torch
import pandas as pd
import os
import tensorflow as tf
import csv

<<<<<<< HEAD
ifile = open("cleanedTweets.csv", "r")  # input file containing cleaned Chopra tweets
reader = csv.reader(ifile)  # object to read csv ifile

tweets = []  # array to store tweets

# read in tweets, save to array for further cleaning
for row in reader:
    tweets.append(row)

ifile.close()  # tweets read in, no longer need input file


=======
# Character mapping to integers
with open("cleanedTweets.csv", 'r', encoding='utf-8') as f:tweets=f.read()
tweetChars = []
tweetChars = set(tweets)
char2int = {ch:i for i,ch in enumerate(tweetChars)}
int2char = dict(enumerate(tweetChars))
text_ints = np.array([char2int[ch] for ch in tweets],dtype=np.int32)

print(text_ints)
>>>>>>> master
