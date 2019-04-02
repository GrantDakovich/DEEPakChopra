import numpy as np
import torch
import pandas as pd
import csv

# Character mapping to integers
with open("cleanedTweets.csv", 'r', encoding='utf-8') as f:tweets=f.read()
tweetChars = []
tweetChars = set(tweets)
char2int = {ch:i for i,ch in enumerate(tweetChars)}
int2char = dict(enumerate(tweetChars))
text_ints = np.array([char2int[ch] for ch in tweets],dtype=np.int32)

print(text_ints)