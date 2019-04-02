import numpy as np
import pytorch
import pandas as pd
import csv

ifile = open("cleanedTweets.csv", "r")  # input file containing cleaned Chopra tweets
reader = csv.reader(ifile)  # object to read csv ifile

tweets = []  # array to store tweets

# read in tweets, save to array for further cleaning
for row in reader:
    tweets.append(row)

ifile.close()  # tweets read in, no longer need input file
