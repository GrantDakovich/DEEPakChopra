import numpy as np
import torch
import pandas as pd
import csv
import tensorflow as tf
import os

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


# Class for character level recurrent neural network
class CharRNN(object):
    # Constructor (note, the sampling parameter is for determining
    # what mode this object is in (training/sampling), and grad_clip
    # is for preventing exploding gradients
    def __init__(self, num_classes, batch_size=64,
                 num_steps=100, lstm_size=128,
                 num_layers=1, learning_rate=0.001,
                 keep_prob=0.5, grad_clip=5,
                 sampling=False):
        # Set variables to values given by parameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.grad_clip = grad_clip
        self.g = tf.Graph()

        with self.g.as_default():
            tf.set_random_seed(123)
            self.build(sampling=sampling)  # builds x and y graphs of data
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()


def build(self, sampling):
    if sampling:
        batch_size, num_steps = 1, 1
    else:
        batch_size = self.batch_size
        num_steps = self.num_steps
    tf_x = tf.placeholder(tf.int32,
                          shape=[batch_size, num_steps],
                          name='tf_x')
    tf_y = tf.placeholder(tf.int32,
                          shape=[batch_size, num_steps],
                          name='tf_y')
    tf_keepprob = tf.placeholder(tf.float32,
                                 name='tf_keepprob')

    # One-hot encoding:
    x_onehot = tf.one_hot(tf_x, depth=self.num_classes)
    y_onehot = tf.one_hot(tf_y, depth=self.num_classes)

    # Build the multi-layer RNN cells
    cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(self.lstm_size),
            output_keep_prob=tf_keepprob)
        for _ in range(self.num_layers)])

    # Define the initial state
    self.initial_state = cells.zero_state(
        batch_size, tf.float32)

    # Run each sequence step through the RNN
    lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
        cells, x_onehot,
        initial_state=self.initial_state)
    print('<< lstm_outputs >>', lstm_outputs)
    seq_output_reshaped = tf.reshape(
        lstm_outputs,
        shape=[-1, self.lstm_size],
        name='seq_output_reshaped')

    logits = tf.layers.dense(
        inputs=seq_output_reshaped,
        units=self.num_classes,
        activation=None,
        name='logits')
    probas = tf.nn.softmax(
        logits,
        name='probabilities')

    y_reshaped = tf.reshape(
        y_onehot,
        shape=[-1, self.num_classes],
        name='y_reshaped')
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y_reshaped),
        name='cost')

    # Gradient clipping to avoid "exploding gradients"
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(cost, tvars),
        self.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        name='train_op')
