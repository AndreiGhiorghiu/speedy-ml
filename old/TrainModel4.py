import os

import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.disable_eager_execution()

# Define the maximum number of words to keep
MAX_NUM_WORDS = 5000

# Define the maximum length of input sequences
MAX_SEQUENCE_TITLE_LENGTH = 100

# Define the maximum length of input sequences
MAX_SEQUENCE_DESCRIPTION_LENGTH = 500

# Load the dataset
directory = os.getcwd()
dataset_dir = directory + '/dataset'
train_dir = dataset_dir + '/train'
test_dir = dataset_dir + '/test'
validation_dir = dataset_dir + '/validation'

train_file = train_dir + '/train_merged_dataset.csv'
test_file = test_dir + '/test_merged_dataset.csv'
validation_file = validation_dir + '/validation_merged_dataset.csv'

df = pd.read_csv(train_file)
df.dropna(inplace=True)

# separate features and labels
X = df[['title', 'description']].values
y = df['storypoint'].values

# tokenizing and padding the text data
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X[:,0])
tokenizer.fit_on_texts(X[:,1])
word_index = tokenizer.word_index

X_title_seq = tokenizer.texts_to_sequences(X[:,0])
X_title_padded = pad_sequences(X_title_seq, maxlen=MAX_SEQUENCE_TITLE_LENGTH, truncating='post')

X_desc_seq = tokenizer.texts_to_sequences(X[:,1])
X_desc_padded = pad_sequences(X_desc_seq, maxlen=MAX_SEQUENCE_DESCRIPTION_LENGTH, truncating='post')

# split data into training and testing sets
X_title_train, X_title_test, X_desc_train, X_desc_test, y_train, y_test = train_test_split(X_title_padded, X_desc_padded, y, test_size=0.1, random_state=42)

# define neural network architecture
inputs1 = tf.keras.layers.Input(shape=(MAX_SEQUENCE_TITLE_LENGTH,))
embedding1 = tf.keras.layers.Embedding(MAX_NUM_WORDS, MAX_SEQUENCE_TITLE_LENGTH)(inputs1)
conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
flat1 = tf.keras.layers.Flatten()(pool1)

inputs2 = tf.keras.layers.Input(shape=(MAX_SEQUENCE_DESCRIPTION_LENGTH,))
embedding2 = tf.keras.layers.Embedding(MAX_NUM_WORDS, MAX_SEQUENCE_TITLE_LENGTH)(inputs2)
conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
flat2 = tf.keras.layers.Flatten()(pool2)

merged = tf.keras.layers.concatenate([flat1, flat2])
dense1 = tf.keras.layers.Dense(10, activation='relu')(merged)
outputs = tf.keras.layers.Dense(1, activation='linear')(dense1)

model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit([X_title_train, X_desc_train], y_train, epochs=50, batch_size=64, validation_split=0.1)

# evaluate model
test_mse_score, test_mae_score = model.evaluate([X_title_test, X_desc_test], y_test)
print('Mean absolute error:', test_mae_score)