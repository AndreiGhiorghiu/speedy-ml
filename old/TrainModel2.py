import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.disable_eager_execution()

######## BUILDING THE MODEL ########

# Define the maximum number of words to keep
MAX_NUM_WORDS = 5000

# Define the maximum length of input sequences
MAX_SEQUENCE_LENGTH = 200

# Define the Fibonacci sequence up to 21 (maximum predicted value)
FIBONACCI_SEQ = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946]

# Define the number of output classes
NUM_CLASSES = len(FIBONACCI_SEQ)

directory = os.getcwd()
dataset_dir = directory + '/dataset'
train_dir = dataset_dir + '/train/csv'
test_dir = dataset_dir + '/test'
validation_dir = dataset_dir + '/validation'

train_file = train_dir + '/springxd.csv'
test_file = test_dir + '/test_merged_dataset.csv'
validation_file = validation_dir + '/validation_merged_dataset.csv'

# Read the CSV file
df = pd.read_csv(train_file)
df.dropna(inplace=True)
vf = pd.read_csv(validation_file)
vf.dropna(inplace=True)

# Split the data into training and testing sets
train_df = df
validate_df = vf

# Define the tokenizer object
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

# Fit the tokenizer on the training data
tokenizer.fit_on_texts(train_df['title'] + ' ' + train_df['description'])

# Convert the text data to sequences
train_sequences_title = tokenizer.texts_to_sequences(train_df['title'])
train_sequences_desc = tokenizer.texts_to_sequences(train_df['description'])
validate_sequences_title = tokenizer.texts_to_sequences(validate_df['title'])
validate_sequences_desc = tokenizer.texts_to_sequences(validate_df['description'])

# Pad the sequences to a fixed length
train_padded_title = pad_sequences(train_sequences_title, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
train_padded_desc = pad_sequences(train_sequences_desc, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
validate_padded_title = pad_sequences(validate_sequences_title, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
validate_padded_desc = pad_sequences(validate_sequences_desc, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

def checkIfNumbeIsFibonacci(point):
    if point in FIBONACCI_SEQ:
        return FIBONACCI_SEQ.index(point)
    elif point + 1 in FIBONACCI_SEQ:
        return FIBONACCI_SEQ.index(point+1)
    elif point - 1 in FIBONACCI_SEQ:
        return FIBONACCI_SEQ.index(point-1)
    else:
        return 0

# Convert the output to one-hot encoded vectors
train_df['storypoint'] = list(map(lambda point: checkIfNumbeIsFibonacci(point), train_df['storypoint']))
train_labels = tf.keras.utils.to_categorical(train_df['storypoint'], num_classes=NUM_CLASSES)

validate_df['storypoint'] = list(map(lambda point: checkIfNumbeIsFibonacci(point), validate_df['storypoint']))
validate_labels = tf.keras.utils.to_categorical(validate_df['storypoint'], num_classes=NUM_CLASSES)

# Define the input layers
title_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='title_input')
desc_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='desc_input')

# Define the embedding layers
embedding_layer = Embedding(input_dim=MAX_NUM_WORDS, output_dim=256, input_length=MAX_SEQUENCE_LENGTH)

# Apply the embedding layers to the inputs
title_embed = embedding_layer(title_input)
desc_embed = embedding_layer(desc_input)

# Define the LSTM layers
lstm_layer = LSTM(256)

# Apply the LSTM layers to the embeddings
title_lstm = lstm_layer(title_embed)
desc_lstm = lstm_layer(desc_embed)

# Concatenate the outputs
concat_layer = tf.keras.layers.concatenate([title_lstm, desc_lstm], axis=-1)

# Define the output layer
output_layer = Dense(NUM_CLASSES, activation='softmax')(concat_layer)

# Define the model
model = Model(inputs=[title_input, desc_input], outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([train_padded_title, train_padded_desc], train_labels, validation_data=([validate_padded_title, validate_padded_desc], validate_labels), epochs=10, batch_size=32)

def convert_to_storypoint(y):
    return FIBONACCI_SEQ[y]

############ USING THE MODEL FOR PREDICTING BASE ON EXTERNAL INPUT ###############
input_title = "Update homepage design"
input_desc = "We need to update the design of our homepage to make it more modern and visually appealing"
input_seq_title = tokenizer.texts_to_sequences([input_title])
input_seq_desc = tokenizer.texts_to_sequences([input_desc])
input_padded_title = pad_sequences(input_seq_title, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
input_padded_desc = pad_sequences(input_seq_desc, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
predicted_fibonacci = model.predict([input_padded_title, input_padded_desc])[0][0]
predicted_storypoint = convert_to_storypoint(predicted_fibonacci)

print("The predicted story point is:", predicted_storypoint)