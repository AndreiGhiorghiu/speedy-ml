import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

tf.compat.v1.disable_eager_execution()

# Define the maximum number of words to keep
MAX_NUM_WORDS = 1000

# Define the maximum length of input sequences
MAX_SEQUENCE_LENGTH = 100

OUTPUT_DIM = 100

VALIDATION_TESTS_FRACTION = 0.2

# Define the Fibonacci sequence up to 89 (maximum predicted value)
FIBONACCI_SERIES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

NUM_CLASSES = len(FIBONACCI_SERIES)

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

# Split the dataset into training and validation sets
train_df, validate_df = train_test_split(df, test_size=VALIDATION_TESTS_FRACTION, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_df['title'].values)
X_train = tokenizer.texts_to_sequences(train_df['title'].values)
X_validate = tokenizer.texts_to_sequences(validate_df['title'].values)

# Pad the input sequences to the same length
X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
X_validate = pad_sequences(X_validate, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

# Convert the target variable to Fibonacci sequence
train_df['storypoint'] = list(map(lambda point: FIBONACCI_SERIES.index(point), train_df['storypoint']))
y_train = train_df['storypoint'].values
validate_df['storypoint'] = list(map(lambda point: FIBONACCI_SERIES.index(point), validate_df['storypoint']))
y_validate = validate_df['storypoint'].values

# Convert the target variable to one-hot encoding
y_train = np.eye(NUM_CLASSES)[y_train]
y_validate = np.eye(NUM_CLASSES)[y_validate]

# Define the neural network architecture
model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=OUTPUT_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=32, activation='relu'),
    Dropout(rate=0.2),
    Dense(units=NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=VALIDATION_TESTS_FRACTION, batch_size=32, epochs=50, callbacks=[model_checkpoint])

# Evaluate the model
y_pred = np.argmax(model.predict(X_validate), axis=1)
y_validate = np.argmax(y_validate, axis=1)

mae = mean_absolute_error(y_validate, y_pred)
accuracy = (y_pred == y_validate).mean()

print('Mean absolute error:', mae)
print('Accuracy:', accuracy)

############ USING THE MODEL FOR PREDICTING BASE ON EXTERNAL INPUT ###############
def convert_to_storypoint(prediction):
    return FIBONACCI_SERIES[prediction]

def predict(title):
    # Tokenize the input data
    title_tokens = tokenizer.texts_to_sequences([title])

    # Pad the input sequences
    title_seq = pad_sequences(title_tokens, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
   
    predicted_probabilities = model.predict(title_seq)
    prediction = np.argmax(predicted_probabilities)

    # Convert the index to a Fibonacci number
    predicted_story_point = convert_to_storypoint(prediction)

    return predicted_story_point

# Define function to prompt user for input and make predictions
def predict_from_input():
    while True:
        # Prompt user for input
        title = input("Enter title: ")
    
        # Call the predict() function to make predictions
        predicted_story_point = predict(title)

        # Print the predicted story point
        print("Predicted story point:", predicted_story_point)

# Call the predict_from_input() function to start the loop
predict_from_input()