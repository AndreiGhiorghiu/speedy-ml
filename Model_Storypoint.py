from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

MAX_SEQUENCE_LENGTH = 100

# Define the Fibonacci sequence up to 144 (maximum predicted value)
FIBONACCI_SERIES = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

# Load the trained model
model = tf.keras.models.load_model('./model/best_model.h5')

# Define a function to preprocess the input text
def preprocess_text(text):
    # Tokenize the text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(text)
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the input sequences
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
   
    return padded_sequence

# Define a Flask app
app = Flask(__name__)

def convert_to_storypoint(prediction):
    return FIBONACCI_SERIES[prediction]

# Define an endpoint for the API
@app.route('/sp-estimation', methods=['POST'])
def predict():
    # Get the text from the request body
    tasks_titles = request.json['titles']
    
    prediction_results = []
    # Preprocess the text
    for title in tasks_titles:
        processed_text = preprocess_text([title])

        # Make the prediction
        predicted_class = model.predict(processed_text)
        prediction = np.argmax(predicted_class)

        #save result
        prediction_results.append(convert_to_storypoint(prediction))
    
    # Return the prediction as JSON
    return jsonify({'storypoints': prediction_results})

# Start the app
if __name__ == '__main__':
    app.run()