import os

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras import Sequential

directory = os.getcwd()
dataset_dir = directory + '/dataset'
train_dir = dataset_dir + '/train'
test_dir = dataset_dir + '/test'
validation_dir = dataset_dir + '/validation'

train_file = train_dir + '/train_merged_dataset.csv'
test_file = test_dir + '/test_merged_dataset.csv'
validation_file = validation_dir + '/validation_merged_dataset.csv'

train_dataset = pd.read_csv(
    train_file,
    names=["issukey", "title", "description", "storypoint"],
    usecols = ['title','description', 'storypoint'],
    skiprows=1,
    skipinitialspace=True
)
train_dataset.dropna(inplace=True)
print(train_dataset)

train_dataset_features = train_dataset.copy()
train_dataset_labels = train_dataset_features.pop('storypoint')
print(train_dataset_features)
print(train_dataset_labels)

inputs = {}

for name, column in train_dataset_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.int
    print(name)
    inputs[name] = Input(shape=(1,), name=name, dtype=dtype)

print(inputs)

preprocessed_inputs = []
for name, input in inputs.items():
    lookup = layers.StringLookup(vocabulary=np.unique(train_dataset_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
train_preprocessing = Model(inputs, preprocessed_inputs_cat)
tf.keras.utils.plot_model(model = train_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

train_features_dict = {name: np.array(value) 
                         for name, value in train_dataset_features.items()}
features_dict = {name:values[:1] for name, values in train_features_dict.items()}
print(train_preprocessing(features_dict))

def train_model(preprocessing_head, inputs):
    body = Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
    return model

print(train_features_dict)
print(train_dataset_labels)

train_model = train_model(train_preprocessing, inputs)
train_model.fit(x=train_features_dict, y=train_dataset_labels, epochs=1)

train_model.save('test')
reloaded = tf.keras.models.load_model('test')

features_dict = {name:values[:1] for name, values in train_features_dict.items()}
print(features_dict)

before = train_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)

result = train_model.predict({
    "title": "Make login page in Android",
    "description": "It must have 2 text boxes and one button",
    })
print(result)