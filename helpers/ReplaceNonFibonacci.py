import os

import pandas as pd
import numpy as np

# Define a function to find the closest Fibonacci number
def find_closest_fibonacci(n):
    fib_nums = [0, 1]
    while fib_nums[-1] < n:
        fib_nums.append(fib_nums[-1] + fib_nums[-2])
    return min(fib_nums, key=lambda x: abs(x-n))

# Load the dataset
directory = os.getcwd()
dataset_dir = directory + '/dataset'
train_dir = dataset_dir + '/train'
test_dir = dataset_dir + '/test'
validation_dir = dataset_dir + '/validation'

train_file = train_dir + '/train_merged_dataset.csv'
test_file = test_dir + '/test_merged_dataset.csv'
validation_file = validation_dir + '/validation_merged_dataset.csv'

# Load the CSV file
df = pd.read_csv(train_file)
vf = pd.read_csv(validation_file)
tf = pd.read_csv(test_file)

# Replace non-Fibonacci storypoints with the closest Fibonacci number
fib_nums = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
df['storypoint'] = df['storypoint'].apply(lambda x: find_closest_fibonacci(x) if x not in fib_nums else x)
vf['storypoint'] = vf['storypoint'].apply(lambda x: find_closest_fibonacci(x) if x not in fib_nums else x)
tf['storypoint'] = tf['storypoint'].apply(lambda x: find_closest_fibonacci(x) if x not in fib_nums else x)

# Save the updated CSV file
df.to_csv(train_file, index=False)
vf.to_csv(validation_file, index=False)
tf.to_csv(test_file, index=False)