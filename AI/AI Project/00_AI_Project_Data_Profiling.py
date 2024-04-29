# Databricks notebook source
# MAGIC %md
# MAGIC Run Initializer

# COMMAND ----------

# MAGIC %run 
# MAGIC /Workspace/Users/vjosep3@lsu.edu/Initializer

# COMMAND ----------

# MAGIC %md
# MAGIC Import Required Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from cnn_model import CNN  # Assuming you have defined your CNN model in cnn_model.py


# COMMAND ----------

# MAGIC %md
# MAGIC Import the datasets

# COMMAND ----------

# Step 1: Load and preprocess the dataset
df_train = spark.read.format('csv').option("header","true").load('abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/AI/MNIST/sign_mnist_train.csv')
df_test = spark.read.format('csv').option("header","true").load('abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/AI/MNIST/sign_mnist_test.csv')



# COMMAND ----------

display(df_train.take(1))

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# Function to convert label index to alphabet letter
def index_to_alphabet(index):
    if index < 9:
        return chr(index + 65)
    elif index < 34:
        return chr(index + 66)
    else:
        return chr(index + 67)

# Define a dictionary to store one image for each alphabet
alphabet_images = {}

# Iterate through the alphabet letters (A-Y)
for i in range(24):
    # Get the alphabet letter
    alphabet_letter = index_to_alphabet(i)
    
    # Adjust index to skip letter 'J'
    label = i + 1 if i >= 9 else i
    
    # Filter the DataFrame for the current alphabet letter
    filtered_data = df_train.filter(df_train['label'] == label).select(df_train.columns[1:]).take(1)
    
    # Check if any data is found for the current alphabet letter
    if filtered_data:
        # Extract the pixel values for the first image
        image_pixels = filtered_data[0]
        
        # Convert pixel values to numpy array and reshape into a 28x28 matrix
        image_array = np.array([int(val) for val in image_pixels])
        image_matrix = image_array.reshape(28, 28)
        
        # Store the image matrix in the dictionary
        alphabet_images[alphabet_letter] = image_matrix
    else:
        print(f"No data found for alphabet: {alphabet_letter}")

# Display one sample image for each alphabet letter
fig, axes = plt.subplots(6, 4, figsize=(12, 12))  # Adjusted subplot size

for i, (alphabet, image_matrix) in enumerate(alphabet_images.items()):
    ax = axes[i // 4, i % 4]
    ax.imshow(image_matrix, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Alphabet: {alphabet}")

plt.tight_layout()
plt.show()






# COMMAND ----------

# MAGIC %md
# MAGIC Map chart for the training data.

# COMMAND ----------

# MAGIC %md
# MAGIC Creating training, testing datasets along with required lables for both train and test.

# COMMAND ----------

# MAGIC %md
# MAGIC Normalization

# COMMAND ----------

# MAGIC %md
# MAGIC Reshaping

# COMMAND ----------

# MAGIC %md
# MAGIC Preview of the images in the training dataset

# COMMAND ----------

# MAGIC %md
# MAGIC
