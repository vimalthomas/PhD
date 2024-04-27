# Databricks notebook source
# MAGIC %md
# MAGIC Run Initializer

# COMMAND ----------

# MAGIC %run 
# MAGIC /Workspace/Users/vjosep3@lsu.edu/Initializer

# COMMAND ----------

# MAGIC %md
# MAGIC Load CNN Model
# MAGIC

# COMMAND ----------

# MAGIC %run 
# MAGIC "/Workspace/Users/vjosep3@lsu.edu/AI Project/01_AI_Project_CNN_Regularized_Model"

# COMMAND ----------

# MAGIC %md
# MAGIC Load CNN Model For Execution

# COMMAND ----------

import torch
import torch.nn as nn
import io

# Load the model DataFrame from Parquet file
abfss_model_path = "abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/Models/CNN_pytorch/model_data/"
df_model_in = spark.read.parquet(abfss_model_path)

# Convert the DataFrame back to a PySpark DataFrame
spark_df_model = df_model_in

# Extract the binary data from the DataFrame
model_binary_data = spark_df_model.select("model_binary").collect()[0][0]

# Deserialize the model from the binary data
loaded_model = torch.load(io.BytesIO(model_binary_data))

# Instantiate the model class and load the state dictionary
model = RegularizedCNN(num_classes=25)  # Assuming you have 25 sign language alphabets
model.load_state_dict(loaded_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Functions for pre-processing images. Since, the model was trained on MNIST format data.

# COMMAND ----------

import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define the function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to match the input size of the model
        transforms.Grayscale(),  # Convert the image to grayscale
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the pixel values
    ])
    # Apply the transformation
    preprocessed_image = transform(image)
    # Add a batch dimension
    preprocessed_image = preprocessed_image.unsqueeze(0)
    return preprocessed_image

# Define the function to perform prediction
def predict_image(model, image):
    # Pass the preprocessed image through the model
    with torch.no_grad():
        outputs = model(image)
        # Get the predicted class probabilities
        probabilities = torch.softmax(outputs, dim=1)
        # Get the predicted class index
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class



# COMMAND ----------

# MAGIC %md
# MAGIC Load Images for the model execution

# COMMAND ----------

# Load the image DataFrame from Parquet file
df_images = spark.read.format('image').load("abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/AI/test_images/")

# Get the file names from the DataFrame

display(df_images.select('image.origin','image'))







# COMMAND ----------

# MAGIC %md
# MAGIC Run the model for the given images. Use a mapper list to map the indexes with the letters. Both J and Z are avoided as they involve gesture movements.

# COMMAND ----------

for filelist in (df_images.select('image.origin').collect()):
    print(filelist[0])
    binary_data = spark.read.format("binaryFile").load(filelist[0]).select("content").collect()[0][0]
    
    pil_image = Image.open(io.BytesIO(binary_data))

    # Preprocess the image
    preprocessed_image = preprocess_image(pil_image)

    # Perform prediction
    predicted_class = predict_image(model, preprocessed_image)
    print("Predicted class index:", predicted_class)
    sign_language_mapping = ['A', 'B', 'C', 'D','E','F','G','H','I','K','L','M','N','O', 'P', 'Q','R' ,'S','T','U','V','W','X', 'Y']
    predicted_alphabet = sign_language_mapping[predicted_class]
    print("Predicted alphabet:", predicted_alphabet)
