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

# MAGIC %md
# MAGIC Map chart for the training data.

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Creating training, testing datasets along with required lables for both train and test.

# COMMAND ----------


# Step 1: Load and preprocess the dataset
df_train1 = df_train.toPandas()
df_test1 = df_test.toPandas()


# Extract features (pixels) and labels
X_train = df_train1.drop(columns=["label"]).values.astype(np.float32)
y_train = df_train1["label"].values
X_test = df_test1.drop(columns=["label"]).values.astype(np.float32)
y_test = df_test1["label"].values

# COMMAND ----------

# MAGIC %md
# MAGIC Normalization

# COMMAND ----------

# Normalize pixel values to range [0, 1]
X_train /= 255.0
X_test /= 255.0

# COMMAND ----------

# MAGIC %md
# MAGIC Reshaping

# COMMAND ----------

# Reshape features into images (assuming 28x28 pixels)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)


# COMMAND ----------

import numpy as np

# Check data types of y_train and y_test
print("Data type of y_train:", y_train.dtype)
print("Data type of y_test:", y_test.dtype)

# Convert data types to ensure they contain only numerical values
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# COMMAND ----------

# MAGIC %md
# MAGIC Preview of the images in the training dataset

# COMMAND ----------

import matplotlib.pyplot as plt

# Display the first 10 images
for i in range(4):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Defining custom datasets

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Step 2: Define a custom dataset class
class CustomSignLanguageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    
    




# COMMAND ----------

# MAGIC %md
# MAGIC Loading the datasets into data loaders.

# COMMAND ----------

# Step 3: Instantiate the dataset and DataLoader
train_dataset = CustomSignLanguageDataset(X_train, y_train)
test_dataset = CustomSignLanguageDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# COMMAND ----------

# MAGIC %md
# MAGIC define and load model class

# COMMAND ----------

# MAGIC %run 
# MAGIC "/Workspace/Users/vjosep3@lsu.edu/AI Project/01_AI_Project_CNN_Regularized_Model"

# COMMAND ----------

# MAGIC
# MAGIC
# MAGIC %run 
# MAGIC "/Workspace/Users/vjosep3@lsu.edu/AI Project/01_AI_Project_CNN_Regularized_Model"
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt

# Lists to store loss and accuracy values
train_losses = []
train_accuracies = []

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        reg_loss = model.l2_regularization_loss()
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item() * images.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = correct_predictions / total_predictions
    
    # Store loss and accuracy values
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Plot loss and accuracy graphs
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Model Evaluation

# COMMAND ----------


# Evaluate the model on the test set
model.eval()
test_predictions = []
test_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.tolist())
        test_true_labels.extend(labels.tolist())

# Convert test_predictions and test_true_labels to NumPy arrays
test_predictions = np.array(test_predictions)
test_true_labels = np.array(test_true_labels)

# Display a sample of test images along with their true and predicted labels
sample_indices = np.random.choice(len(test_predictions), size=10, replace=False)
sample_images = X_test[sample_indices]
sample_true_labels = test_true_labels[sample_indices]
sample_predicted_labels = test_predictions[sample_indices]


# Define mapping from numerical labels to alphabets (excluding 'J' and 'Z')
label_to_alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 10: 'K', 11: 'L', 12: 'M',13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'}

# Define a function to display images with true and predicted labels
def show_images_with_labels(images, true_labels, predicted_labels):
    num_images = len(images)
    num_rows = (num_images + 1) // 2  # Calculate number of rows for subplots
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(num_rows, 2, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        true_label = label_to_alphabet[true_labels[i].item()]  # Map true label to alphabet
        predicted_label = label_to_alphabet[predicted_labels[i].item()]  # Map predicted label to alphabet
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()



show_images_with_labels(sample_images, sample_true_labels, sample_predicted_labels)



# COMMAND ----------

# MAGIC %md
# MAGIC Model Accuracy

# COMMAND ----------


# Calculate accuracy
accuracy = (test_predictions == test_true_labels).mean()

# Print accuracy
print(f"Accuracy on the test set: {accuracy:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC Saving the trained Model

# COMMAND ----------

# Specify the file path where you want to save the model

from datetime import datetime

# Get the current date
current_date = datetime.now()

# Format the date as YYYYMMDD
formatted_date = current_date.strftime("%Y%m%d")

model_name = 'CNN_pytorch'
model_path = 'abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/Models/'+model_name+'/'+'model.pth'

print(model_path)

# COMMAND ----------

import torch
import io
from pyspark.sql import SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

# Serialize the model to binary data
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
model_binary_data = buffer.getvalue()


# Create a DataFrame with a single column containing the binary data
schema = StructType([StructField("model_binary", BinaryType(), True)])
df_model = spark.createDataFrame([(model_binary_data,)], schema)


# COMMAND ----------

# MAGIC %md
# MAGIC Displaying the model as a dataframe.

# COMMAND ----------

display(df_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Writing the model into cloud container.

# COMMAND ----------

df_model.write.format('parquet').mode("overwrite").save('abfss://raw@azurelsuvjosephadls.dfs.core.windows.net/Models/CNN_pytorch/model_data')
