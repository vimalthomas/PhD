Detection of Sign Language Alphabets Using Regularized CNN Model 

This project contains five databaricks notebooks.
00_AI_Project_Data_Profiling - for data profiling
01_AI_Project_CNN_Regularized_Model - for model definition
02_AI_Project_CNN_Base_Model_Training_And_Testing - for model training and testing
03_AI_Project_Execution_Using_CNN_Regularization_Model - for data validation
04_Final_Sign_Language_Detector - for the final project execution

The data profiling notebook shows the MNIST data in the image format. 

![image](https://github.com/vimalthomas/PhD/assets/40187949/c50787d8-3314-4eaf-8ebc-3f9af6004028)

The final notebook 04_Final_Sign_Language_Detector has a databricks utils widget to accept the file path as the input file. 
![image](https://github.com/vimalthomas/PhD/assets/40187949/436c4a36-d7cf-4d3e-bc4b-65ac2fb78096)

The datasets, input and output are expected to be stored under azure container. One could simply run the model notebooks (01_AI_Project_CNN_Regularized_Model, 02_AI_Project_CNN_Base_Model_Training_And_Testing) in any environments. However, the following steps should have been completed.
1) Install the required libraries.
2) Change the source datapath from abfss containers to the local drive. the datasets are present in https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data
3) Finally change the model output path to local drive to use in the prediction notebook.

The final notebook provides the prediction results at the end.
![image](https://github.com/vimalthomas/PhD/assets/40187949/4988b161-bb3a-40cb-babd-338baa469597)

