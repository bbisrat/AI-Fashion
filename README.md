
----------------------------------------------------------------------------------------------------------------
SUMMARY
This program is designed to classify images from the Fashion MNIST dataset using TensorFlow. It includes integration with Google Cloud Logging for logging model evaluations.

----------------------------------------------------------------------------------------------------------------
INSTALLATION
- Before running the script, ensure you have the following installed:

    1. Python: The script is written in Python.

    2. TensorFlow: A core open-source library to help you develop and train ML models.         
        Install using pip: pip install tensorflow
    
    3. TensorFlow Datasets: An easy-to-use collection of datasets for TensorFlow. 
        Install using pip: pip install tensorflow-datasets

    4. NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices. 
        Install using pip:pip install numpy

    5. Google Cloud Logging: For cloud logging purposes. 
        1- First, set up a Google Cloud project. 
        2- Then, install the client library: pip install google-cloud-logging
----------------------------------------------------------------------------------------------------------------
USAGE
1- Setting Up Google Cloud Logging:
    - You need to set up authentication by creating a service account key in your Google Cloud project. 
    
    Export the path of your service account key as an environment variable: export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"

2- Running the Script:
    Run the script using Python: python fashion_mnist_classifier.py

    - The script will load the dataset, preprocess it, train a neural network model, evaluate its performance, and save the model in both the default TensorFlow format and HDF5 format.

----------------------------------------------------------------------------------------------------------------
PURPOSE
This script demonstrates:

- Loading and preprocessing data using TensorFlow and TensorFlow Datasets.

- Building and training a simple neural network for image classification.

- Utilizing Google Cloud Logging for monitoring model performance.

--------------------------------------------------------------------------------------

