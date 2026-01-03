Overview

This project implements a machine learning–based system to predict the risk of heart disease using structured patient data. The objective is to demonstrate the application of data preprocessing, model training, and deployment in a medical data context.

The system uses a trained classification model and provides predictions through a simple web interface.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Methodology

A publicly available heart disease dataset is used as the data source.

The dataset is preprocessed and split into training and testing sets.

A Decision Tree classifier is trained on the data.

The trained model is saved and reused for inference.

A Streamlit application is used to collect inputs and display prediction results.

Model Information

Model type: Decision Tree Classifier

Output: Probability-based prediction of heart disease risk

The model is designed to demonstrate the machine learning workflow rather than to provide clinically accurate diagnoses.

How to Run
pip install -r requirements.txt
python train.py
streamlit run app.py

Disclaimer

This project is intended strictly for educational and demonstration purposes.
It is not a medical diagnostic tool and should not be used for clinical decision-making.

Learning Outcomes

Working with medical datasets

Implementing a supervised learning model

Saving and loading trained models

Deploying machine learning models using a web interface