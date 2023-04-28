# Diabetes Prediction with SVM

This project is a machine learning project that aims to predict whether or not a patient has diabetes based on certain features such as age, BMI, and blood pressure using the Pima Indians Diabetes Dataset. The goal of the project is to build a machine-learning model that can accurately predict whether a patient has diabetes or not based on these features.

## Requirements

To run this project, you will need the following software:

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Getting Started

To get started, download the Pima Indians Diabetes Dataset from [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database). Alternatively, you can clone this repository to your local machine by running the following command:

git clone https://github.com/yourusername/diabetes.git


Once you have the dataset and the repository cloned, you can run the `diabetes.ipynb` script to train the machine learning model and make predictions on new data. The script will load the dataset into a Pandas data frame, preprocess the data by scaling the features, and then split the data into training and testing sets. It will then train a support vector machine (SVM) model on the training data and use it to make predictions on the testing data.

## Model Evaluation

The performance of the machine learning model is evaluated using two metrics - accuracy and F1 score.

- **Accuracy:** Accuracy measures the proportion of correctly classified instances out of the total number of instances. A higher accuracy indicates better performance of the model.

- **F1 score:** F1 score is the harmonic mean of precision and recall. It takes into account both false positives and false negatives. A high F1 score indicates better performance of the model.

In this project, we achieved an accuracy of 0.79 and an F1 score of 0.63, which are decent results. However, it's important to remember that these metrics do not tell the whole story and it's necessary to consider the overall context of the problem and the model's limitations.

## Conclusion

In this project, we used the Pima Indians Diabetes Dataset to build a machine-learning model that can predict whether a patient has diabetes or not based on different features. We achieved decent results with the SVM model, but it's important to remember that there are other factors to consider when evaluating a model's performance. This project can be extended by trying different machine-learning algorithms and feature engineering techniques to improve the performance of the model.


