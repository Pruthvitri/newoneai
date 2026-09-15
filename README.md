# Student Marks Prediction using Linear Regression

## Overview

This project demonstrates a basic **Machine Learning regression problem** using Python and the Scikit-learn library.

The objective of this practical is to predict a student's marks based on the number of hours they study. A simple dataset containing **Study Hours** and **Marks** is created using Pandas. The data is then divided into training and testing sets, and a **Linear Regression** model is trained using the training data.

After training, the model is used to predict marks for the test data and for a new student. The performance of the model is evaluated using **Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score**.

Matplotlib is also used to visualize the relationship between study hours and marks and to display the regression line.

## Objective

* To understand the basic workflow of Machine Learning.
* To create a dataset using Pandas.
* To analyze and inspect the dataset.
* To separate features and target variables.
* To split the dataset into training and testing data.
* To train a Linear Regression model.
* To make predictions using the trained model.
* To evaluate model performance.
* To visualize the relationship between study hours and marks.
* To understand how Linear Regression can be used for prediction.

## Technologies and Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Dataset

The project uses a simple student performance dataset with two columns:

| Feature     | Description                            |
| ----------- | -------------------------------------- |
| Study_Hours | Number of hours studied by the student |
| Marks       | Marks obtained by the student          |

### Sample Data

| Study Hours | Marks |
| ----------: | ----: |
|           1 |    35 |
|           2 |    40 |
|           3 |    45 |
|           4 |    50 |
|           5 |    55 |
|           6 |    60 |
|           7 |    65 |
|           8 |    72 |
|           9 |    80 |
|          10 |    88 |

## Machine Learning Workflow

```text
Create Dataset
      ↓
Load Dataset using Pandas
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Select Features and Target
      ↓
Train-Test Split
      ↓
Create Linear Regression Model
      ↓
Train Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Visualize Results
      ↓
Predict Marks for New Student
```

## Feature and Target

In this project:

```text
Feature (X) → Study_Hours
Target (y)  → Marks
```

The model learns the relationship between study hours and marks.

## Model Used

### Linear Regression

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value.

The basic equation is:

<img width="800" height="500" alt="pr7image" src="https://github.com/user-attachments/assets/b36534c0-3499-47e9-a6cf-1b288b433b4b" />
