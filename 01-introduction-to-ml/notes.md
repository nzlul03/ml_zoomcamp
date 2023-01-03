# 1. Introduction to Machine Learning
## - 1.1 Introduction to Machine Learning

[Video](https://www.youtube.com/watch?v=Crm_5n4mvmg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=2)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-11-introduction-to-machine-learning)

The concept of ML is depicted with an example of predicting the price of a car. The ML model
learns from data, represented as some **features** such as year, mileage, among others, and the **target** variable, in this case, the car's price, by extracting patterns from the data.

Then, the model is given new data (**without** the target) about cars and predicts their price (target).

ML is a process of **extracting patterns from data**, which is of two types:
* features (information about the object)
* target (property to predict for unseen objects)

Therefore, new feature values are presented to the model, and it makes **predictions** from the learned patterns.


## - 1.2 ML vs Rule-Based Systems
The difference between ML and Rule-Based systems is explained with the example of a spam filter.

Traditional Rule-Based systems are based on a set of characteristics (keywords, email length, etc.) that identify an email as spam or not. As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

###1. Get Data
Emails from the user's spam folder and inbox gives examples of spam and non-spam.

###2. Define and calculate features
Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).
Each email can be encoded (converted) to the values of it's features and target.

### 3. Train and use the model
A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.


## - 1.3 Supervised Machine Learning
## - 1.4 CRISP-DM
## - 1.5 Model Selection Process
## - 1.6 Setting up the Environment
## - 1.7 Introduction to NumPy
## - 1.8 Linear Algebra Refresher
## - 1.9 Introduction to Pandas
