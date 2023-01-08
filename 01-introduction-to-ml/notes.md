# 1. Introduction to Machine Learning
## 1.1 Introduction to Machine Learning

[Video](https://www.youtube.com/watch?v=Crm_5n4mvmg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=2)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-11-introduction-to-machine-learning)

The concept of ML is depicted with an example of predicting the price of a car. The ML model
learns from data, represented as some **features** such as year, mileage, among others, and the **target** variable, in this case, the car's price, by extracting patterns from the data.

Then, the model is given new data (**without** the target) about cars and predicts their price (target).

![image](https://user-images.githubusercontent.com/54148951/211178494-83bf407e-79d5-4972-8f6a-e6b6ef2a47a0.png)

ML is a process of **extracting patterns from data**, which is of two types:
* features (information about the object)
* target (property to predict for unseen objects)

Therefore, new feature values are presented to the model, and it makes **predictions** from the learned patterns.
Predictions:
![image](https://user-images.githubusercontent.com/54148951/211178519-553f1b36-c63c-4153-b221-d72efe570c9b.png)



## 1.2 ML vs Rule-Based Systems

[Video](https://www.youtube.com/watchv=CeukwyUdaz8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=3)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-12-ml-vs-rulebased-systems)

The difference between ML and Rule-Based systems is explained with the example of a spam filter.

Traditional Rule-Based systems are based on a set of characteristics (keywords, email length, etc.) that identify an email as spam or not.\
Rule Based System
![image](https://user-images.githubusercontent.com/54148951/211178861-f2bb8293-4904-4254-bf8a-d37e82132627.png)

As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

### 1. Get Data
Emails from the user's spam folder and inbox gives examples of spam and non-spam.

### 2. Define and calculate features
Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).
Each email can be encoded (converted) to the values of it's features and target.

### 3. Train and use the model
A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.

Machine Learning\
![image](https://user-images.githubusercontent.com/54148951/211178911-8722914b-9226-49c9-a983-724971aae481.png)

## 1.3 Supervised Machine Learning
[Video](https://www.youtube.com/watchv=j9kcEuGcC2Y&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=4)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-13-supervised-machine-learning)

In Supervised Machine Learning (SML) there are always labels associated with certain features. The model is trained, and then it can make predictions on new features. In this way, the model is taught by certain features and targets.
* **Feature matrix (X)**: made of observations or objects (rows) and features (columns).
* **Target variable (y)**: a vector with the target information we want to predict. For each row of X there's a value in y.
The model can be represented as a function **g** that takes the X matrix as a parameter and tries to predict values as close as possible to y targets. The obtention of the g function is what it is called **training**.

![image](https://user-images.githubusercontent.com/54148951/211186453-86fbd02f-9f06-4ac7-a9ce-b051615b24fc.png)



### Types of Supervised Machine Learning
* **Regression:** the output is a number (car's price)
* **Classification:** the output is a category (spam example).\
        * **Binary:** there are two categories.\
        * **Multiclass problems:** there are more than two categories.\
        * **Multilabel problems:** one data can be categorized by more than one labels.
* **Ranking:** the output is the big scores associated with certain items. It is applied in recommender systems.

Supervised machine learning is about teaching the model by showing different examples, and the goal is to come up with a function that takes the feature matrix as a parameter and makes predictions as close as possible to the y targets.


## 1.4 CRISP-DM
[Video](https://www.youtube.com/watchv=dCa3JvmJbr0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=5)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-14-crispdm)

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an open standard process model that describes common approaches used by data mining experts. It is the most widely-used analytics model. The project was led by five companies: Integral Solutions Ltd (ISL), Teradata, Daimler AG, NCR Corporation and OHRA, an insurance company.

1. **Business understanding:** An important question is if do we need ML for the project. The goal of the project has to be measurable. 
2. **Data understanding:** Analyze available data sources, and decide if more data is required. 
3. **Data preparation:** Clean data and remove noise applying pipelines, and the data should be converted to a tabular format, so we can put it into ML.
4. **Modeling:** training Different models and choose the best one. Considering the results of this step, it is proper to decide if is required to add new features or fix data issues. 
5. **Evaluation:** Measure how well the model is performing and if it solves the business problem. 
6. **Deployment:** Roll out to production to all the users. The evaluation and deployment often happen together - **online evaluation**. 

It is important to consider how well maintainable the project is.

## - 1.5 Model Selection Process
## - 1.6 Setting up the Environment
## - 1.7 Introduction to NumPy
## - 1.8 Linear Algebra Refresher
## - 1.9 Introduction to Pandas
