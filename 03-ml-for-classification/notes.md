# 3. Machine Learning for Classification
## 3.1 Churn prediction project
[Video](https://www.youtube.com/watch?v=0Zw04wdeTQo&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

The project aims to identify customers that are likely to churn or stoping to use a service. Each
customer has a score associated with the probability of churning. Considering this data, the
company would send an email with discounts or other promotions to avoid churning. 

![image](https://user-images.githubusercontent.com/54148951/215039805-3682b6ab-02a0-40a4-a593-34e1a2cfae8a.png)


![image](https://user-images.githubusercontent.com/54148951/213963054-b7a5d296-1c0f-46db-bb94-cd565add39a4.png)

In the formula, yi is the model's prediction and belongs to {0,1}, being 0 the negative value or no churning, and 1 the positive value or churning. The output corresponds to the likelihood of
churning. 

![image](https://user-images.githubusercontent.com/54148951/213963219-1a596a25-4625-4443-bb25-a6203d9494e0.png)


In brief, the main idea behind this project is to build a model with historical data from customers and assign a score of the likelihood of churning. 


## 3.2 Data preparation
[Video](https://www.youtube.com/watch?v=VSGGU9gYvdg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

![image](https://user-images.githubusercontent.com/54148951/215046260-6b119e7b-f79f-4916-a570-831613db0a35.png)


**Commands, functions, and methods:** 

* `!wget` - Linux shell command for downloading data 
* `pd.read.csv()` - read csv files 
* `df.head()` - take a look of the dataframe 
* `df.head().T` - take a look of the transposed dataframe 
* `df.columns` - retrieve column names of a dataframe 
* `df.columns.str.lower()` - lowercase all the letters 
* `df.columns.str.replace(' ', '_')` - replace the space separator 
* `df.dtypes` - retrieve data types of all series 
* `df.index` - retrive indices of a dataframe
* `pd.to_numeric()` - convert a series values to numerical values. The `errors=coerce` argument allows making the transformation despite some encountered errors. 
* `df.fillna()` - replace NAs with some value 
* `(df.x == "yes").astype(int)` - convert x series of yes-no values to numerical values. 

## 3.3 Setting up the validation framework
## 3.4 EDA
## 3.5 Feature importance: Churn rate and risk ratio
## 3.6 Feature importance: Mutual information
## 3.7 Feature importance: Correlation
## 3.8 One-hot encoding
## 3.9 Logistic regression 
## 3.10 Training logistic regression with Scikit-Learn
## 3.11 Model interpretation
## 3.12 Using the model
