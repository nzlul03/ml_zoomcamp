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
[Video](https://www.youtube.com/watch?v=VSGGU9gYvdg&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

**Classes, functions, and methods:** 

* `train_test_split` - Scikit-Learn class for splitting datasets. Linux shell command for downloading data. The `random_state` argument set a random seed for reproducibility purposes.  
* `df.reset_index(drop=True)` - reset the indices of a dataframe and delete the previous ones. 
* `df.x.values` - extract the values from x series
* `del df['x']` - delete x series from a dataframe 

## 3.4 EDA
[Video](https://www.youtube.com/watch?v=BNF1wjBwTQA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

The EDA for this project consisted of: 
* Checking missing values 
* Looking at the distribution of the target variable (churn)
* Looking at numerical and categorical variables 

**Churn Rate**

![image](https://user-images.githubusercontent.com/54148951/215251578-0776db71-2248-413a-8e9e-e2537e3786b8.png)


**Functions and methods:** 

* `df.isnull().sum()` - retunrs the number of null values in the dataframe.  
* `df.x.value_counts()` returns the number of values for each category in x series. The `normalize=True` argument retrieves the percentage of each category. In this project, the mean of churn is equal to the churn rate obtained with the value_counts method. 
* `round(x, y)` - round an x number with y decimal places
* `df[x].nunique()` - returns the number of unique values in x series 

## 3.5 Feature importance: Churn rate and risk ratio
[Video](https://www.youtube.com/watch?v=fzdzPLlvs40&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

1. **Churn rate:** Difference between mean of the target variable and mean of categories for a feature. If this difference is greater than 0, it means that the category is less likely to churn, and if the difference is lower than 0, the group is more likely to churn. The larger differences are indicators that a variable is more important than others. 

![image](https://user-images.githubusercontent.com/54148951/215252539-4a96df78-ef2e-46c0-8a92-b1bd8a64e359.png)

2. **Risk ratio:** Ratio between mean of categories for a feature and mean of the target variable. If this ratio is greater than 1, the category is more likely to churn, and if the ratio is lower than 1, the category is less likely to churn. It expresses the feature importance in relative terms. 

![image](https://user-images.githubusercontent.com/54148951/215252752-5782280d-a98b-47d7-af26-354da14bca85.png)


**Functions and methods:** 

* `df.groupby('x').y.agg([mean()])` - returns a dataframe with mean of y series grouped by x series 

![image](https://user-images.githubusercontent.com/54148951/215252877-073df216-3f4e-4aba-b729-30af94ed1dd1.png)

## 3.6 Feature importance: Mutual information
[Video](https://www.youtube.com/watch?v=_u2YaGT6RN0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-3-machine-learning-for-classification)

Mutual information is a concept from information theory, which measures how much we can learn about one variable if we know the value of another. In this project, we can think of this as how much do we learn about churn if we have the information from a particular feature. So, it is a measure of the importance of a categorical variable. 

**Classes, functions, and methods:** 

* `mutual_info_score(x, y)` - Scikit-Learn class for calculating the mutual information between the x target variable and y feature. 
* `df[x].apply(y)` - apply a y function to the x series of the df dataframe. 
* ` df.sort_values(ascending=False).to_frame(name='x')` - sort values in an ascending order and called the column as x. 

## 3.7 Feature importance: Correlation
## 3.8 One-hot encoding
## 3.9 Logistic regression 
## 3.10 Training logistic regression with Scikit-Learn
## 3.11 Model interpretation
## 3.12 Using the model
