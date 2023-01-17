# 2. Machine Learning for Regression
## 2.1 Car price prediction project

[Video](https://www.youtube.com/watch?v=vM3SqPNlStE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=12)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-21-car-price-prediction-project)

This project is about the creation of a model for helping users to predict car prices. The dataset was obtained from [this kaggle competition](https://www.kaggle.com/CooperUnion/cardataset).

**Project plan:**
* Prepare data and Exploratory Data Analysis (EDA)
* Use linear regression for predicting price
* Understanding the internals of linear regression
* Evaluating the model with RMSE
* Feature engineering
* Regularization
* Using the model

![image](https://user-images.githubusercontent.com/54148951/212797889-a8c5b30d-5af8-4879-86cc-4afca7d1a44d.png)

## 2.2 Data preparation

[Video](https://www.youtube.com/watchv=Kd74oR4QWGM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=13)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

**Pandas attributes and methods**

* pd.read_csv(<file_path_string>) - read csv files 
* df.head() - take a look of the dataframe 
* df.columns - retrieve colum names of a dataframe 
* df.columns.str.lower() - lowercase all the letters 
* df.columns.str.replace(' ', '_') - replace the space separator 
* df.dtypes - retrieve data types of all features 
* df.index - retrieve indices of a dataframe

## 2.3 Exploratory data analysis

[Video](https://www.youtube.com/watchv=k6k8sQ0GhPM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=14)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

**Pandas attributes and methods:** 

* df[col].unique() - returns a list of unique values in the series 
* df[col].nunique() - returns the number of unique values in the series 
* df.isnull().sum() - returns the number of null values in the dataframe 

**Matplotlib and seaborn methods:**

* %matplotlib inline - assure that plots are displayed in notebook's cells
* sns.histplot() - show the histogram of a series

**Numpy methods:**
* np.log1p() - applies log transformation to a variable and adds one to each result 

Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible.


## 2.4 Setting up the validation framework

[Video](https://www.youtube.com/watch?v=ck0IfiPaQi0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=15)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.

![image](https://user-images.githubusercontent.com/54148951/212808675-0ce5ff27-b434-4ac7-ac86-ed938a29b57a.png)



**Pandas attributes and methods:** 
* `df.iloc[]` - returns subsets of records of a dataframe, being selected by numerical indices
* `df.reset_index()` - restate the original indices 
* `del df[col]` - eliminates target variable 

**Numpy methods:**
* `np.arange()` - returns an array of numbers 
* `np.random.shuffle()` - returns a shuffled array
* `np.random.seed()` - set a seed 

## 2.5 Linear regression
## 2.6 Linear regression: vector form
## 2.7 Training linear regression: Normal equation
## 2.8 Baseline model for car price prediction project
## 2.9 Root mean squared error
## 2.10 Using RMSE on validation data
## 2.11 Feature engineering
## 2.12 Categorical variables
## 2.13 Regularization
## 2.14 Tuning the model
## 2.15 Using the model
## 2.16 Car price prediction project summary

