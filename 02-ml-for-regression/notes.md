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

* `pd.read_csv(<file_path_string>)` - read csv files 
* `df.head()` - take a look of the dataframe 
* `df.columns` - retrieve colum names of a dataframe 
* `df.columns.str.lower()` - lowercase all the letters 
* `df.columns.str.replace(' ', '_')` - replace the space separator 
* `df.dtypes` - retrieve data types of all features 
* `df.index` - retrieve indices of a dataframe

## 2.3 Exploratory data analysis

[Video](https://www.youtube.com/watchv=k6k8sQ0GhPM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=14)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

**Pandas attributes and methods:** 

* `df[col].unique()` - returns a list of unique values in the series 
* `df[col].nunique()` - returns the number of unique values in the series 
* `df.isnull().sum()` - returns the number of null values in the dataframe 

**Matplotlib and seaborn methods:**

* `%matplotlib inline` - assure that plots are displayed in notebook's cells
* `sns.histplot()` - show the histogram of a series

**Numpy methods:**
* `np.log1p()` - applies log transformation to a variable and adds one to each result 

Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible.


## 2.4 Setting up the validation framework

[Video](https://www.youtube.com/watch?v=ck0IfiPaQi0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=15)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

In general, the dataset is split into three parts: training, validation, and test. For each partition, we need to obtain feature matrices (X) and y vectors of targets. First, the size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset, and the partitions are created with the shuffled indices.

![image](https://user-images.githubusercontent.com/54148951/212808851-a17714d3-275f-445c-8c4a-f122287e0247.png)


**Pandas attributes and methods:** 
* `df.iloc[]` - returns subsets of records of a dataframe, being selected by numerical indices
* `df.reset_index()` - restate the original indices 
* `del df[col]` - eliminates target variable 

**Numpy methods:**
* `np.arange()` - returns an array of numbers 
* `np.random.shuffle()` - returns a shuffled array
* `np.random.seed()` - set a seed 

## 2.5 Linear regression

[Video](https://www.youtube.com/watchv=Dn1eTQLsOdA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=16)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Model for solving regression tasks, in which the objective is to adjust a line for the data and make predictions on new values. The input of this model is the **feature matrix** `X` and a `y` **vector of predictions** is obtained, trying to be as close as possible to the **actual** `y` values. The linear regression formula is the sum of the bias term \( $w_0$ \), which refers to the predictions if there is no information, and each of the feature values times their corresponding weights as \( $x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$ \).

![image](https://user-images.githubusercontent.com/54148951/212811218-0ff29b20-bc39-45ef-9bcf-4c53ef9b3ee2.png)

So the simple linear regression formula looks like:

$g(x_i) = w_0 + x_{i1} \cdot w_1 + x_{i2} \cdot w_2 + ... + x_{in} \cdot w_n$.

And that can be further simplified as:

$g(x_i) = w_0 + \displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$

Here is a simple implementation of Linear Regression in python:

~~~~python
w0 = 7.1
def linear_regression(xi):
    
    n = len(xi)
    
    pred = w0
    w = [0.01, 0.04, 0.002]
    for j in range(n):
        pred = pred + w[j] * xi[j]
    return pred
~~~~
        

If we look at the $\displaystyle\sum_{j=1}^{n} w_j \cdot x_{ij}$ part in the above equation, we know that this is nothing else but a vector-vector multiplication. Hence, we can rewrite the equation as $g(x_i) = w_0 + x_i^T \cdot w$

We need to assure that the result is shown on the untransformed scale by using the inverse function `exp()`. 

## 2.6 Linear regression: vector form

[Video](https://www.youtube.com/watchv=YkyevnYyAww&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=17)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

The formula of linear regression can be synthesized with the dot product between features and weights. The feature vector includes the *bias* term with an *x* value of one, such as $w_{0}^{x_{i0}},\ where\ x_{i0} = 1\ for\ w_0$.

When all the records are included, the linear regression can be calculated with the dot product between ***feature matrix*** and ***vector of weights***, obtaining the `y` vector of predictions. 

## 2.7 Training linear regression: Normal equation
[Video](https://www.youtube.com/watch?v=hx6nakY11g&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=18)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Obtaining predictions as close as possible to $y$ target values requires the calculation of weights from the general Linear Regression equation.
The feature matrix does not 
have an inverse because it is not square, so it is required to obtain an approximate solution, which can be obtained using the **Gram matrix** (multiplication of feature matrix ($X$) and its transpose ($X^T$)). The vector of weights or coefficients $w$ obtained with this formula is the closest possible solution to the LR system.

Normal Equation:

$w$ = $(X^TX)^{-1}X^Ty$

Where:

$X^TX$ is the Gram Matrix

General implementation:
~~~~python
 def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
~~~~

## 2.8 Baseline model for car price prediction project
[Video](https://www.youtube.com/watch?v=SvPpMMYtYbU&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=19)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

* In this lesson we build a baseline model and apply the `df_train` dataset to derive weights for the bias (w0) and the features (w). For this, we use the `train_linear_regression(X, y)` function from the previous lesson.
* Linear regression only applies to numerical features. Therefore, only the numerical features from `df_train` are used for the feature matrix. 
* We notice some of the features in `df_train` are `nan`. We set them to `0` for the sake of simplicity, so the model is solvable, but it will be appropriate if a non-zeo value is used as the filler (e.g. mean value of the feature).
* Once the weights are calculated, then we apply them on  $$\\\\ \large g(X) = w_0 + X \cdot w$$ to derive the predicted y vector.
* Then we plot both predicted y and the actual y on the same histogram for a visual comparison.


## 2.9 Root mean squared error (RMSE)

[Video](https://www.youtube.com/watch?v=0LWoFtbzNUM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=20)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

* In the previous lesson we found out our predictions were a bit off from the actual target values in the training dataset. We need a way to quantify how good or bad the model is. This is where RMSE can be of help.
* Root Mean Squared Error (RMSE) is a way to evaluate regression models. It measures the error associated with the model being evaluated. This numerical figure then can be used to compare the models, enabling us to choose the one that gives the best predictions.

$$RMSE = \sqrt{ \frac{1}{m} \sum {(g(x_i) - y_i)^2}}$$

- $g(x_i)$ is the prediction
- $y_i$ is the actual
- $m$ is the number of observations in the dataset (i.e. cars)

~~~~python
def rmse(y, y_pred):
  se = (y - y_pred) ** 2
  mse = se.mean()
  return np.sqrt(mse)
~~~~
## 2.10 Using RMSE on validation data
[Video](https://www.youtube.com/watch?v=rawGPXg2ofE&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=21)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Calculation of the RMSE on validation partition of the dataset of car price prediction. In this way, we have a metric to evaluate the model's  performance.

## 2.11 Feature engineering
[Video](https://www.youtube.com/watch?v=-aEShw4ftB0&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=22)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Feature engineering is the process of creating new features
The feature age of the car was included in the dataset, obtained with the subtraction of the maximum year of cars and each of the years of cars. 
This new feature improved the model performance, measured with the RMSE and comparing the distributions of y target variable and predictions. 

## 2.12 Categorical variables
[Video](https://www.youtube.com/watch?v=sGLAToAAMa4&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=23)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Categorical variables are typically strings, and pandas identifies them as object types. These variables need to be converted to a numerical form because ML models can interpret only numerical features. It is possible to incorporate certain categories from a feature, not necessarily all of them. 
This transformation from categorical to numerical variables is known as One-Hot encoding.

## 2.13 Regularization
[Video](https://www.youtube.com/watch?v=91ve3EJlHBc&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=24)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

If the feature matrix has duplicate columns (or columns that can be expressed as a linear combination of other columns), it will not have an inverse matrix. But, sometimes this error
could be passed if certain values are slightly different between duplicated columns.

So, if we apply the normal equation with this feature matrix, the values associated with duplicated columns are very large, which decreases the model performance. 

To solve this issue, one alternative is adding a small number to the diagonal of the feature matrix, which corresponds to regularization. 

![image](https://user-images.githubusercontent.com/54148951/213959410-a3a4f496-cce8-41e4-adca-a213cc75990c.png)


This technique  works because the addition of small values to the diagonal makes it less likely to have duplicated columns. The regularization value is a hyperparameter of the model. After applying  regularization the model performance improved. 

## 2.14 Tuning the model
[Video](https://www.youtube.com/watch?v=lW-YVxPgzQw&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=25)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-2-slides)

Tuning the model consisted of finding the best regularization hyperparameter value, using the validation partition of the dataset. The model was then trained with this regularization value.

## 2.15 Using the model
## 2.16 Car price prediction project summary

