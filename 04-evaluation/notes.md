# 4. Evaluation Metrics for Classification
## 4.1 Evaluation Metrics: Session Overview

[Video](https://www.youtube.com/watch?v=gmg5jw1bM8A&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

The fourth week of Machine Learning Zoomcamp is about different metrics to evaluate a binary classifier. These measures include accuracy, confusion table, precision, recall, ROC curves(TPR, FRP, random model, and ideal model), AUROC, and cross-validation. 

## 4.2 Accuracy and Dummy Model

[Video](https://www.youtube.com/watch?v=FW_l7lB0HUI&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

**Accuracy** measures the fraction of correct predictions. Specifically, it is the number of correct predictions divided by the total number of predictions. 

![image](https://user-images.githubusercontent.com/54148951/215947641-cc9acfd0-6bba-4c4a-a558-8bf20c1bc481.png)


We can change the **decision threshold**, it should not be always 0.5. But, in this particular problem, the best decision cutoff, associated with the hightest accuracy (80%), was indeed 0.5. 

Note that if we build a **dummy model** in which the decision cutoff is 1, so the algorithm predicts that no clients will churn, the accuracy would be 73%. Thus, we can see that the improvement of the original model with respect to the dummy model is not as high as we would expect. 

Therefore, in this problem accuracy can not tell us how good is the model because the dataset is **unbalanced**, which means that there are more instances from one category than the other. This is also known as **class imbalance**. 

**Classes and methods:** 

* `np.linspace(x,y,z)` - returns a numpy array starting at x until y with a z step 
* `Counter(x)` - collection class that counts the number of instances that satisfy the x condition
* `accuracy_score(x, y)` - sklearn.metrics class for calculating the accuracy of a model, given a predicted x dataset and a target y dataset.

## 4.3 Confusion Table

[Video](https://www.youtube.com/watch?v=Jt2dDLSlBng&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

Confusion table is a way of measuring different types of errors and correct decisions that binary classifiers can make. Considering this information, it is possible to evaluate the quality of the model by different strategies.

![image](https://user-images.githubusercontent.com/54148951/216754909-1e3bd1a7-5cef-4ea8-967b-84c631d45419.png)

When comes to a prediction of an LR model, each falls into one of four different categories:

* Prediction is that the customer WILL churn. This is known as the **Positive class**
    * And Customer actually churned - Known as a **True Positive (TP)**
    * But Customer actually did not churn - Knwon as a **False Positive (FP)**
* Prediction is that the customer WILL NOT churn' - This is known as the **Negative class**
    * Customer did not churn - **True Negative (TN)**
    * Customer churned - **False Negative (FN)**
    
'Confusion Table' is a way to summarize the above results in a tabular format, as shown below: 

![image](https://user-images.githubusercontent.com/54148951/216755353-470e99f0-a410-4221-93b5-3e7c7cf576ef.png)

|**Actual :arrow_down:     Predictions:arrow_right:**|**Negative**|**Positive**|
|:-:|---|---|
|**Negative**|TN|FP|
|**Postive**|FN|TP| 

The **accuracy** corresponds to the sum of TN and TP divided by the total of observations. 

![image](https://user-images.githubusercontent.com/54148951/216756590-02e68f90-4908-4a98-8488-7484a0a1f67e.png)


## 4.4 Precision and Recall

[Video](https://www.youtube.com/watch?v=gRLP_mlglMM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

**Precision** tell us the fraction of positive predictions that are correct. It takes into account only the **positive class** (TP and FP - second column of the confusion matrix), as is stated in the following formula:


$$P = \cfrac{TP}{TP + FP}$$

![image](https://user-images.githubusercontent.com/54148951/216757526-c15f52ad-1d8f-4afa-b57d-29b3b137d02b.png)

![image](https://user-images.githubusercontent.com/54148951/216826376-0461a4e8-2151-4642-87ae-9dd353967317.png)

**Recall** measures the fraction of correctly identified postive instances. It considers parts of the **postive and negative classes** (TP and FN - second row of confusion table). The formula of this metric is presented below: 


$$R = \cfrac{TP}{TP + FN}$$

![image](https://user-images.githubusercontent.com/54148951/216826656-e8983dab-d513-42e2-9e4c-c3331fd4297c.png)

![image](https://user-images.githubusercontent.com/54148951/216826737-8cd87282-734b-4ac7-8a82-9763912deb63.png)

 In this problem, the precision and recall values were 67% and 54% respectively. So, these measures reflect some errors of our model that accuracy did not notice due to the **class imbalance**. 
 
![image](https://user-images.githubusercontent.com/54148951/216826845-d6871a97-299a-4d8e-a7eb-5095436e9b56.png)
 
 
## 4.5 ROC Curves

[Video]()

[Slides]()

## 4.6 ROC AUC

[Video]()

[Slides]()

## 4.7 Cross-Validation

[Video]()

[Slides]()

## 4.8 Summary

[Video](https://www.youtube.com/watch?v=-v8XEQ2AHvQ&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

### General definitions: 

* **Metric:** A single number that describes the performance of a model
* **Accuracy:** Fraction of correct answers; sometimes misleading 
* Precision and recall are less misleading when we have class inbalance
* **ROC Curve:** A way to evaluate the performance at all thresholds; okay to use with imbalance
* **K-Fold CV:** More reliable estimate for performance (mean + std)

In brief, this weeks was about different metrics to evaluate a binary classifier. These measures included accuracy, confusion table, precision, recall, ROC curves(TPR, FRP, random model, and ideal model), and AUROC. Also, we talked about a different way to estimate the performance of the model and make the parameter tuning with cross-validation. 
