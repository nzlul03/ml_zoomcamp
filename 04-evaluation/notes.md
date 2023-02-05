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
 
![image](https://user-images.githubusercontent.com/54148951/216826899-5c1d06cc-1852-47ba-b1b5-efa8162dddc4.png)
 
 **Recall**
 - For 46% of people, who are churning, we failed to identify them 
 
 **Accuracy**
 - Pretty high number
 - Our purpose : is to identify the churning users, so for this purpose, accuracy is not the best
 metric, because when we look at this, we think, okay. Model is doing pretty well but when we look at precision and when we look at recall, we see that we failed to identify 46% users and we actually sent a promotional mail to 33% of users who were not going to churn, but they probably will take advantage of our discount.
 
Now, we see that our model is not as good as we thought.

Accuracy can be misleading (especially when we have class imbalance cases). That's why it's useful to look at metrics like precision and recall.

![image](https://user-images.githubusercontent.com/54148951/216828055-29c0e558-3e11-4ee1-b375-7e93c3982615.png)

![image](https://user-images.githubusercontent.com/54148951/216828102-c51c6d15-09e6-4a59-a58e-1deafc2519a5.png)

 
## 4.5 ROC Curves

[Video](https://www.youtube.com/watch?v=dnBZLk53sQI&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR)

[Slides](https://www.slideshare.net/AlexeyGrigorev/ml-zoomcamp-4-evaluation-metrics-for-classification)

ROC stands for Receiver Operating Characteristic, and this idea was applied during the Second World War for evaluating the strength of radio detectors. This measure considers **False Positive Rate** (FPR) and **True Postive Rate** (TPR), which are derived from the values of the confusion matrix.

**FPR** is the fraction of false positives (FP) divided by the total number of negatives (FP and TN - the first row of confusion matrix), and we want to `minimize` it. The formula of FPR is the following: 

![image](https://user-images.githubusercontent.com/54148951/216828888-8f553701-c06d-4347-b3de-983aafd0380c.png)


In the other hand, **TPR** or **Recall** is the fraction of true positives (TP) divided by the total number of positives (FN and TP - second row of confusion table), and we want to `maximize` this metric. The formula of this measure is presented below: 

![image](https://user-images.githubusercontent.com/54148951/216828919-713a5d84-0dee-42b9-b33c-6e98d68ce364.png)


ROC curves consider Recall and FPR under all the possible thresholds. If the threshold is 0 or 1, the TPR and Recall scores are the opposite of the threshold (1 and 0 respectively), but they have different meanings, as we explained before. 

We need to compare the ROC curves against a point of reference to evaluate its performance, so the corresponding curves of random and ideal models are required. It is possible to plot the ROC curves with FPR and Recall scores vs thresholds, or FPR vs Recall. 


**Classes and methods:** 
* `np.repeat([x,y], [z,w])` - returns a numpy array with a z number of x values, and a w number of y values. 
* `roc_curve(x, y)` - sklearn.metrics class for calculating the false positive rates, true positive rates, and thresholds, given a target x dataset and a predicted y dataset.

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
