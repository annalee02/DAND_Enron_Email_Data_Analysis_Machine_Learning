# Identify Fraud from Enron Email
#### By Anna Lee

**1. Summarize the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?**

### Project Goal
The project goal is to build a model to predict who were involved in the Enron scandal by using Enron's email and financial dataset.
### Data Overview
Enron was one of the largest companies in United States in 2000, however it had collapsed into bankruptcy due to their corporate fraud by 2002. Tens of thousands of emails and detailed financial data for top executives entered into the public record after Federal investigation. Person of interest (POI) identifiers have been created based on this open data in order to answer project questions.
### Why Machine Learning?
Implementing designed machine learning algorithms can effectively reduce the hard work of making predictions from data with reasonable accuracy. Accuracy of each prediction model is measurable, so we can compare several models' performance in the algorithm selection process.
### Data Exploration
The code in `explore_enron_data.py` conducts data exploration and provides information as follows.
* Number of employees: 146
* Number of POIs in the dataset: 18 out of 35 total POIs
* Number of features in total: 21 features
* Missing values:

|Column Name               | Number of NaN values |
|-------------------------:|---------------------:|
|bonus                     |63                    |
|deferral_payments         |106                   |
|director_fees             |128                   |
|deferred_income           |96                    |
|exercised_stock_options   |43                    |
|email_address             |33                    |
|expenses                  |50                    |
|from_messages             |58                    |
|from_poi_to_this_person   |58                    |
|from_this_person_to_poi   |58                    |
|loan_advances             |141                   |
|long_term_incentive       |79                    |
|other                     |53                    |
|poi                       |0                     |
|poi_ratio                 |58                    |
|restricted_stock          |35                    |
|restricted_stock_deferred |127                   |
|salary                    |50                    |
|shared_receipt_with_poi   |58                    |
|to_messages               |58                    |
|total_payments            |21                    |
|total_stock_value         |19                    |

### Outliers
From `enron61702insiderpay.pdf`, I could tell that 'Total' is sum of values as an outlier and 'LOCKHART, EUGENE E' does not have any values. Their key-value pair from the dictionary have been cleaned away using pop() before moving on.

**2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.**

### SelectKBest
**SelectKBest** function returns an array of all tuples except 'poi' (boolean) and 'email_address' (string) in descending order of its score. To see overall, I set k='all'. The output helps decide which features should be chosen in the final feature selection process later.\
[('exercised_stock_options', 25.097541528735491),
('total_stock_value', 24.467654047526398),
('bonus', 21.060001707536571),
('salary', 18.575703268041785),
('deferred_income', 11.595547659730601),
('long_term_incentive', 10.072454529369441),
('restricted_stock', 9.3467007910514877),
('total_payments', 8.8667215371077752),
('shared_receipt_with_poi', 8.7464855321290802),
('loan_advances', 7.2427303965360181),
('expenses', 6.2342011405067401),
('from_poi_to_this_person', 5.3449415231473374),
('other', 4.204970858301416),
('from_this_person_to_poi', 2.4265081272428781),
('director_fees', 2.1076559432760908),
('to_messages', 1.6988243485808501),
('deferral_payments', 0.2170589303395084),
('from_messages', 0.16416449823428736),
('restricted_stock_deferred', 0.06498431172371151)]
### Feature Creation
I decided to find a ratio of how frequently each person communicated with POIs by email, because POIs are more likely to interact with each other often. 'poi_email_ratio' has been newly created and added to the feature list.
### Final Feature Selection
* In the final feature_list, half of features are excluded because of their low scores shown on the output of SelectKBest algorithm and a new feature was added. \
['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'loan_advances', 'expenses', 'shared_receipt_with_poi', 'poi_email_ratio']

* The new feature, 'poi_email_ratio' is in Top 5 in terms of SelectKBest score. \
[('exercised_stock_options', 25.097541528735491), ('total_stock_value', 24.467654047526398), ('bonus', 21.060001707536571), ('salary', 18.575703268041785), ('poi_email_ratio', 16.23649190163686)]

### Feature Scaling
Normalization seems important before testing of the algorithm as there are many different units such as million dollars and count of emails in the dataset. After feature scaling, larger units would not influence on the classifier when it uses distance measurement.

**3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?**

### Algorithm Selection
Many supervised machine learning algorithms were tested and 2 algorithms (__Naive Bayes__, __Decision Tree__) ran relatively faster than others. __AdaBoost__ performed adequately using default parameter values. RandomForest had the highest accuracy score but Recall score was lower than .3, whereas GaussianNB and AdaBoost provided a bit lower accuracy score but Precision and Recall scores are over .3.

**4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).**

### Parameter Tuning Definition
Tuning parameters means changing the default setting of a chosen algorithm to optimize its performance and then produce the best results. If you do not tune it properly, it might produce false data. For example, the algorithm may show its accuracy level is very high but with very low precision and recall scores.
### Tune the Algorithm
To tune chosen algorithms, I used __GridSearchCV__ to find best parameter values. I picked Naive Bayes, Decision Tree, and Adaboost to tune the algorithm because they are close to .3 precision and recall scores or higher than that. GaussianNB does not have many parameters that I needed to tune so I only checked the number of features I should use. The best performing algorithm out of 3, was __GaussianNB__ with 5 features. \
Its result is:

|Accuracy      |Precision    | Recall      |
|-------------:|------------:|------------:|
|0.85000       |0.42138      |0.33500      |


**5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?**

### Validation Definition
Validation is a process of splitting a dataset into training and test data in order to estimate the performance of an algorithm before the model is applied to new data in practice. It is important to create a prediction model using a training dataset and then test on the remainder because if we train and test it on the same data, its accuracy score would be over 99 percent. The overfitted model is most likely to perform poorly on new datasets.
### Validation Strategy
Stratified Shuffle Split cross validator preserves the percentage of POI labels in a random split, which reduce risks from imbalance in the POI/non-POI ratio between training and test datasets.  

**6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.**

### Evaluation Metrics
* Final algorithm, Naive Bayes evaluation metrics:

|Precision    | Recall      |
|------------:|------------:|
|0.42138      |0.33500      |

Using accuracy score is not enough to gauge algorithm performance due to a large imbalance in data labels (POIs/non-POIs). Precision and recall scores show whether the final algorithm correctly predicted POIs or not. Precision score means how many of them truly belong to the positive class and Recall score means how many of them were correctly classified as positive.

* true-positive: algorithm predicts it is a POI who is a POI.
* true-negative: algorithm predicts it is not a POI who is actually a POI.
* false-positive: algorithm predicts it is a POI but he/she is actually a non-POI.
* false-negative: algorithm predicts it is not a POI who is a non-POI.

Precision = tp / (tp+fp) \
Recall = tp / (tp+fn)
