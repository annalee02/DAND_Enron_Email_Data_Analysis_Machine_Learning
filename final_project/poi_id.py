#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

##############################################
### Task 1: Select what features I'll use. ###
##############################################
# poi = a Person of Interest
# Starter features
# 'email_address' is not included because it causes error due to its string values.
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred',
                'deferred_income', 'total_stock_value', 'expenses',
                'exercised_stock_options', 'other', 'long_term_incentive',
                'restricted_stock', 'director_fees', 'to_messages',
                'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Get the number of NaN values in the column
df = pd.DataFrame.from_records(list(data_dict.values()))
df.replace(to_replace='NaN', value=np.nan, inplace=True)
print "Number of NaN values \n", df.isnull().sum()

###############################
### Task 2: Remove outliers ###
###############################

data_dict.pop("TOTAL", 0) # Sum of values
data_dict.pop("LOCKHART EUGENE E", 0) # It contains no values

#####################################
### Task 3: Create new feature(s) ###
#####################################

# Provides a score of how useful each feature is
def Select_K_Best(data_dict, features_list, k):
    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    skbest = SelectKBest(k='all') # to check all features' score
    skbest.fit(features, labels)
    scores = skbest.scores_
    tuples = zip(features_list[1:], scores)
    skbest_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    return skbest_features[:k] # an array of tuples

print 'All features: \n', Select_K_Best(data_dict, features_list, 21)

# Create new email feature and add to the dataframe and features_list
def create_new_feature():
    # 'poi_email_ratio'
    features = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi']

    for key in data_dict:
        person = data_dict[key]
        is_valid = True
        for feature in features:
            if person[feature] == 'NaN':
                is_valid = False
        if is_valid:
            total_from_poi = person['from_poi_to_this_person'] + person['from_messages']
            total_to_poi = person['from_this_person_to_poi'] + person['to_messages']
            to_poi_ratio = float(person['from_this_person_to_poi']) / total_to_poi
            from_poi_ratio = float(person['from_poi_to_this_person']) / total_from_poi
            person['poi_email_ratio'] = to_poi_ratio + from_poi_ratio
        else:
            person['poi_email_ratio'] = 'NaN'

    # Add new feature to features_list
    features_list.extend(['poi_email_ratio'])

# Exclude low score features
features_list = ['poi','salary', 'total_payments', 'bonus', 'deferred_income',
                'total_stock_value', 'exercised_stock_options', 'long_term_incentive',
                'restricted_stock', 'loan_advances', 'expenses', 'shared_receipt_with_poi']
# Final features
create_new_feature()
print 'Final features_list: \n', features_list # 12 features

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print 'Top 5 useful features: \n', Select_K_Best(data_dict, features_list, 5)

### Scale features to a range between 0 and 1
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

###########################################
### Task 4: Try a varity of classifiers ###
###########################################
### Name your classifier clf for easy export below.

def algorithm_tester(clf):
    tester.dump_classifier_and_data(clf, my_dataset, features_list)
    return tester.main()

print "Algorithms with default parameter values \n"
'''
# Gaussian Naive Bayes
clf = GaussianNB()
algorithm_tester(clf)
### Accuracy: 0.82840	Precision: 0.34248	Recall: 0.31200	F1: 0.32653

# Decision tree
clf = DecisionTreeClassifier()
algorithm_tester(clf)
### Accuracy: 0.81253	Precision: 0.29618	Recall: 0.29500	F1: 0.29559

# RandomForest
clf = RandomForestClassifier()
algorithm_tester(clf)
### Accuracy: 0.86173	Precision: 0.44527	Recall: 0.15050	F1: 0.22496

# AdaBoost Classifier
clf = AdaBoostClassifier()
algorithm_tester(clf)
### Accuracy: 0.84600	Precision: 0.39948	Recall: 0.30800	F1: 0.34783
'''
###################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall ###
###################################################################################

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# By default, GridSearchCV uses the KFold cross-validation method
# (with 3 folds i.e. 3 train/validation splits).
# Because of the small dataset, I use StratifiedShuffleSplit, with at least 100 folds
# to give GridSearchCV a better chance of finding a good model

cv = StratifiedShuffleSplit(labels, n_iter = 100, test_size = 0.3, random_state = 42 )

############ Naive Bayes Classifier #############
'''
# Create a pipeline
nb_pipe = Pipeline([('select_features', SelectKBest(k=13)),
                    ('classify', GaussianNB())
                    ])
# Define parameter configuration
nb_params = dict(select_features__k = range(1,13))
# Find optimal parameters using GridSearchCV
clf = GridSearchCV(nb_pipe, param_grid=nb_params, scoring='f1', cv = cv)
clf.fit(features, labels)
print clf.best_params_
'''
# output: {'select_features__k': 5}

############ Decision Tree Classifier #############
'''
# Create a pipeline
tree_pipe = Pipeline([('select_features', SelectKBest(k=13)),
                    ('classify', DecisionTreeClassifier())
                    ])
# Define parameter configuration
tree_params = dict(select_features__k = range(1,13),
                  classify__min_samples_split=[2,4,6,8,10,20],
                  classify__max_depth=[None,5,10,15,20],
                  classify__max_features=[None,'sqrt','log2','auto'],
                  classify__criterion=['gini', 'entropy'])
# Find optimal parameters using GridSearchCV
clf = GridSearchCV(tree_pipe, param_grid=tree_params, scoring='f1', cv = cv)
clf.fit(features, labels)
print clf.best_params_
'''
### output: {'select_features__k': 6, 'classify__max_features': 'log2', 'classify__min_samples_split': 2, 'classify__criterion': 'gini', 'classify__max_depth': 10}

############ AdaBoost Classifier #############
'''
# Create a pipeline
ada_pipe = Pipeline([('select_features', SelectKBest(k=13)),
                    ('classify', AdaBoostClassifier())
                    ])
# Define parameter configuration
ada_params = dict(select_features__k = range(1,13),
                  classify__n_estimators = [50, 75, 100, 120])
# Find optimal parameters using GridSearchCV
clf = GridSearchCV(ada_pipe, param_grid=ada_params, scoring='f1', cv = cv)
clf.fit(features, labels)
print clf.best_params_
'''
### output: {'select_features__k': 6, 'classify__n_estimators': 120}


print "Algorithms with best parameter values \n"

# Gaussian Naive Bayes
nb_clf = Pipeline([('select_features', SelectKBest(k=5)),
                    ('classify', GaussianNB())
                    ])
algorithm_tester(nb_clf)
### Accuracy: 0.85000	Precision: 0.42138	Recall: 0.33500	F1: 0.37326
'''
# Decision tree
tree_clf= Pipeline([('select_features', SelectKBest(k=6)),
                    ('classify', DecisionTreeClassifier(max_features='log2', min_samples_split=2, criterion='gini', max_depth=10))
                    ])
algorithm_tester(tree_clf)
### Accuracy: 0.82393	Precision: 0.33951	Recall: 0.33900	F1: 0.33925

# Adaboost
ada_clf = Pipeline([('select_features', SelectKBest(k=6)),
                    ('classify', AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=120))
                    ])
algorithm_tester(ada_clf)
'''
### Accuracy: 0.86733	Precision: 0.50584	Recall: 0.21650	F1: 0.30322



##############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can ###
##############################################################################

### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(nb_clf, my_dataset, features_list)
