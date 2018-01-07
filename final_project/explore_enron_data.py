#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

features = {}
for person, value in enron_data.iteritems():
    features[person] = len(person)

# How many data points (people) are in the dataset?
print "Number of people:", len(features)

# For each person, how many features are available?
print "Features for each person:", features

# How many POIs (Person of Interest) are there in the E+F(email + finance) dataset?
count = 0
for person in enron_data:
    if enron_data[person]['poi'] == 1:
        count += 1
print "How many POIs are there?", count

# How many POIs were there in total? (in ../final_project/poi_names.txt)
count = 0
with open('../final_project/poi_names.txt', 'r') as f:
    for line in f:
        if '(y)' in line or '(n)' in line:
            count += 1

print "How many POIs were there in total?", count

# What is the total value of the stock belonging to James Prentice?
stock = enron_data['PRENTICE JAMES']['total_stock_value']
print "What is the total value of the stock belonging to James Prentice?", stock

# How many email messages do we have from Wesley Colwell to persons of interest?
email_to_pois =  enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "How many email messages do we have from Wesley Colwell to persons of interest?", email_to_pois

# What is the value of stock options exercised by Jeffrey K Skilling?
exercised_stock =  enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print "What is the value of stock options exercised by Jeffrey K Skilling?", exercised_stock

# Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of total_payments feature)?
print "Who took home the most money?"
print 'LAY KENNETH L:', enron_data['LAY KENNETH L']['total_payments']
print 'FASTOW ANDREW S:', enron_data['FASTOW ANDREW S']['total_payments']
print 'SKILLING JEFFREY K:', enron_data['SKILLING JEFFREY K']['total_payments']

# How many folks in this dataset have a quantified salary?
# What about a known email address?

count_salary = 0
count_email = 0
for person in enron_data:
    if enron_data[person]['salary'] != 'NaN':
        count_salary += 1
    if enron_data[person]['email_address'] != 'NaN':
        count_email += 1
print "How many folks in this dataset have a quantified salary?", count_salary
print "What about a known email address?", count_email

# How many people in the E+F dataset (as it currently exists) have NaN for their total payments? What percentage of people in the dataset as a whole is this?
count_total = 0
count_payment_nan = 0
for person in enron_data:
    count_total += 1
    if enron_data[person]['total_payments'] == 'NaN':
        count_payment_nan += 1
result = float(count_payment_nan)/count_total * 100
print "How many people have NaN for their total payments? What percentage of people in the dataset as a whole is this?", result
