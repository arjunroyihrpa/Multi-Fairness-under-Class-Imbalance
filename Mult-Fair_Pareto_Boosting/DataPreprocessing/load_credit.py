from __future__ import division
# import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

# import utils as ut

SEED = 1234
seed(SEED)
np.random.seed(SEED)



def load_credit():
	FEATURES_CLASSIFICATION = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0',
							   'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1',
							   'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
							   'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
							   'PAY_AMT5','PAY_AMT6']


	CONT_VARIABLES = ['LIMIT_BAL','EDUCATION','AGE','PAY_0',
							   'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1',
							   'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
							   'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
							   'PAY_AMT5','PAY_AMT6'] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = "default.payment.next.month" # the decision variable
	SENSITIVE_ATTRS = ["SEX",'MARRIAGE','AGE']
	p_group=[0,0,0]


	# COMPAS_INPUT_FILE = "bank-full.csv"
	COMPAS_INPUT_FILE = "DataPreprocessing/UCI_Credit_Card.csv"


	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)

	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])


	""" Feature normalization and one hot encoding """
	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
	y =  np.array([int(k) for k in y])
	y[y==0] = -1
	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	x_control = defaultdict(list)

	feature_names = []
	for attr in FEATURES_CLASSIFICATION:
		vals = data[attr]
		if attr=='AGE':
			va=[v for v in vals]           
		if attr=='MARRIAGE':
			vals=[0 if v==1 else 1 for v in vals]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals)  # 0 mean and 1 variance
			vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col
			if attr=='AGE':
			    v=[vals[i][0] for i in range(len(vals)) if va[i]==25 or va[i]==60]
			    v=list(set(v))
			    print(v)
			    p_group[SENSITIVE_ATTRS.index('AGE')]=[min(v),max(v)]
            
		else:  # for binary categorical variables, the label binarizer uses just one var instead of two
			lb = preprocessing.LabelBinarizer()
			lb.fit(vals)
			vals = lb.transform(vals)

		# add to sensitive features dict
		if attr in SENSITIVE_ATTRS:
			x_control[attr] = vals

		# add to learnable features
		X = np.hstack((X, vals))

		if attr in CONT_VARIABLES:  # continuous feature, just append the name
			feature_names.append(attr)
		else:  # categorical features
			if vals.shape[1] == 1:  # binary features that passed through lib binarizer
				feature_names.append(attr)
			else:
				for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
					feature_names.append(attr + "_" + str(k))

	# convert the sensitive feature to 1-d array
	x_control = dict(x_control)
	for k in x_control.keys():
		assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()

	feature_names.append('target')
	print( "Features we will be using for classification are:", feature_names, "\n")
	# print (np.sum(X[:,feature_names.index(SENSITIVE_ATTRS[0])]))
	#print (len(X[:,feature_names.index(SENSITIVE_ATTRS[0])]))
	return X, y, [feature_names.index(SENSITIVE_ATTRS[0]),feature_names.index(SENSITIVE_ATTRS[1]),feature_names.index(SENSITIVE_ATTRS[2])], p_group, x_control,feature_names

# sensitive_attrs = x_control.keys()