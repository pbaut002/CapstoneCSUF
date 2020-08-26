## GAN NEURAL NETWORK LIBRARY
import os 
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow.keras.layers as layers

from math import ceil

def createTestSet(dataset,size=20):
	"""
	Returns a training and validation set
	@param dataset: Original dataset to be shuffled and batch
	@param size: Size of batch
	"""
	dataset = dataset.copy()
	results = np.array(dataset.pop('real'))
	dataset = tf.data.Dataset.from_tensor_slices((dict(dataset),results))
	return dataset

def getFeatures(dataset):
	return np.array([k for k in dataset.columns.values if k != 'real'])


def splitKeywords(dataframe,*args):
	"""
	Splits into different datasets based on the keyword
	splitKeywords("Real): Pandas dataset that contains the word "Real"
	Primarily used to split the dataset into different types of grading scales

	@param: dataset: Original dataset
	@param: *args: Keywords to split by
	"""
	dataset_splits = {}
	for kw in args:
		if dataset_splits.get(kw) == None:
			dataset_splits[kw] = []
		try:
			dataset_splits[kw] = [col for col in dataframe if (kw in col and len(re.findall(r"\)\.1",col)) == 0)]
			if len(dataset_splits[kw]) == 0:
				print("No column named",kw)
				del dataset_splits[kw]
		except:
			pass
	return dataset_splits.values()


def cleanDataName(dataset):

	def cleanNames(column_name):
		column_name = re.sub(r"[' ',':','(',')']|Real|(Percentage)|Quiz:|Assignment:","",column_name)
		return column_name
	
	dataset.rename(cleanNames, axis='columns',inplace=True)

def cleanDataset(dataset):
	"""
	Cleans the columns
	Removes empty cells and replaces it with a 0 or null keyword
	Columns that contain 25% missing data are automatically dropped

	@param: dataset: Original dataset
	"""

	# Remove a column if its column contains more than 25% empty values
	for col in dataset:
		value_freq = dataset[col].value_counts().to_dict()
		num_blank = value_freq.get("-")
		if num_blank != None:
			if num_blank > len(dataset)*.25:
				dataset.drop(col, axis=1, inplace=True)

	# Replace percent values into real numbers i.e. 25% ==> 25.0
	for col in dataset.columns.values:
		dataset.replace(" %","",regex=True,inplace=True)
		dataset[col] = dataset[col].apply(pd.to_numeric,errors='coerce')    

def discriminatorModel(dataset):

	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.Dense(1))
	return model


def generatorModel(dataset):
	features = [k for k in dataset.columns.values if k!='real']
	
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.Dense(20, activation="relu"))
	model.add(layers.Dense(30, activation="tanh"))
	model.add(layers.Dense(40, activation="relu"))
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dense(len(features)))

	return model

