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

def getFeatures(columnList, *args):
	if len(args) == 0:
		return np.array([k for k in dataset.columns.values if k != 'real'])
	else:
		features = set()
		for columnName in columnList:
			for keyword in args:
				if keyword.lower() in columnName.lower():
					features.add(columnName)
	
		return np.array(list(features))
	
	
def showStudentGradeHeatMap(grades, features):
	"""
	Credit: Matplotlib.org for majority of logic for the heatmap
	"""
	plt.close()
	fig, ax = plt.subplots()
	im = ax.imshow(grades)

	number_students = 15
	# We want to show all ticks...
	ax.set_xticks(np.arange(len(features)))
	ax.set_yticks(np.arange(number_students))
	# ... and label them with the respective list entries
	ax.set_xticklabels(features)

	students = ["Student {0}".format(k+1) for k in range(0,number_students)]
	ax.set_yticklabels(students)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			rotation_mode="anchor")

	## NOTE: Uncomment below to put numbers on individual grid
	# Loop over data dimensions and create text annotations.
	# for i in range(len(students)):
	# 	for j in range(len(features)):
	# 		text = ax.text(j, i, grades[i, j],
	# 					ha="center", va="center", color="w")

	ax.figure.colorbar(im, ax=ax)

	ax.set_title("Student Grades Over a Semester")
	ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

	fig.tight_layout()
	plt.show()
	plt.close()


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

def getHighestCorrFeatures(dataset):
	
	def create_corrMatrix(dataframe):
    # Create the correlation matrix and strip where all values NAN
		assert(len(dataframe) != None)
		matrix = dataframe.corr(method="pearson")
		np.fill_diagonal(matrix.values, np.nan)
		for x in matrix:
			value_freq = matrix[x].value_counts().to_dict()
			# Drop values if it is empty
			if len(value_freq) == 0:
				matrix.drop(x, axis=1, inplace=True)
				matrix.drop(x, axis=0, inplace=True)
		return matrix

	def find_highest_corr(data):
		highest_corr_labels = []
		for x in data:
			# Get columns with highest correlation values
			large = data[x].nlargest()
			for d in large.iteritems():
				keys = [x,d[0]]
				keys.sort()
				if (keys,d[1]) not in highest_corr_labels:
					highest_corr_labels.append((keys,d[1]))

		return highest_corr_labels

	def max_corr(val):
		return val[1]
	

	corrMatrix = create_corrMatrix(dataset)
	# Get the labels that have high correlations with other values
	highest_corr_labels = find_highest_corr(corrMatrix)
	highest_corr_labels.sort(key=max_corr,reverse=True)
	highest_corr_labels = highest_corr_labels[:ceil(len(highest_corr_labels)*.25)]

	relevant_labels = set()
	for keys in highest_corr_labels:
		for k in keys[0]:
			relevant_labels.add(k)

	return relevant_labels


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


def RNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.SimpleRNN(128))
	model.add(layers.Dense(1))
	return model

def CNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.SimpleRNN(128))
	model.add(layers.Dense(1))
	return model