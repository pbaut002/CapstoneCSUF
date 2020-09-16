import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers


def discriminatorModel(dataset):

	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256,  
			kernel_regularizer='l1', 
			input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, kernel_regularizer='l2'))
	model.add(layers.Dense(1))
	return model

def generatorModel(dataset):
	features = [k for k in dataset.columns.values if k!='real']
	
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, activation="relu"))
	model.add(layers.Dense(len(features)))

	return model


def RNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	
	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features),1]))
	model.add(layers.LSTM(256, return_sequences=True))
	model.add(layers.LSTM(256, return_sequences=True))
	model.add(layers.Dense(1))
	return model

def generatorModelModified(dataset):
	features = [k for k in dataset.columns.values if k!='real']
	
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, activation="relu"))
	model.add(layers.Dense(len(features)))
	model.add(layers.Reshape([len(features),1]))

	return model

def CNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k,dtype=tf.dtypes.float64) 
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	
	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.SimpleRNN(128))
	model.add(layers.Dense(1))
	return model