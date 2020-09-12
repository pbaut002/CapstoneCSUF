import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers


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