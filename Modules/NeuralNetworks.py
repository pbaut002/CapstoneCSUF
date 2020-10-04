import tensorflow.keras.layers as layers
import tensorflow as tf
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def discriminatorModel(dataset):

	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256,
						   kernel_regularizer='l1',
						   input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, kernel_regularizer='l2'))
	model.add(layers.Dense(1))
	return model


def RNNDiscriminator(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features), 1]))
	model.add(layers.SimpleRNN(128, return_sequences=True,
						  kernel_regularizer='l1', bias_regularizer='l2',
						  activation='relu'))
	model.add(layers.SimpleRNN(128, return_sequences=False,
						  kernel_regularizer='l1', bias_regularizer='l2',
						  activation='relu'))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(1))
	return model


def generatorModel(dataset):
	features = [k for k in dataset.columns.values if k != 'real']

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, activation="relu"))
	model.add(layers.Dense(len(features)))

	return model

def RNNGenerator(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	def customRELU(x):
		return tf.keras.activations.relu(x, max_value=160)

	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features), 1]))
	model.add(layers.SimpleRNN(128, return_sequences=True,
						  kernel_regularizer='l1', bias_regularizer='l2',
						  activation='relu'))
	model.add(layers.SimpleRNN(128, return_sequences=False,
						  kernel_regularizer='l1', bias_regularizer='l2',
						  activation='relu'))
	model.add(layers.Dropout(.15))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(len(features),
						   activation=customRELU))

	return model

def generatorModelModified(dataset):
	features = [k for k in dataset.columns.values if k != 'real']

	def customRELU(x):
		return tf.keras.activations.relu(x, max_value=100)

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),),
							kernel_regularizer='l2', bias_regularizer='l2'))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, activation='relu',
							kernel_regularizer='l1', bias_regularizer='l1'))
	model.add(layers.Dense(len(features),
						   kernel_regularizer='l1',
						   activation=customRELU))

	return model



def CNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(64, input_shape=(len(features),)))
	model.add(layers.SimpleRNN(128))
	model.add(layers.Dense(1))
	return model


if __name__ == "__main__":
	print('Can only be used as neural network module')
else:
	print('Successfully loaded', __name__)
