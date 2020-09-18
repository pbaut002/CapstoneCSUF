import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import random as rand

from math import ceil
from collections import Counter



class GAN():

	def __init__(self, feature_names, generator=None, discriminator=None, filepath=None, input_shape=None):
		self.generator = generator
		self.discriminator = discriminator
		self.features = feature_names
		self.filepath = filepath
		self.generator_optimizer = tf.keras.optimizers.RMSprop(1e-3)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

		if input_shape == None:
			self.input_shape = [1, len(self.features)]
		else:
			self.input_shape = input_shape
	def discriminatorLoss(self,real_output, fake_output):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		real_loss = cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss
	
	def generatorLoss(self,fake_output, similarOutput=0):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		return cross_entropy(tf.ones_like(fake_output), fake_output)


	def initializeNetworks(self,generator=None,discriminator=None):
		"""
		Allows users to create new generator and discriminators from scratch
		or use their own generator/discriminator
		@param generator: If None, create a new pre-built generator network. Assign otherwise
		@param discriminator: same as generator
		"""
		if generator == None:
			raise ValueError("No Generator Network was initialized.")
		else:
			self.generator = generator

		if discriminator == None:
			raise ValueError("No Discriminator Network was initialized.")
		else:
			self.discriminator = discriminator
	

	def createDatasets(self,size=20,dataset=None,results=None):
		"""
		Returns a training and validation set
		@param dataset: Original dataset to be shuffled and batch
		@param size: Size of batch
		"""
		dataset = dataset.copy()
		results = np.array(dataset.pop('real'))
		dataset = tf.data.Dataset.from_tensor_slices((dict(dataset),results))
		return dataset

	def generateFakeData(self,size=30, shape=None):
		if self.generator == None:
			raise ValueError("There is no generator")
		if shape == None:
			shape = [1, len(self.features)]
		fake_data = pd.DataFrame(columns=self.features)

		for x in range(size):
			noise_vector = self.generateNoiseVector(1)
			gen_output = self.generator(noise_vector, training=False)
			gen_output = tf.reshape(gen_output, shape)
			fake_data.loc[len(fake_data)] = gen_output[0].numpy()

		return fake_data

	def generateNoiseVector(self, size=30):
		c = tf.random.normal([size,len(self.features)], mean=0.0, stddev=12.0, dtype=tf.float32)
		return c

	def animateHistogram(self, save_path='./Project_Data/Histogram.mp4'):
		max_value = np.amax(self.distribution_history)
		min_value = np.amin(self.distribution_history)
		def update(tensor):
				plt.clf()
				plt.xlabel('Grades (%)')
				plt.ylabel('Probability')
				plt.ylim(0,.1)
				tensor = np.concatenate(tensor, axis=None)
				plt.hist(tensor, bins=10, histtype='stepfilled', range=(min_value,max_value),color='blue',density=True)
		print("Making history")	
		try:
			animate =  animation.FuncAnimation(self.fig, update , self.distribution_history, interval=60, blit=False)
			animate.save(save_path)
		except:
			print("Training history not defined")
		finally:
			plt.close()
	
	def saveLossHistory(self, save_path='./Project_Data/LossHistory.png'):
		plt.figure()
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('GAN Network Losses')
		try:
			plt.plot(range(0, len(self.loss_history_generator)), self.loss_history_generator, label='Generator Loss', lw=1.2)
			plt.plot(range(0, len(self.loss_history_discriminator)), self.loss_history_discriminator, label= 'Discriminator Loss', lw=1.2)
			plt.legend()
			plt.savefig(save_path)
		except:
			raise AttributeError('Must train network before trying to view network losses')
		finally:
			plt.close()

	def createTrainingBatchData(self, batch_size=32):
		
		def pack_features_vector(features, labels):
			features = tf.stack(list(features.values()),axis=1)
			return features, labels

		real_batch_data = tf.data.experimental.make_csv_dataset(
				self.filepath,
				batch_size,
				label_name="real",
				num_epochs=1
			)
		
		real_batch_data = real_batch_data.map(pack_features_vector)
		return real_batch_data

	def train_network(self, epochs=10, batch_size=32, history_steps=1):
		"""
		Train the network for a number of epochs.
		@param epochs: Number of times it goes through a dataset
		@param batch_size: Number of examples when training
		@param history_steps: Take a snapshot of generator distribtuion for every number of steps
		"""

		def trackHistory(self):
			self.distribution_history = []
			self.loss_history_generator = []
			self.loss_history_discriminator = []
			self.fig = plt.figure()

		def addEpochToHistory(tensor):
			self.distribution_history.append(tensor)
					
		def checkArrayDifference(output):
			numSame = 0
			for x in output:
				random_array = rand.choice(output)
				values = Counter(np.isclose(x,random_array, rtol=.5))
				if values[True] == len(x): numSame = numSame + 1
			print(numSame)
			return numSame

		trackHistory(self)

		if self.generator == None or self.discriminator == None:
			raise RuntimeError("Generator and/or discriminator not initialized")

		batchData = self.createTrainingBatchData(batch_size)

		# Create a new set that consists of generated and real data for training
		for x in range(epochs):
			features, labels = next(iter(batchData))

			for data_item in batchData:
				
				with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :  
				
					noise_vector = self.generateNoiseVector(ceil(batch_size * (1/4)))
					gen_output = self.generator(noise_vector, training=True)
					
					true_predictions = self.discriminator(data_item[0], training=True)
					false_predictions = self.discriminator(gen_output, 
					training=True)

					loss_disc = self.discriminatorLoss(true_predictions, false_predictions)
					loss_gen = self.generatorLoss(false_predictions)

				gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
				gradients_of_discriminator = disc_tape.gradient(loss_disc, self.discriminator.trainable_variables)

				self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
				self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.variables))
			
			self.loss_history_generator.append(tf.cast(loss_gen, float))
			self.loss_history_discriminator.append(tf.cast(loss_disc,float))
			
			if ((x % history_steps) == 0):
				tf.print("Epoch:", x)
				noise_vector = self.generateNoiseVector(2)
				tf.print('Noise Vector:',noise_vector)
				generated_grades = self.generator(noise_vector, training=False)
				tf.print('Generated Grades:',generated_grades)
				addEpochToHistory(generated_grades)
		
