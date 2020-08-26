import os 
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

from math import ceil



class GAN():

	def __init__(self, dataset, feature_names, generator=None, discriminator=None, realData=True,filepath=None):
		self.generator = generator
		self.discriminator = discriminator
		self.dataset = dataset.copy()
		self.features = feature_names
		self.filepath = filepath
		if realData==True:
			self.dataset['real'] = np.full(len(self.dataset), 1)

	def discriminatorLoss(self,real_output, fake_output):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		real_loss = cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		return total_loss
	
	def generatorLoss(self,fake_output):
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
		#data = (dataset.shuffle(buffer_size=50)).#shuffle(buffer_size=210)
		#random_data = data.batch(size)
		return dataset

	def generateFakeData(self,size=30):
		if self.generator == None:
			raise ValueError("There is no generator")
		
		fake_data = pd.DataFrame(columns=self.features)

		for x in range(size):
			noise_vector = tf.random.normal([1,len(self.features)],dtype=tf.float32)
			gen_output =self.generator(noise_vector, training=False)
			fake_data.loc[len(fake_data)] = gen_output[0].numpy()

		return fake_data


	def train_network(self, epochs, batch_size):
		"""
		Train the network for a number of epochs.
		@param epochs: Number of times it goes through a dataset
		"""

		def generateHistogram(self):
			self.history = []
			self.fig = plt.figure()

		def addEpoch(self, tensor):
			self.history.append(tensor)
		
		def animateHistograms(self):
			
			def update(tensor):
				plt.clf()
				plt.ylim(0,.1)
				plt.hist(tensor, bins=10, histtype='stepfilled', range=(-100,100),color='blue',density=True)

			
			animate =  animation.FuncAnimation(self.fig, update , self.history, interval=60, blit=False)
			animate.save('history.mp4')


		generateHistogram(self)

		if self.generator == None or self.discriminator == None:
			raise RuntimeError("Generator and/or discriminator not initialized")

		generator_optimizer = tf.keras.optimizers.RMSprop(1e-3)
		discriminator_optimizer = tf.keras.optimizers.RMSprop(1e-4)


		features = [k for k in self.dataset.columns.values]
		real_batch_data = tf.data.experimental.make_csv_dataset(
				self.filepath,
				batch_size,
				select_columns=features,
				label_name="real",
				num_epochs=1
			)
		

		def pack_features_vector(features, labels):
				features = tf.stack(list(features.values()),axis=1)
				return features, labels

		real_batch_data = real_batch_data.map(pack_features_vector)

		# Create a new set that consists of generated and real data for training
		for x in range(epochs):
			features, labels = next(iter(real_batch_data))

			for data_item in real_batch_data:
				noise_vector = tf.random.normal([batch_size,len(self.features)],dtype=tf.float32)

				with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :  
					
					gen_output =self.generator(noise_vector, training=True)

					true_predictions = self.discriminator(data_item, training=True)
					false_predictions = self.discriminator(gen_output, 
					training=True)
					loss_disc = self.discriminatorLoss(true_predictions, false_predictions)
					loss_gen = self.generatorLoss(false_predictions)
					
	
				gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
				gradients_of_discriminator = disc_tape.gradient(loss_disc, self.discriminator.trainable_variables)


				generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
				discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.variables))
			tf.print("Discriminator Loss: ", loss_disc)
			tf.print("Generator Loss: ", loss_gen)

			noise_vector = tf.random.normal([1,len(self.features)],dtype=tf.float32)
			addEpoch(self,self.generator(noise_vector, training=False))
	
		animateHistograms(self)
