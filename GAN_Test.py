import sys
sys.path.insert(0, './Modules')

import pandas as pd
import numpy as np
import scipy.io as scp

from DataHelper import *
from NeuralNetworks import *

from GAN import GAN

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load dataset and set up features
education_data = pd.read_csv("./Processed_Data/clean_data.csv", index_col=False)
features = education_data.columns.values
features = np.delete(features, -1)

GAN_NN = GAN(education_data, features, realData=True, filepath="./Processed_Data/clean_data.csv")

# Initialize models for the GAN
D_Network = discriminatorModel(education_data)
G_Network = generatorModel(education_data)

GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)
print("Initial generation", GAN_NN.generateFakeData(size=1))

print("Training Network...")
test = GAN_NN.train_network(epochs=300,batch_size=16)

print("Finished Training, creating histogram")
GAN_NN.animateHistogram()
print("Final generation", GAN_NN.generateFakeData(size=1))

d = GAN_NN.generateFakeData(size=100)

d.to_csv("./GeneratedData.csv")
GAN_NN.saveLossHistory()
showStudentGradeHeatMap(d.to_numpy(), features, save=True, save_path='./Project_Data/GeneratedHeatmap.png',  title="Generated Student Grades Over a Semester")
