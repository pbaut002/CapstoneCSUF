import pandas as pd
import numpy as np
import scipy.io as scp

from DataHelper import *

from GAN import GAN

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#filename = input("Enter the filename: ")
dataset = pd.read_csv("./Datasets/StudentData_121.csv")
real, percentage, letter = splitKeywords(dataset,"Real","Percentage","Letter","sadfasd")
dataset = dataset[percentage]

# Clean data and save it to a new file
cleanDataset(dataset)
cleanDataName(dataset)

features = getHighestCorrFeatures(dataset)
features = getFeatures(dataset.columns.values,"Quiz", "Midterm exam total", "Assignment")
features = np.delete(features, np.where(features=='Quizzestotal'))
features.sort()
print(features)


education_data = (dataset[features]).sort_index(axis=1)

showStudentGradeHeatMap(dataset[features].to_numpy(), features, save_path="./Project_Data/InitialHeatmap.png")

education_data = (education_data.replace(to_replace="-",value=0.0)).astype("float64")
education_data = education_data.fillna(0.0)
label  = np.full_like(len(education_data),1)
education_data['real'] = label
education_data.to_csv("./Processed_Data/clean_data.csv")


# Randomly apply labels to the dataset for testing purposes
GAN_NN = GAN(education_data, features, realData=False, filepath="./Processed_Data/clean_data.csv")

# Initialize models for the GAN
D_Network = discriminatorModel(education_data)
G_Network = generatorModel(education_data)

GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)
print("Initial generation", GAN_NN.generateFakeData(size=1))
test = GAN_NN.train_network(epochs=300,batch_size=16)


GAN_NN.animateHistogram()
print("Final generation", GAN_NN.generateFakeData(size=1))

d = GAN_NN.generateFakeData(size=100)

d.to_csv("./GeneratedData.csv")
GAN_NN.saveLossHistory()
showStudentGradeHeatMap(d.to_numpy(), features, save=True, save_path='./Project_Data/GeneratedHeatmap.png')
