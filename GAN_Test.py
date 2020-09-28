import sys
sys.path.insert(0, './Modules')

from GAN import GAN
from NeuralNetworks import *
from DataHelper import *
import scipy.io as scp
import numpy as np
import pandas as pd


# Load dataset and set up features
education_data = pd.read_csv(
    "./Processed_Data/clean_data.csv", index_col=False)
features = education_data.columns.values
features = np.delete(features, -1)

RNNShape = [len(features), 1]
GAN_NN = GAN(features, filepath="./Processed_Data/clean_data.csv")

# Initialize models for the GAN
D_Network = RNNDiscriminator(education_data)
G_Network = generatorModelModified(education_data)

epoch = 800
checkpoint_steps = 5
GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)
print("Initial generation", GAN_NN.generateFakeData(size=1))

print("Training Network...")

test = GAN_NN.train_network(epochs=epoch, batch_size=8, history_steps=checkpoint_steps)

print("Finished Training, creating histogram")

while True:
    try:
        GAN_NN.animateHistogram(epoch, checkpoint_steps)
        print("Final generation", GAN_NN.generateFakeData(size=1))
        d = GAN_NN.generateFakeData(size=len(education_data))
        d.to_csv("./Project_Data/GeneratedData.csv")
        GAN_NN.saveLossHistory()
        sampleStudents = d.sample(20).to_numpy()
        showStudentGradeHeatMap(d.to_numpy(), features, save=True,
                                save_path='./Project_Data/GeneratedHeatmap.png',  
                                title="Generated Student Grades Over a Semester")
        createHistogram(d, save_path='./Project_Data/GeneratedStudentHistogram.png', title='Histogram of Generated Student Grades')
        break
    except:
        print('Make sure files are closed')
        s = input('Press Enter to Continue')




