import sys
sys.path.insert(0, './Modules')

from GAN import GAN
from NeuralNetworks import *
from DataHelper import *
import scipy.io as scp
import numpy as np
import pandas as pd

savePaths = {
    'Quizzes': {
        'dataPath'   :'./Processed_Data/QuizMidtermData.csv',
        'folderName' : './Project_Data/QuizMidterms/',
        'checkpointPath': 'E:\\training_checkpoints\\Quizzes'
    },
    'Correlations' : {
        'dataPath'   : './Processed_Data/CleanCorrData.csv',
        'folderName' : './Project_Data/CorrelationFeatures/',
        'checkpointPath': 'E:\\training_checkpoints\\Correlations'
    },
}

currentData = 'Correlations'

dataFile = savePaths[currentData]['dataPath']
folder = savePaths[currentData]['folderName']

# Load dataset and set up features
education_data = pd.read_csv(
    dataFile, index_col=False)

features = education_data.columns.values

showPerformance(education_data, 'Real Student Performance', save_path=folder + 'RealStudentPerformance.png')

RNNShape = [len(features), 1]
GAN_NN = GAN(features, filepath=dataFile)

# Initialize models for the GAN
D_Network = RNNDiscriminator(education_data)
G_Network = generatorModelModified(education_data)

epoch = 1500
checkpoint_steps = 5
GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)
print("Initial generation\n", GAN_NN.generateFakeData(size=1))

print("Training Network...")

batch = len(education_data) # Quiz round(len(education_data)/4)
test = GAN_NN.train_network(epochs=epoch, batch_size=batch, history_steps=checkpoint_steps,checkpoint_path=savePaths[currentData]['checkpointPath'])

print("Finished Training, creating histogram")

while True:
    try:
        GAN_NN.animateHistogram(epoch, checkpoint_steps, save_path=folder + 'Histogram.mp4')
        print("Final generation\n", GAN_NN.generateFakeData(size=1))
        d = GAN_NN.generateFakeData(size=len(education_data))
        d.to_csv(folder + 'GeneratedData.csv')
        GAN_NN.saveLossHistory(folder + 'LossHistory')
        sampleStudents = d.sample(20).to_numpy()
        showStudentGradeHeatMap(d.to_numpy(), features, save=True,
                                save_path=folder + 'GeneratedHeatmap.png',  
                                title="Generated Student Grades Over a Semester")
        createHistogram(d, save_path=folder + 'GeneratedStudentHistogram.png', title='Histogram of Generated Student Grades')
        showPerformance(d, 'Generated Student Performance', save_path=folder + 'GeneratedStudentPerformance.png')
        showPerformanceOverlap(education_data, d, 'Class Average Performance',  save_path=folder + 'ClassPerformance.png')
        break
    except Exception as e:
        print(e)
        print('Make sure files are closed')
        s = input('Press Enter to Continue')




