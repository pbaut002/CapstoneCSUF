import sys
sys.path.insert(0, './Modules')

from GAN import GAN
from NeuralNetworks import *
from DataHelper import *
import scipy.io as scp
import numpy as np
import pandas as pd

DATA_PARAMETERS = {
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

HYPERPARAMETERS = {
    'Epochs' : 1500,
    'Checkpoint Frequency' : 5,
    'Batch Size' : 24
}

currentData = 'Correlations'
fileInformation = DATA_PARAMETERS[currentData]

dataFile = fileInformation['dataPath']
folder = fileInformation['folderName']


# Load dataset and set up features
education_data = pd.read_csv(
    dataFile, index_col=False)

GAN_NN = GAN(filepath=dataFile)

# Initialize models for the GAN
D_Network = RNNDiscriminator(education_data)
G_Network = generatorModelModified(education_data)


GAN_NN.initializeNetworks(generator=G_Network, discriminator=D_Network)
print("Initial generation\n", GAN_NN.generateFakeData(size=1))
print("Training Network...")

test = GAN_NN.train_network(epochs=HYPERPARAMETERS['Epochs'], 
                            batch_size=HYPERPARAMETERS['Batch Size'], 
                            history_steps=HYPERPARAMETERS['Checkpoint Frequency'],
                            checkpoint_path=fileInformation['checkpointPath'])

print("Finished Training, creating histogram")

while True:
    try:
        showPerformance(education_data, 'Real Student Performance', save_path=folder + 'RealStudentPerformance.png')
        GAN_NN.animateHistogram(epoch, checkpoint_steps, save_path=folder + 'Histogram.mp4')
        print("Final generation\n", GAN_NN.generateFakeData(size=1))
        d = GAN_NN.generateFakeData(size=len(education_data))
        d.to_csv(folder + 'GeneratedData.csv')

        GAN_NN.saveLossHistory(folder + 'LossHistory')

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




