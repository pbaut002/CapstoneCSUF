import csv
import scipy.io as scp
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, './Modules')

from DataHelper import *

#filename = input("Enter the filename: ")
dataset = pd.read_csv("./Datasets/StudentData_121.csv")
save_folder = './Project_Data/CorrelationFeatures/'

real, percentage, letter = splitKeywords(
    dataset, "Real", "Percentage", "Letter", "sadfasd")
dataset = dataset[percentage]

# Clean data and save it to a new file
cleanDataset(dataset)
cleanDataName(dataset, readable=True)

features = getHighestCorrFeatures(dataset)


# Display correlation table with readable and chosen features
features = np.delete(features, np.where(features == 'Quizzestotal'))
showStudentCorrelation(dataset[features], save_path=save_folder + 'CorrelationMatrix.png')

# Clean up names, remove spaces for Tensorflow readability
cleanDataName(dataset, readable=False)
features = getHighestCorrFeatures(dataset)

# Create an initial map of the real data
education_data = (dataset[features])
education_data = education_data.fillna(0.0).clip(0,100)


sampleStudents = education_data.sample(20).to_numpy()
showStudentGradeHeatMap(sampleStudents, features, save_path=save_folder + "InitialHeatmap.png")

# Save the cleaned data
education_data = (education_data.replace(
    to_replace="-", value=0.0)).astype("float64")
education_data = education_data.fillna(0.0).clip(0,100)
label = np.full_like(len(education_data), 1)
education_data['real'] = label
education_data.to_csv("./Processed_Data/CleanCorrData.csv", index=False)

createHistogram(education_data, save_path=save_folder + 'RealStudentHistogram')