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
real, percentage, letter = splitKeywords(
    dataset, "Real", "Percentage", "Letter", "sadfasd")
dataset = dataset[percentage]

# Clean data and save it to a new file
cleanDataset(dataset)
cleanDataName(dataset, readable=True)

dataset.sort_index(axis=1, inplace=True)
features = getHighestCorrFeatures(dataset)
features = sorted(list(features))
# with open('./Processed_Data/CorrelationFeatures.csv', 'w', newline='') as filehandle:
#     writer = csv.writer(filehandle)
#     for feature in features:
#         w = feature.split(':')
#         writer.writerow(w)

# print(dataset.columns.values)
features = getFeatures(dataset.columns.values, "Quiz",
                       "Midterm exam total", "Assignment")
features = sorted(features)

# Display correlation table with readable and chosen features
features = np.delete(features, np.where(features == 'Quizzestotal'))
corr_features = ["Quiz {} ".format(x)
                 for x in range(1, 13)] + ['Midterm exam total ', 'Final exam total ']
showStudentCorrelation(dataset[corr_features])
features.sort()


# Clean up names, remove spaces for Tensorflow readability
cleanDataName(dataset, readable=False)
features = ["Quiz{}".format(x) for x in range(
    1, 13)] + ['Midtermexamtotal', 'Finalexamtotal']

# Create an initial map of the real data
education_data = (dataset[features]).sort_values(by=features)
education_data = education_data.fillna(0.0).clip(0,100)


sampleStudents = education_data.sample(20).to_numpy()
showStudentGradeHeatMap(sampleStudents, features, save_path="./Project_Data/InitialHeatmap.png")

# Save the cleaned data
education_data = (education_data.replace(
    to_replace="-", value=0.0)).astype("float64")
education_data = education_data.fillna(0.0).clip(0,100)
label = np.full_like(len(education_data), 1)
education_data['real'] = label
education_data.to_csv("./Processed_Data/clean_data.csv", index=False)

createHistogram(education_data)