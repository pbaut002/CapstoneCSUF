import sys
sys.path.insert(0, './Modules')

import pandas as pd
import numpy as np
import scipy.io as scp

from DataHelper import *

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#filename = input("Enter the filename: ")
dataset = pd.read_csv("./Datasets/StudentData_121.csv")
real, percentage, letter = splitKeywords(dataset,"Real","Percentage","Letter","sadfasd")
dataset = dataset[percentage]

# Clean data and save it to a new file
cleanDataset(dataset)
cleanDataName(dataset, readable=True)

dataset.sort_index(axis=1, inplace=True)
features = getHighestCorrFeatures(dataset)

features = getFeatures(dataset.columns.values,"Quiz", "Midterm exam total", "Assignment")
features = sorted(features)

# Display correlation table with readable and chosen features
features = np.delete(features, np.where(features=='Quizzestotal'))
corr_features = ["Quiz {} ".format(x) for x in range(1, 13)]
showStudentCorrelation(dataset[corr_features])
features.sort()


cleanDataName(dataset, readable=False)
features = ["Quiz{}".format(x) for x in range(1, 13)]

education_data = (dataset[features]).sort_values(by=features)
showStudentGradeHeatMap(dataset[features].to_numpy(), features, save_path="./Project_Data/InitialHeatmap.png")

education_data = (education_data.replace(to_replace="-",value=0.0)).astype("float64")
education_data = education_data.fillna(0.0)
label  = np.full_like(len(education_data),1)
education_data['real'] = label
education_data.to_csv("./Processed_Data/clean_data.csv", index=False)

