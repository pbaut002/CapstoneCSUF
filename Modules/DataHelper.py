# GAN NEURAL NETWORK LIBRARY
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

from math import ceil


def truncate(name):
        if len(name) > 20:
            return name[:21]
        return name
    

def getFeatures(columnList, *args):
    if len(args) == 0:
        return np.array([k for k in dataset.columns.values if k != 'real'])
    else:
        features = set()
        for columnName in columnList:
            for keyword in args:
                if keyword.lower() in columnName.lower():
                    features.add(columnName)

        return np.array(list(features))

def showPerformance(dataset, title, save_path='./Project_Data/StudentGradeHeatMap.png'):
    if len(dataset) != 0:
        plt.close()
        class_assignments = dataset.loc[:, dataset.columns != 'real']

        plt.title(title)
        plt.plot(class_assignments.columns.values, class_assignments.mean(axis=0))
        plt.xticks(class_assignments.columns.values, rotation=45, ha="right",
             rotation_mode="anchor")
        plt.yticks(range(0,101,10))
        plt.tight_layout()
        plt.savefig(save_path)

def showPerformanceOverlap(dataset1, dataset2, title, save_path='./Project_Data/StudentGradeHeatMap.png'):
    if len(dataset1) != 0:
        plt.close()
        size = min(14, len(dataset1.columns.values))
        
        columns = dataset1.columns.values[:size]
        
        dataset1 = dataset1[columns]
        dataset2 = dataset2[columns]
        
        class_assignments1 = dataset1.loc[:, dataset1.columns != 'real']
        class_assignments2 = dataset2.loc[:, dataset2.columns != 'real']
        
        plt.title(title)
        plt.plot(list(map(truncate, class_assignments1.columns.values)), class_assignments1.mean(axis=0),c='red', label='Real')
        plt.plot(list(map(truncate, class_assignments2.columns.values)), class_assignments2.mean(axis=0),c='blue', label='Generated', linestyle='dashed')
        plt.legend()
        plt.xticks(class_assignments1.columns.values, rotation=45, ha="right",
             rotation_mode="anchor")
        plt.yticks(range(0,101,10))
        plt.tight_layout()
        plt.savefig(save_path)

def showStudentGradeHeatMap(grades, save=True, save_path='./Project_Data/StudentGradeHeatMap.png', title="Student Grades Over a Semester"):
    """
    Credit: Matplotlib.org for majority of logic for the heatmap
    """
    features = grades.columns.values
    grades = grades.to_numpy()

    number_students = 15
    number_assignments = min(14,  len(features))
    
    features = list(map(truncate, features))[:number_assignments]
    plt.close()
    fig, ax = plt.subplots()
    im = ax.imshow(grades[:number_students, :number_assignments], aspect='auto')

    
    # We want to show all ticks...
    ax.set_xticks(np.arange(number_assignments))
    ax.set_yticks(np.arange(number_students))
    # ... and label them with the respective list entries
    ax.set_xticklabels(features)

    students = ["Student {0}".format(k+1) for k in range(0, number_students)]
    ax.set_yticklabels(students)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # NOTE: Uncomment below to put numbers on individual grid
    # Loop over data dimensions and create text annotations.
    # for i in range(len(students)):
    # 	for j in range(len(features)):
    # 		text = ax.text(j, i, grades[i, j],
    # 					ha="center", va="center", color="w")

    ax.figure.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.close()


def showStudentCorrelation(dataset, save=True, save_path='./Project_Data/CorrelationMatrix.png'):
    """
    Credit: Matplotlib.org for majority of logic for the heatmap
    """

    plt.close()
    number_students = min(15, len(dataset.columns.values))
    dataset = dataset[dataset.columns.values[:number_students]].corr(method="pearson")
    column_vals = list(map(truncate, dataset.columns.values))

    fig, ax = plt.subplots()
    im = ax.imshow(dataset, aspect='auto', cmap='YlGn')

    # We want to show all ticks...
    ax.set_xticks(np.arange(number_students))
    ax.set_yticks(np.arange(number_students))
    # ... and label them with the respective list entries
    ax.set_xticklabels(column_vals)
    ax.set_yticklabels(column_vals)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # NOTE: Uncomment below to put numbers on individual grid
    # Loop over data dimensions and create text annotations.
    grades = dataset.to_numpy()
    for i in range(number_students):
        for j in range(number_students):
            text = ax.text(j, i, round(grades[i, j], 1),
                           ha="center", va="center", color="black")

    ax.figure.colorbar(im, ax=ax)

    ax.set_title("Correlation Matrix")
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))

    fig.tight_layout()
    if save:
        plt.savefig(save_path)
    plt.close()


def splitKeywords(dataframe, *args):
    """
    Splits into different datasets based on the keyword
    splitKeywords("Real): Pandas dataset that contains the word "Real"
    Primarily used to split the dataset into different types of grading scales

    @param: dataset: Original dataset
    @param: *args: Keywords to split by
    """
    dataset_splits = {}
    for kw in args:
        if dataset_splits.get(kw) == None:
            dataset_splits[kw] = []
        try:
            dataset_splits[kw] = [col for col in dataframe if (
                kw in col and len(re.findall(r"\)\.1", col)) == 0)]
            if len(dataset_splits[kw]) == 0:
                print("No column named", kw)
                del dataset_splits[kw]
        except:
            pass

    if len(dataset_splits) == 1:
        return dataset_splits[kw]
    return dataset_splits.values()


def cleanDataName(dataset, readable=True):

    def cleanNames(column_name):
        if readable:
            column_name = re.sub(
                r"['',':','(',')']|Real|(Percentage)|Quiz:|Assignment:", "", column_name)
        else:
            column_name = re.sub(
                r"[' ',':','(',')']|Real|(Percentage)|Quiz:|Assignment:", "", column_name)
        return column_name

    dataset = dataset.rename(cleanNames, axis='columns')

    return dataset


def cleanDataset(dataset):
    """
    Cleans the columns
    Removes empty cells and replaces it with a 0 or null keyword
    Columns that contain 25% missing data are automatically dropped

    @param: dataset: Original dataset
    """

    # Remove a column if its column contains more than 25% empty values
    dataset = dataset.copy()

    for col in dataset:
        value_freq = dataset[col].value_counts().to_dict()
        num_blank = value_freq.get("-")
        num_zero = value_freq.get(0)
        if num_blank != None:
            if num_blank > len(dataset)*.25:
                dataset = dataset.drop(col, axis=1)
        if num_zero != None:
            if num_zero > len(dataset)*.25:
                dataset = dataset.drop(col, axis=1, inplace=True)
    dataset.replace(" %", "", regex=True, inplace=True)

    # Replace percent values into real numbers i.e. 25% ==> 25.0
    for col in dataset.columns.values:
        dataset[col] = dataset[col].apply(pd.to_numeric, errors='coerce')
    
    dataset = dataset.replace(to_replace="-", value=0.0).astype("float64")
    dataset = dataset.fillna(0.0).clip(0,100)

    return dataset


def getHighestCorrFeatures(dataset):

    def create_corrMatrix(dataframe):
        # Create the correlation matrix and strip where all values NAN
        assert(len(dataframe) != None)
        matrix = dataframe.corr(method="pearson")
        np.fill_diagonal(matrix.values, np.nan)
        for x in matrix:
            value_freq = matrix[x].value_counts().to_dict()
            # Drop values if it is empty
            if len(value_freq) == 0:
                matrix.drop(x, axis=1, inplace=True)
                matrix.drop(x, axis=0, inplace=True)

        return matrix

    def find_highest_corr(data):
        highest_corr_labels = []
        for x in data:
            # Get columns with highest correlation values
            large = data[x].nlargest()
            for d in large.iteritems():
                keys = [x, d[0]]
                keys.sort()
                if (keys, d[1]) not in highest_corr_labels:
                    highest_corr_labels.append((keys, d[1]))

        return highest_corr_labels

    def max_corr(val):
        return val[1]

    corrMatrix = create_corrMatrix(dataset)
    # Get the labels that have high correlations with other values
    highest_corr_labels = find_highest_corr(corrMatrix)
    highest_corr_labels.sort(key=max_corr, reverse=True)
    highest_corr_labels = highest_corr_labels[:ceil(
        len(highest_corr_labels)*.25)]

    relevant_labels = set()
    for keys in highest_corr_labels:
        for k in keys[0]:
            relevant_labels.add(k)
    return [label for label in dataset.columns.values if label in relevant_labels]

def createHistogram(dataset: pd.DataFrame, save=True, save_path='./Project_Data/RealStudentHistogram.png', title='Histogram of Real Student Grades'):
    tensor = dataset.to_numpy().flatten()
    max_value = 100
    min_value = 0

    plt.clf() 
    plt.title(title)
    plt.xlabel('Grades (%)')
    plt.ylabel('Frequency Density')
    plt.ylim(0,.1)
    plt.hist(tensor, bins=10, histtype='stepfilled', range=(min_value,max_value),color='blue',density=True)
    plt.savefig(save_path)
