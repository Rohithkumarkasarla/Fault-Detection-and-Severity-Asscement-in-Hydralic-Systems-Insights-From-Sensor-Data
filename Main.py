from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import numpy as np
import pandas as pd
from time import time
import random
import sklearn.utils as sf
import sklearn.model_selection as ms
import sklearn.metrics as eva
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.linear_model import SGDClassifier

from xgboost import XGBClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

pd.options.mode.chained_assignment = None

# Tkinter UI
main = Tk()
main.title("Hydralics")
main.geometry("1300x1200")

global X,Y,X_train, X_test, Y_train, Y_test,RFC,pca


def uploadDataset():
    global X,Y 

    # Ask the user to select a directory containing the CSV files
    file_location = filedialog.askdirectory(initialdir=".")

    # Load the training dataset (Train_data.csv)
    train_file_path = os.path.join(file_location, "PS1_output.csv")
    X = pd.read_csv(train_file_path)

    # Load the testing dataset (Test_data.csv)
    test_file_path = os.path.join(file_location, "Target_data.csv")
    Y = pd.read_csv(test_file_path)

    # Display messages in the 'text' widget
    text.insert(END, "Training Dataset Loaded\n")
    text.insert(END, "Training Dataset Shape: " + str(X.shape) + "\n")

    text.insert(END, "Testing Dataset Loaded\n")
    text.insert(END, "Testing Dataset Shape: " + str(Y.shape) + "\n")
    
    text.insert(END, "Type the output coloumn name excatly in entry window \n\n")
    text.insert(END, "Coloumn Names: \n")
    text.insert(END, "Cooler condition\n")
    text.insert(END, "Valve condition\n")
    text.insert(END, "Internal pump leakage\n")
    text.insert(END, "Hydraulic accumulator\n")
    text.insert(END, "stable flag\n\n")

def target_data(Y, column_name):
    if column_name not in Y.columns:
        print(f"Column '{column_name}' not found in DataFrame Y.")
        return None

    return Y[column_name].values

def label_coding(column_name):
    if column_name == 'Cooler condition':
        label_mapping = ['Close to total failure', 'Reduced efficiency', 'Full efficiency']  
    elif column_name == 'Valve condition':
        label_mapping = ['Optimal switching behavior', 'Small lag', 'Severe lag', 'Close to total failure']
    elif column_name == 'Internal pump leakage':
        label_mapping = ['No leakage', 'Weak leakage', 'Severe leakage']
    elif column_name == 'Hydraulic accumulator':
        label_mapping = ['Optimal pressure', 'Slightly reduced pressure', 'Severely reduced pressure', 'Close to total failure']
    elif column_name == 'stable flag':
        label_mapping = ['Stable', 'Static']
    else:
        print(f"Column '{column_name}' not found in label mapping.")
        return None

    return label_mapping
def dataPreprocessing():
    global X_train, X_test, Y_train, Y_test
    global X, Y,pca

    input_text = entry_text.get()
    Y_array = target_data(Y,input_text)

    # Check for null values in X and Y
    if X.isnull().values.any():
        text.insert(END, "Null values found in X. Please handle missing data.\n")
    else:
        text.insert(END, "No Null values found in X. \n")


    if pd.Series(Y_array).isnull().any():
        text.insert(END, "Null values found in Y.Please handle missing data. \n")
    else:
        text.insert(END, "No Null values found in Y.\n\n")


    # Apply PCA for dimensionality reduction
    
   

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y_array, test_size=0.2, random_state=np.random.randint(0, 1000))

    text.insert(END, "Data Used For Training: " + str(X_train.shape) + "\n")
    text.insert(END, "Data Used For Testing: " + str(X_test.shape) + "\n\n") 
    
    text.insert(END, "Selected Target is:"+ str(input_text) + "\n")
    text.insert(END, "Classed in Taregt:"+ str(label_coding(input_text)) + "\n\n")


    
def performance_evaluation(model_name, y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred,average='weighted')  
    rec = recall_score(y_true, y_pred,average='weighted')  
    f1s = f1_score(y_true, y_pred,average='weighted')  
    report = classification_report(y_true, y_pred, target_names=classes)

    text.insert(END, f"{model_name} Accuracy: {accuracy}\n")
    text.insert(END, f"{model_name} Precision: {pre}\n")
    text.insert(END, f"{model_name} Recall: {rec}\n")
    text.insert(END, f"{model_name} F1-score: {f1s}\n")
    text.insert(END, f"{model_name} Classification report\n{report}\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    
def train_svm():
    global X_train, X_test, Y_train, Y_test
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    # Create an SVM model
    model = svm.SVC(kernel='linear')  
    # Train the model
    model.fit(X_train_std, Y_train)
    # Make predictions on the test set
    Y_pred = model.predict(X_test_std)
    input_text = entry_text.get()
    classes = label_coding(input_text)
    performance_evaluation("SVM Model", Y_test, Y_pred, classes)

def train_random_forest():
    global X_train, X_test, Y_train, Y_test,RFC,pca

    # Standardize the data (optional but often recommended for tree-based models)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Create a RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the RandomForest model
    RFC.fit(X_train_std, Y_train)

    # Make predictions on the test set
    Y_pred = RFC.predict(X_test_std)

    # Evaluate the model
    input_text = entry_text.get()
    classes = label_coding(input_text)
    performance_evaluation("Random Forest Model", Y_test, Y_pred, classes)


def train_extra_trees():
    global X_train, X_test, Y_train, Y_test

    # Standardize the data (optional but often recommended for tree-based models)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Create an Extra Trees classifier
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    # Train the Extra Trees model
    model.fit(X_train_std, Y_train)

    # Make predictions on the test set
    Y_pred = model.predict(X_test_std)

    # Evaluate the model
    input_text = entry_text.get()
    classes = label_coding(input_text)
    performance_evaluation("Extra Tree Classifier", Y_test, Y_pred, classes)

def prediction():
    global scaler
    global filename, dataset,RFC
    
    # Ask the user to select a file
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, f"{filename} Test Data Loaded\n\n")
    
    # Read the test dataset
    dataset = pd.read_csv(filename)
    
    # Fill missing values and scale the dataset
    dataset.fillna(0, inplace=True)
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    dataset_scaled = scaler.transform(dataset)

    # Make predictions on the dataset
    Y_pred = RFC.predict(dataset_scaled)
    
    # Map numeric labels to their descriptions
    input_text = entry_text.get()
    print(Y_pred)
    # Display the original rows alongside predicted outcomes
    column_name = entry_text.get()
    predicted_labels = map_labels(column_name, Y_pred)
    dataset = pd.DataFrame(dataset_scaled)


    # Display the original rows alongside predicted outcomes
    for i in range(len(dataset)):
        original_row = dataset.iloc[i, :]  # Get the original row
        predicted_outcome = predicted_labels[i]

        text.insert(END, f"Original Row {i + 1}:\n")
        text.insert(END, f"{original_row}\n")
        text.insert(END, f"Predicted Outcome: {predicted_outcome}\n\n")


def map_labels(column_name, y_pred):
    label_mapping = None

    if column_name == 'Cooler condition':
        label_mapping = {
            3: 'Close to total failure',
            20: 'Reduced efficiency',
            100: 'Full efficiency'
        }
    elif column_name == 'Valve condition':
        label_mapping = {
            100: 'Optimal switching behavior',
            90: 'Small lag',
            80: 'Severe lag',
            73: 'Close to total failure'
        }
    elif column_name == 'Internal pump leakage':
        label_mapping = {
            0: 'No leakage',
            1: 'Weak leakage',
            2: 'Severe leakage'
        }
    elif column_name == 'Hydraulic accumulator / bar':
        label_mapping = {
            130: 'Optimal pressure',
            115: 'Slightly reduced pressure',
            100: 'Severely reduced pressure',
            90: 'Close to total failure'
        }
    elif column_name == 'stable flag':
        label_mapping = {
            0: 'Stable',
            1: 'Static'
        }
    else:
        print(f"Column '{column_name}' not found in label mapping.")
        return None

    return [label_mapping.get(label, f"Label {label} not found") for label in y_pred]




font = ('times', 18, 'bold')
title = Label(main, text='Innovative Predictive Maintenance in Hydraulic Systems Using Multivariate Data', justify=LEFT)
title.config(bg='sky blue', fg='dark green')  
title.config(font=font)           
title.config(height=3, width=120)       
title.pack()

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=font1)

entry_label = Label(main, text="Target Variable:")
entry_label.place(x=20, y=150)
entry_label.config(font=font1)
entry_text = Entry(main, width=40)
entry_text.place(x=220, y=150)
entry_text.config(font=font1)


preprocessButton = Button(main, text="Data Preprocessing", command=dataPreprocessing)
preprocessButton.place(x=20, y=200)
preprocessButton.config(font=font1)

cleanButton = Button(main, text="SVM Model", command=train_svm)
cleanButton.place(x=20, y=250)
cleanButton.config(font=font1)

visualizeButton = Button(main, text="Random Forest Classifier", command=train_random_forest)
visualizeButton.place(x=20, y=300)
visualizeButton.config(font=font1)

trainDTButton = Button(main, text="Extra Tree Classifier", command=train_extra_trees)
trainDTButton.place(x=20, y=350)
trainDTButton.config(font=font1)


predictButton = Button(main, text="prediction", command=prediction)
predictButton.place(x=20, y=400)
predictButton.config(font=font1)

text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500, y=100)
text.config(font=font1)

main.config()
main.mainloop()