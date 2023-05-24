import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.model_selection import train_test_split
from map_feature import map_feature
from sklearn.metrics import accuracy_score
import json
from utils import read_config, save_report

hyper_params_config_file = "config.json"
data_file = "training_data.txt"
model_file = "model.json"
classification_report_file = "classification_report.json"

hyper_params = read_config(hyper_params_config_file)

model = LogisticRegression(lr = hyper_params["Alpha"], lambda_val = hyper_params["Lambda"])

f = open(data_file,"r")

raw_data = f.readlines()

X = []
y = []
for i in range(len(raw_data)):
    
    line = raw_data[i].replace("\n","")
    x1, x2, label = line.split(",")

    X.append(map_feature(np.array([float(x1)]), np.array([float(x2)]))[0])
    y.append(float(label))

print("There are {} samples for training".format(len(X)))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("There are {} samples for training, {} samples for testing".format(y_train.shape[0], y_test.shape[0]))

print("START TRAINING:")

model.train(X_train, y_train, X_test, y_test, hyper_params["NumIter"])
model.save_weight(model_file)

print("Finish!")

print("Evaluation of models")
acc, precision, recall, f1_score = model.evaluate(X, y)

print("Acc on the training set:", acc)

report = {
    "Accuracy":acc,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1_score
}

print("SAVING CLASSIFICATION REPORT")
save_report(report, classification_report_file)
