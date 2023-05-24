import numpy as np
import json 
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

class LogisticRegression():
    def __init__(self, lr = 1e-2, lambda_val = 5e-4) -> None:
        self.lr = lr
        self.lambda_val = lambda_val
        self._theta = None

    def _initialize_weight(self, weight_len):

        self._theta = np.zeros(weight_len)

    def _compute_cost(self, X, y):

        m = y.shape[0]
        
        h_theta = sigmoid(np.dot(self._theta, np.transpose(X)))

        loss = (1.0/m)*(np.dot(np.transpose(-y), np.log(h_theta)) - np.dot((1.0 - np.transpose(y)), np.log(1 - h_theta)))

        return loss 

    def _compute_gradient(self, X, y):

        m = len(y) 

        h_theta = sigmoid(np.dot(self._theta, np.transpose(X)))

        dJ = (1.0/m)*np.dot((h_theta - y), X) + (self.lambda_val/m)* self._theta

        return dJ

    def load_weights(self, model_file):
        
        f = open(model_file,"r")
        thetas = json.load(f)
        self._theta = np.array(list(thetas.values()))
        # print(self._theta)
        
    def gradient_descent(self, X, y):

        dJ = self._compute_gradient(X, y)
        # print(dJ)
        self._theta = self._theta - self.lr * dJ

        loss = self._compute_cost(X, y)

        return loss 
    
    def train(self, X, y, X_val, y_val, num_iters):

        number_of_feature = X.shape[-1]

        self._initialize_weight(number_of_feature)
        print(self._theta)
        print(number_of_feature, self._theta)
        total_loss = 0

        for i in range(num_iters):

            loss = self.gradient_descent(X, y)
            # print(self._theta)
            # print("Loss at step {}: {}".format(i, loss))
            total_loss+=loss

            if i % 50 == 0:

                acc, precision, recall, f1_score = self.evaluate(X_val, y_val)
                val_loss = self._compute_cost(X_val, y_val)

                print("Iter: {}, Train Loss: {}, Val Loss: {}, Val_Accuracy: {}, Val Precision: {}, Val Recall: {}, Val F1-score: {}".format(i, loss, val_loss, acc, precision, recall, f1_score))

        return total_loss, total_loss/len(y)  

    def predict(self, X):

        pred = sigmoid(np.dot(self._theta, np.transpose(X)))

        return pred

    def evaluate(self, X, y, threshold = 0.5):

        y_pred = self.predict(X)

        y_pred = np.where(y_pred > threshold, 1, 0)

        # y_pred = np.round(y_pred)

        acc = accuracy_score(y, y_pred)

        precision = precision_score(y, y_pred)

        recall = recall_score(y, y_pred)

        f1 = f1_score(y, y_pred)
        # print("Accuracy", acc)
        return acc, precision, recall, f1

    def save_weight(self, model_path):

        weights = dict()
        weights_list =  list(self._theta)
        
        m = len(weights_list)
        # print(self._theta)
        for i in range(m):

            weights["theta_{}".format(i)] = weights_list[i]

        json_object = json.dumps(weights)

        with open(model_path,"w") as outfile:
            outfile.write(json_object)

        # self._theta = self._initialize_weight(number_of_features)

if "__name__" == "__main__":
    X = [[1,2],
        [3,4],
        [5,1],
        [2,0]]
    y = [0,1,0,1]

    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    model = LogisticRegression(lr = 1e-2, lambda_val = 5e-4)
    model.train(X, y, 5)

