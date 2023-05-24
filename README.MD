## Logistic Regression Implementation

This is the Logistic Regression demo implement from Scratcth :smile: .

### Installation :laughing:

To install packages, install from ```requirements.txt``` file via this command:
```
pip install -r requirements.txt
```

### Get modules :space_invader:
You can get Logistic Regression class from the ```LogisticRegresion.py``` file, for example:
```python
from LogisticRegression import LogisticRegression

X_train = ...
y_train = ...

X_val = ...
y_val = ...

num_iters = 1000

model = LogisticRegression(lr = 1e-2, lambda_val = 1e-3)

model.train(X_train, y_train, X_test, y_test, num_iters)

acc, precision, recall, f1_score = model.evaluate(X_test, y_test)

print(acc, precision, recall, f1_score)

```

All of the examples you can find in the ```train.py``` file.

You can get the ```sigmoid(z) ``` from the file above.

### Data sample

Data sample can be found at the ```training_data.txt``` file

### Model saving

Model can be saved at the ```model.json``` file, which includes the weight of the model, you can use load_weight attribute from the model to load the model json file. 

```python
from LogisticRegression import LogisticRegression
model = LogisticRegression(lr = 1e-2, lambda_val = 1e-3)
model.load_weights("model.json")
model.evaluate()
```

### Prediction

For prediction, you can use ```predict()``` attribute for predict the data (note: the data should be as same as the input format for the training).


```python
from LogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(lr = 1e-2, lambda_val = 1e-3)
model.load_weights("model.json")
y_pred = model.predict(X)

acc = accuracy_score(y_true, y_pred) 
print(acc)
```

### Config

The ```config.json``` file includes the hyperparameters for training, you can customize by yourself in the file

### Note :wink:

Let me know if are there any issues :wink:
