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

### Note :wink:

Let me know if are there any issues :wink: