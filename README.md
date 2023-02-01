# Docs for srlearn

# _____________________________________________

## Imports


```python
# Our models
from srlearn import LinearModels

# Other libraries to train, test and evaluate our models
from sklearn import datasets
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, roc_curve
import matplotlib.pyplot as plt
```

# _____________________________________________

## Linear Regression


```python
# Loading dataset

d = datasets.load_diabetes()
x = d['data']
y = d['target']
```


```python
# Splitting dataset

x_train = x[:400]
y_train = y[:400]
x_test = x[400:]
y_test = y[400:]
```


```python
# Initializing linear regression model
model = LinearModels.LinearRegression()

# Fitting the model on our data
model.fit(x_train, y_train)

# Predicting test data with our trained model
y_pred = model.predict(x_test)
```


```python
# Visualizing the model predictions

plt.scatter([i for i in range(len(y_test))], y_test, label="acutual values")
plt.plot(y_pred, c='red', label="prediction")
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a45c359610>




    
![png](output_9_1.png)
    



```python
# r2 score of our trained model
r2_score(y_test, y_pred)
```




    0.7088916940378884



# _____________________________________________

## Logistic Regression


```python
# Loading dataset

d = datasets.load_breast_cancer()
x = d['data']
y = d['target']
```


```python
# Splitting dataset

x_train = x[:400]
y_train = y[:400]
x_test = x[400:]
y_test = y[400:]
```


```python
# Initializing linear regression model
model2 = LinearModels.LogisticRegression()

# Fitting the model on our data
model2.fit(x_train, y_train)

# Predicting test data with our trained model
y_pred = model2.predict(x_test)
```


```python
# Visualizing the model predictions

plt.scatter([i for i in range(len(y_test))], y_test, label="False prediction")
plt.scatter([i for i in range(len(y_test))], y_pred, c="red", label="True prediction")
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a45c3a3110>




    
![png](output_16_1.png)
    



```python
# Confusion metrics of model predictions
confusion_matrix(y_test, y_pred)
```




    array([[ 17,  22],
           [  0, 130]], dtype=int64)




```python
# accuracy score of model predictions

accuracy_score(y_test, y_pred)
```




    0.8698224852071006




```python
# ROC curve of our model

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='roc curve')
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a45e406010>




    
![png](output_19_1.png)
    



```python
# ROC parameters of our model

plt.plot(roc_curve(y_test, y_pred), label=["false positive rate", "true positive rate", "thresholds"])
plt.legend()
```




    <matplotlib.legend.Legend at 0x2a45f67e010>




    
![png](output_20_1.png)
    


# _____________________________________________


```python

```
