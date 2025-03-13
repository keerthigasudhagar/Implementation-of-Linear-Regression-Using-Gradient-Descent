# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KEERTHIKA.S
RegisterNumber: 212223040093 
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
print(X1)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

Dataset :

![420557884-47c95712-9877-42c5-80cf-fc7ded975c5a](https://github.com/user-attachments/assets/ce9788af-a7b9-4170-935b-7f0c3f893d77)



X nd Y values:

![420557950-cd7bdcb7-226d-4483-93b2-dd30efb6c495](https://github.com/user-attachments/assets/0637cc1e-6cc5-4092-8f6d-b2e1fc2aed67)



Predicted Value :

![420558041-53dca760-cebb-4b12-bf36-a46e47d8c2c1](https://github.com/user-attachments/assets/ea230468-1090-40a6-aa1e-b23041cc51a1)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
