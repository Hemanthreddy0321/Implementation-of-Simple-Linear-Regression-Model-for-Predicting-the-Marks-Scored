# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
'''
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas. 
'''
## Program:
```
/*
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HEMANTH A
RegisterNumber: 212223220025
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
![WhatsApp Image 2024-08-30 at 14 05 38_a3c3a123](https://github.com/user-attachments/assets/4bc79405-6009-4c75-bbf0-d1073e1e1a72)

![WhatsApp Image 2024-08-30 at 14 05 39_1dc59e3e](https://github.com/user-attachments/assets/15c7542f-4e4e-4c10-893f-a8feb59d7030)

![WhatsApp Image 2024-08-30 at 14 05 40_37bd9159](https://github.com/user-attachments/assets/044bae5a-7dd8-4e0b-aea9-3e8481090f92)

![WhatsApp Image 2024-08-30 at 14 05 56_a4296d6d](https://github.com/user-attachments/assets/c848ee02-4b25-4128-a811-2f80d7d66c26)

![WhatsApp Image 2024-08-30 at 14 06 02_6e529ca7](https://github.com/user-attachments/assets/fbf53b66-fc95-45b0-86ff-6d2d608760b1)

![WhatsApp Image 2024-08-30 at 14 06 10_44593a5c](https://github.com/user-attachments/assets/639d5229-6fe0-4bfc-96b4-24d1a9a108c9)

![WhatsApp Image 2024-08-30 at 14 06 12_73ee24d7](https://github.com/user-attachments/assets/e4b0ed8d-1c73-4250-bf97-51a1f0fee022)
















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
