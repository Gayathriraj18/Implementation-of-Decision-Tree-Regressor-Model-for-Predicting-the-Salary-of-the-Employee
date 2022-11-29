# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.


## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Gayathri A
RegisterNumber:  212221230028
*/
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```
## Output:

![71](https://user-images.githubusercontent.com/94154854/204559534-b03b97b8-b018-48c0-9df2-c4dce19a077b.png)

![72](https://user-images.githubusercontent.com/94154854/204559558-d65faee9-7a56-4b73-ad15-192aa8ee1dd8.png)

![73](https://user-images.githubusercontent.com/94154854/204559649-e4f95f49-a247-4fda-9f5f-7a7143964e1a.png)

![74](https://user-images.githubusercontent.com/94154854/204559676-a394022a-ed8e-467a-b8e0-3d5c2743ae27.png)

![75](https://user-images.githubusercontent.com/94154854/204559712-9ab224bc-b626-4d4a-94db-8c8b1a5656a8.png)

![76](https://user-images.githubusercontent.com/94154854/204559748-24cda613-7cad-4af1-9419-5a462e7542d7.png)

![77](https://user-images.githubusercontent.com/94154854/204559786-9f355bfd-e6c9-458a-8f00-96bea5d28c1a.png)

![78](https://user-images.githubusercontent.com/94154854/204559825-6d194dac-8cec-467d-a07c-fb3f83cf67e7.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
