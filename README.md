# Implementation of Decision Tree Classifier Model for Predicting Employee Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset (Employee.csv) using Pandas.Check the dataset info and preprocess it by encoding the salary column with LabelEncoder.
2. Select input features (X) and output label (Y).Split the data into training and testing sets using train_test_split.
3. Initialize a DecisionTreeClassifier with entropy criterion.Train the model on the training data (xtrain, ytrain).
4. Predict on the test set (xtest) and calculate the accuracy_score.Make new predictions using custom input feature values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Prajin S
RegisterNumber: 212223230151 
*/
```
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("Employee.csv")
df.head()
df.info()
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
X=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
Y=df["left"]
X.head()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(xtrain,ytrain)
ypred=dt.predict(xtest)
ypred
acc=accuracy_score(ytest,ypred)
print(acc)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
dt.predict([[0.3,0.5,4,160,5,3,0,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/036d5b88-2911-4777-a01b-4c6e71dad318)

![image](https://github.com/user-attachments/assets/3653df52-dbbb-441c-91fa-061c5bef33be)

![image](https://github.com/user-attachments/assets/378690f0-a746-4927-ae53-19d495f6d732)

![image](https://github.com/user-attachments/assets/87d78289-510d-49e5-ae2b-b609163fd47b)

![image](https://github.com/user-attachments/assets/9c6273af-7a25-4198-a19d-1d81905a5149)

![image](https://github.com/user-attachments/assets/e709d64b-4f73-4d41-a645-ff4a066d813f)

![image](https://github.com/user-attachments/assets/3a36bbd3-ec79-468f-9a5c-f99a0045c6a0)

![image](https://github.com/user-attachments/assets/f0c328fd-030d-49fc-a7af-cea0e32cd91a)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
