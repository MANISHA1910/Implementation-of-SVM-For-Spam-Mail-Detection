# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MANISHA M
RegisterNumber: 212224220061 
*/
```
```


import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```


## Output:

<img width="723" height="209" alt="509447532-e6faca6a-88a8-4a1b-a924-4c53ebbcad8f" src="https://github.com/user-attachments/assets/b446b2b9-9e84-4234-842f-ec7ea186884d" />


<img width="845" height="41" alt="509447496-55af6eaf-d900-4516-90a2-94887dace211" src="https://github.com/user-attachments/assets/038be2ee-fb12-4051-912b-8cffa263949d" />



<img width="317" height="146" alt="509447668-6ecf051c-697d-417b-a33b-c71e0f1652ed" src="https://github.com/user-attachments/assets/ff2e8b50-7f6c-47b8-88a1-e5ca4e13fa06" />



<img width="879" height="94" alt="509447722-209e3cbe-3762-474f-9d2a-45bd8f9b1c30" src="https://github.com/user-attachments/assets/723cefb1-9e4e-4e96-bd3f-ee4e8d84ae51" />

<img width="420" height="272" alt="509447622-d31d90b9-e21e-47f9-b0ca-6dc24fd154b8" src="https://github.com/user-attachments/assets/0c1972c6-cf11-4f7b-9766-5634b5318b2e" />

<img width="285" height="42" alt="509447784-f8ddaacd-6aff-4cb7-bed1-37c13f6ef762" src="https://github.com/user-attachments/assets/dea8dbcc-3a4a-44a2-9c11-814a2a440402" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
