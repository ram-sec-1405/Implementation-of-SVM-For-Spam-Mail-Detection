# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output
   
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAMPRASATH R
RegisterNumber:  212223220086
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

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
## Result output:

![282243022-28a5795a-2580-433a-9443-e2f07c687b5e](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/aad50e36-cc59-4c8d-97ee-43b4e6d78754)

## data.head():

![282243035-5cdf8c27-cb0a-43b9-86c0-a2db5671c78d](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/3fca5130-2f69-4cf5-9a51-df7ff9ba4766)

## data.info():

![282243043-6c9e9c39-9def-41c3-993e-73ef4e37ac30](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/415311e1-20d5-4702-bd59-a11a5b3cdeb9)

## data.isnull().sum():

![282243050-8cc08474-d436-4d1f-b9fd-300e03f40aca](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/4ee57de7-de51-4c29-b063-29ef4f8e1b3f)

## Y_prediction value:

![282243051-4ae9ec2d-9001-432f-8bb9-9ae3de9e2311](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/3e8f3cb9-023e-48ba-80c3-fd54a54982c4)

## Accuracy value:

![282243052-7d5ffca2-ba4e-4690-b5d1-524d12659f1c](https://github.com/Nithish23013509/Implementation-of-SVM-For-Spam-Mail-Detection/assets/149038138/1c70231b-3edc-4b5f-a97a-18c0beab29db)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
