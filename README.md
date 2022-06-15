# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required library packages.
2. Import the dataset to operate on.
3. Split the dataset into required segments.
4. Predict the required output.
5. Run the programme.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Palamakula Deepika
RegisterNumber: 212221240035
*/
import cv2
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
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
<img width="403" alt="8 1" src="https://user-images.githubusercontent.com/94154679/173930657-fb202176-50b0-4f7a-bd44-0e67de650948.png">

<img width="238" alt="8 2" src="https://user-images.githubusercontent.com/94154679/173930683-58b83f30-63eb-4b20-90b9-1e52cc2b9a4b.png">

<img width="128" alt="8 3" src="https://user-images.githubusercontent.com/94154679/173930696-a95c24f4-a522-4f26-9782-f83557243cce.png">

<img width="361" alt="8 4" src="https://user-images.githubusercontent.com/94154679/173930721-5e954809-1e83-4570-9eb6-089c3c41c9b0.png">

<img width="243" alt="8 5" src="https://user-images.githubusercontent.com/94154679/173930737-4a83df8f-cdbf-4694-9263-528902b698fd.png">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
