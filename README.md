# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MANISHA M
RegisterNumber: 212224220061 
*/
```

```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]           # keep only relevant columns
df.columns = ['label', 'message']

# Step 2: Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train SVM model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)



# Step 6: Predict and Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```


## Output:
<img width="1066" height="299" alt="image" src="https://github.com/user-attachments/assets/69c09e9e-1561-421f-bb8f-dc2d45a9a0f4" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
