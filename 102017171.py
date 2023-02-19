import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



data = pd.read_csv('C:\\Users\\adity\\OneDrive\\Desktop\\assign\\Data.csv')
data


X = data.drop(['Class'], axis=1)
y = data['Class']
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
balanced_data = pd.concat([X, y], axis=1)


sample_sizes = [int(len(balanced_data)*0.1)] * 5 # 

samples = []
for size in sample_sizes:
    samples.append(balanced_data.sample(n=size, replace=False, random_state=42)) 


models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), SVC(), MLPClassifier()]


for i, model in enumerate(models):
    max_accuracy = 0
    max_sampling = None
    for j, sample in enumerate(samples):
        X = sample.drop(['Class'], axis=1)
        y = sample['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test) 
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_sampling = f"Sampling{j+1}"
    print(f"Model {i+1}: Highest accuracy is {max_accuracy} with {max_sampling} sampling\n")

