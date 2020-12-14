from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('DriverState.csv')

X = data.iloc[:,:-1].values
# Now let's tell the dataframe which column we want for the target/labels.
y = data['Classes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

KNN_model = KNeighborsClassifier(n_neighbors=9)

KNN_model.fit(X_train, y_train)
if __name__ == '__main__':
    KNN_prediction = KNN_model.predict(X_test)
    print(confusion_matrix(KNN_prediction, y_test))
    print(accuracy_score(KNN_prediction, y_test))
