
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def run_knn():
    url = "../data/data.csv"
    features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]
    dataset = pd.read_csv(url, names=features)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X = MinMaxScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=7)

    clf = KNeighborsClassifier()
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_val)

    acc = accuracy_score(y_val, predictions)
    mcc = matthews_corrcoef(y_val, predictions)
    return [("KNN (Scaled)", acc, mcc)]
