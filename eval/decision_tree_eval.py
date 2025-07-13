
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

def run_rescaled_dt():
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

    model = DecisionTreeClassifier()
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    acc = accuracy_score(y_val, predictions)
    mcc = matthews_corrcoef(y_val, predictions)

    return [("DT (Scaled)", acc, mcc)]
