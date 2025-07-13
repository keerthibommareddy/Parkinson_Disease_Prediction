
from praat_subset_eval import run_algorithm_comparison_praat

def run_algorithm_comparison():
    # Just extend the praat version to full feature list
    import pandas as pd
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    url = "../data/data.csv"
    features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]
    dataset = pd.read_csv(url, names=features)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=7)

    models = [
        ('LR', LogisticRegression(max_iter=1000)),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('DT', DecisionTreeClassifier()),
        ('NN', MLPClassifier(solver='lbfgs')),
        ('NB', GaussianNB()),
        ('GB', GradientBoostingClassifier(n_estimators=100))
    ]

    metrics = []
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=7)
        cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        acc = accuracy_score(y_val, predictions)
        mcc = matthews_corrcoef(y_val, predictions)
        metrics.append((name, acc, mcc))

    return metrics
