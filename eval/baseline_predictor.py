from sklearn.metrics import accuracy_score, matthews_corrcoef
import pandas as pd
from sklearn.model_selection import train_test_split

def run_benchmark():
    url = "../data/data.csv"
    features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
                "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
                "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
                "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]
    dataset = pd.read_csv(url, names=features)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.3, random_state=7)

    predictions = [1] * len(y_val)
    acc = accuracy_score(y_val, predictions)
    mcc = matthews_corrcoef(y_val, predictions)

    return [("Benchmark (All 1s)", acc, mcc)]
