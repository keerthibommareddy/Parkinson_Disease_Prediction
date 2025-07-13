# plots.py
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    names = [m[0] for m in metrics]
    acc = [m[1] * 100 for m in metrics]
    mcc = [m[2] for m in metrics]

    plt.figure(figsize=(10,5))
    plt.bar(names, acc, color='skyblue')
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    plt.bar(names, mcc, color='salmon')
    plt.ylabel("Matthews Corr. Coef")
    plt.title("Model MCC Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
