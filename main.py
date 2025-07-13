# main.py
from algorithm_comparison import run_algorithm_comparison
from algorithm_comparison_praat import run_algorithm_comparison_praat
from benchmark import run_benchmark
from knn import run_knn
from rescaled_data_algorithm_comparison import run_rescaled_dt
from plots import plot_metrics

if __name__ == "__main__":
    metrics = []
    metrics += run_benchmark()
    metrics += run_algorithm_comparison_praat()
    metrics += run_algorithm_comparison()
    metrics += run_knn()
    metrics += run_rescaled_dt()

    for name, acc, mcc in metrics:
        print(f"{name:<25} | Accuracy: {acc*100:.2f}% | MCC: {mcc:.4f}")

    # Show plots
    plot_metrics(metrics)
