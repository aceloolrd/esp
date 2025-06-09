import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metric(metric_history, ylabel, filename):
    plt.figure(figsize=(6, 4))

    sns.lineplot(x=np.arange(len(metric_history)), y=metric_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over epochs")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()