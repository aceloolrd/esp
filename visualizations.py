import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch, Circle

def visualize_network(network, filename):
    """
    Визуализация структуры рекуррентной сети с отображением рекуррентных связей
    """
    # ---- 1. Собираем все веса ----
    all_w = np.concatenate([
        network.weights_input_hidden.flatten(),
        network.weights_hidden_hidden.flatten(),  # Добавляем рекуррентные веса
        network.weights_hidden_output.flatten()
    ])
    
    if all_w.size == 0:
        return

    # Робастный «максимум» 
    robust_max = np.percentile(np.abs(all_w), 95)  # 95-й перцентиль
    w_max = max(robust_max, 1e-8)                  # защита от нуля

    # ---- 2. Расставляем узлы ----
    in_x, in_y  = [0.0] * network.input_size,  np.linspace(0.1, 0.9, network.input_size)
    hid_x, hid_y = [0.5] * network.hidden_size, np.linspace(0.05, 0.95, network.hidden_size)
    out_x, out_y = [1.0] * network.output_size, np.linspace(0.3, 0.7, network.output_size)

    cmap = cm.get_cmap('coolwarm')
    plt.figure(figsize=(12, 8))

    # ---- 3. Связи input → hidden ----
    for i, (x0, y0) in enumerate(zip(in_x, in_y)):
        for j, (x1, y1) in enumerate(zip(hid_x, hid_y)):
            w = network.weights_input_hidden[j, i]
            w_clip = np.clip(w, -w_max, w_max)           # клипуем выбросы
            norm_signed = (w_clip + w_max) / (2 * w_max)
            norm_abs = abs(w_clip) / w_max
            color = cmap(norm_signed)
            lw = 0.5 + 4.5 * norm_abs
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.8, linewidth=lw)

    # ---- 4. Рекуррентные связи hidden → hidden ----
    # Используем дуги для визуального разделения
    for i, (x0, y0) in enumerate(zip(hid_x, hid_y)):
        for j, (x1, y1) in enumerate(zip(hid_x, hid_y)):
            if i == j:  # Петли - рисуем круги
                w = network.weights_hidden_hidden[i, j]
                if abs(w) > w_max * 0.05:  # Только значимые связи
                    w_clip = np.clip(w, -w_max, w_max)
                    norm_signed = (w_clip + w_max) / (2 * w_max)
                    norm_abs = abs(w_clip) / w_max
                    color = cmap(norm_signed)
                    lw = 0.5 + 4.5 * norm_abs
                    
                    # Рисуем петлю
                    circle = Circle((x0, y0), 0.03, 
                                   edgecolor=color, 
                                   fill=False, 
                                   linewidth=lw,
                                   alpha=0.8)
                    plt.gca().add_patch(circle)
            else:  # Связи между разными нейронами
                w = network.weights_hidden_hidden[j, i]
                if abs(w) > w_max * 0.05:  # Только значимые связи
                    w_clip = np.clip(w, -w_max, w_max)
                    norm_signed = (w_clip + w_max) / (2 * w_max)
                    norm_abs = abs(w_clip) / w_max
                    color = cmap(norm_signed)
                    lw = 0.5 + 4.5 * norm_abs
                    
                    # Рисуем дугу
                    con = ConnectionPatch(
                        xyA=(x0, y0), 
                        xyB=(x1, y1),
                        coordsA="data", 
                        coordsB="data",
                        arrowstyle="-",
                        shrinkA=5,
                        shrinkB=5,
                        mutation_scale=20,
                        color=color,
                        linewidth=lw,
                        alpha=0.8,
                        connectionstyle=f"arc3,rad={0.1 * (j-i)}"  # Изгиб дуги
                    )
                    plt.gca().add_patch(con)

    # ---- 5. Связи hidden → output ----
    for i, (x0, y0) in enumerate(zip(hid_x, hid_y)):
        for j, (x1, y1) in enumerate(zip(out_x, out_y)):
            w = network.weights_hidden_output[j, i]
            w_clip = np.clip(w, -w_max, w_max)
            norm_signed = (w_clip + w_max) / (2 * w_max)
            norm_abs = abs(w_clip) / w_max
            color = cmap(norm_signed)
            lw = 0.5 + 4.5 * norm_abs
            plt.plot([x0, x1], [y0, y1], color=color, alpha=0.8, linewidth=lw)

    # ---- 6. Узлы ----
    plt.scatter(in_x,  in_y,  s=200, color='blue',   edgecolors='black', zorder=3)
    plt.scatter(hid_x, hid_y, s=200, color='orange', edgecolors='black', zorder=3)
    plt.scatter(out_x, out_y, s=200, color='green',  edgecolors='black', zorder=3)

    # Подписи нейронов
    for i in range(network.input_size):
        plt.text(in_x[i] - 0.03,  in_y[i],  str(i), fontsize=9, ha='right',  va='center', color='white')
    for i in range(network.hidden_size):
        plt.text(hid_x[i], hid_y[i], str(i), fontsize=9, ha='center', va='center', color='black')
    for i in range(network.output_size):
        plt.text(out_x[i] + 0.03, out_y[i], str(i), fontsize=9, ha='left',   va='center', color='black')

    plt.axis('off')
    plt.title('Recurrent Network Structure (95-percentile scaling)')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_metric(metric_history, ylabel, filename):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=np.arange(len(metric_history)), y=metric_history, marker="o"))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over epochs")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()