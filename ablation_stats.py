import numpy as np
import matplotlib.pyplot as plt
from models.NNModel import NeuralNetwork
from models.NPModel import NPNeuralNetwork

def show_comparison_stats(acc_1, acc_2, lss_1, lss_2, acc_3, lss_3, acc_4, lss_4, acc_5, lss_5, label_1='Pruning/genesis', label_2='Adaptive LR', label_3='Hebbian', label_4='Full NPNN', label_5='Standard NN'):
    epochs = range(1, len(acc_1) + 1)
    acc_1_percent = [a * 100 for a in acc_1]
    acc_2_percent = [a * 100 for a in acc_2]
    acc_3_percent = [a * 100 for a in acc_3]
    acc_4_percent = [a * 100 for a in acc_4]
    acc_5_percent = [a * 100 for a in acc_5]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(epochs, acc_1_percent, label=f'{label_1} Accuracy', color='blue')
    ax1.plot(epochs, acc_2_percent, label=f'{label_2} Accuracy', color='green')
    ax1.plot(epochs, acc_3_percent, label=f'{label_3} Accuracy', color='orange')
    ax1.plot(epochs, acc_4_percent, label=f'{label_4} Accuracy', color='black')
    ax1.plot(epochs, acc_5_percent, label=f'{label_5} Accuracy', color='red')
    ax1.plot(epochs[-1], acc_1_percent[-1], 'o', color='blue')
    ax1.plot(epochs[-1], acc_2_percent[-1], 'o', color='green')
    ax1.plot(epochs[-1], acc_3_percent[-1], 'o', color='orange')
    ax1.plot(epochs[-1], acc_4_percent[-1], 'o', color='black')
    ax1.plot(epochs[-1], acc_5_percent[-1], 'o', color='red')
    ax1.annotate(f'{acc_1_percent[-1]:.2f}%', (epochs[-1], acc_1_percent[-1]),
                 textcoords="offset points", xytext=(-30,10), ha='center',
                 fontsize=8, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax1.annotate(f'{acc_2_percent[-1]:.2f}%', (epochs[-1], acc_2_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(f'{acc_3_percent[-1]:.2f}%', (epochs[-1], acc_3_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange'))
    ax1.annotate(f'{acc_4_percent[-1]:.2f}%', (epochs[-1], acc_4_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='black',
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax1.annotate(f'{acc_5_percent[-1]:.2f}%', (epochs[-1], acc_5_percent[-1]),
                 textcoords="offset points", xytext=(30,10), ha='center',
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.set_title("Model Comparison: Accuracy")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, lss_1, label=f'{label_1} Loss', color='blue')
    ax2.plot(epochs, lss_2, label=f'{label_2} Loss', color='green')
    ax2.plot(epochs, lss_3, label=f'{label_3} Loss', color='orange')
    ax2.plot(epochs, lss_4, label=f'{label_4} Loss', color='black')
    ax2.plot(epochs, lss_5, label=f'{label_5} Loss', color='red')
    ax2.plot(epochs[-1], lss_1[-1], 'o', color='blue')
    ax2.plot(epochs[-1], lss_2[-1], 'o', color='green')
    ax2.plot(epochs[-1], lss_3[-1], 'o', color='orange')
    ax2.plot(epochs[-1], lss_4[-1], 'o', color='black')
    ax2.plot(epochs[-1], lss_5[-1], 'o', color='red')
    ax2.annotate(f'{lss_1[-1]:.5f}', (epochs[-1], lss_1[-1]),
                 textcoords="offset points", xytext=(-30,-10), ha='center',
                 fontsize=8, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax2.annotate(f'{lss_2[-1]:.5f}', (epochs[-1], lss_2[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='green',
                 arrowprops=dict(arrowstyle='->', color='green'))
    ax2.annotate(f'{lss_3[-1]:.5f}', (epochs[-1], lss_3[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='orange',
                 arrowprops=dict(arrowstyle='->', color='orange'))
    ax2.annotate(f'{lss_4[-1]:.5f}', (epochs[-1], lss_4[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='black',
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax2.annotate(f'{lss_5[-1]:.5f}', (epochs[-1], lss_5[-1]),
                 textcoords="offset points", xytext=(30,-10), ha='center',
                 fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red'))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_title("Model Comparison: Loss")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'Images/comparison_stats_ablation.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def intermediate_func(model_1, model_2, model_3, model_4, model_5):
    acc_1, lss_1 = model_1.get_stats()
    acc_2, lss_2 = model_2.get_stats()
    acc_3, lss_3 = model_3.get_stats()
    acc_4, lss_4 = model_4.get_stats()
    acc_5, lss_5 = model_5.get_stats()
    show_comparison_stats(acc_1, acc_2, lss_1, lss_2, acc_3, lss_3, acc_4, lss_4, acc_5, lss_5)

def main():
    arch = [784, 256, 128, 10]
    model_1 = NPNeuralNetwork(arch)
    model_2 = NPNeuralNetwork(arch)
    model_3 = NPNeuralNetwork(arch)
    model_4 = NPNeuralNetwork(arch)
    model_5 = NeuralNetwork(arch)

    model_1.load_model(1000)
    model_2.load_model(2000)
    model_3.load_model(3000)
    model_4.load_model(0)
    model_5.load_model(0)

    intermediate_func(model_1, model_2, model_3, model_4, model_5)

if __name__ == "__main__":
    main()
