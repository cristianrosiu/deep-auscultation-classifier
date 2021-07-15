import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_confusion_matrix(cm, classes, save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = float('%.2f' % (cm))
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()


def plot_roc(fpr, tpr, auc, save_path):
    """
    This function plots the ROC-AUC Curve
    """
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.show()


def plot_history(history, save_path):
    """
    Plots the training and validation history of a model.
    """
    fig = plt.figure(constrained_layout=False, figsize=(8, 4))

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.plot(history.history["whistling_accuracy"], label='Whistling Train')
    ax1.plot(history.history["val_whistling_accuracy"], label='Whistling Test')

    ax2.plot(history.history["rhonchus_accuracy"], label='Rhonchus Train')
    ax2.plot(history.history["val_rhonchus_accuracy"], label='Rhonchus Test')

    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.set_title("CNN Accuracy")

    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    ax2.set_title("CNN Accuracy")

    ax3.plot(history.history["loss"], label='Train Error')
    ax3.plot(history.history["val_loss"], label='Test Error')
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Epoch")
    ax3.legend(loc="upper right")
    ax3.set_title("CNN Error")

    plt.savefig(save_path)
    plt.show()