import itertools
import matplotlib.pyplot as plt
import numpy as np

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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
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


def plot_history(history, i, save_path):
    fig, axs = plt.subplots(2)

    # Accuracy subplot
    axs[0].plot(history.history[0]["accuracy"], label='Train Accuracy')
    axs[0].plot(history.history[0]["val_accuracy"], label='Test Accuracy')
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("CNN Accuracy {i}".format(i=i))

    # Error subplot
    axs[1].plot(history.history[0]["loss"], label='Train Error')
    axs[1].plot(history.history[0]["val_loss"], label='Test Error')
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("CNN Error")

    plt.savefig(save_path)
    plt.show()


def plot_kfold(model_history, save_path, label='Whistling'):
    plt.title('Accuracies vs Epochs')
    plt.plot(model_history[0].history['val_{label}_accuracy'.format(label=label.lower())],
             label='{label} Training Fold 1'.format(label=label))
    plt.plot(model_history[1].history['val_{label}_accuracy'.format(label=label.lower())],
             label='{label} Training Fold 2'.format(label=label))
    plt.plot(model_history[2].history['val_{label}_accuracy'.format(label=label.lower())],
             label='{label} Training Fold 3'.format(label=label))
    plt.plot(model_history[3].history['val_{label}_accuracy'.format(label=label.lower())],
             label='{label} Training Fold 4'.format(label=label))
    plt.plot(model_history[4].history['val_{label}_accuracy'.format(label=label.lower())],
             label='{label} Training Fold 5'.format(label=label))
    plt.legend()
    plt.savefig(save_path)
    plt.show()