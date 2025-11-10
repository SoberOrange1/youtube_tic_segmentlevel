import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, results_path):
    """
    Plots the confusion matrix.
    :param y_true: Ground truth labels (integer format).
    :param y_pred: Predicted labels (integer format).
    :param class_names: List of class names corresponding to labels.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Create a ConfusionMatrixDisplay for better visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Configure the plot
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')  # Blues color map for clarity
    plt.title("Confusion Matrix")  # Add a title
    plt.savefig(results_path, bbox_inches='tight')
    print(f"Confusion matrix saved to: {results_path}")

# Function to plot ROC curves for multiclass classification
def plot_roc_curve(y_true, y_scores, num_classes, results_path, class_names=None):
    """
    Plots ROC curves for multiclass classification.
    :param y_true: Ground truth labels (one-hot encoded).
    :param y_scores: Predicted probabilities for each class.
    :param num_classes: Total number of classes.
    :param results_path: Path to save the ROC curve image.
    :param class_names: Optional list of class names to use in the legend.
    """

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        label_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(results_path, bbox_inches='tight')
    print(f"ROC curve saved to: {results_path}")

def compute_metrics(y_true, y_pred, num_classes=7):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro = report['macro avg']
    per_class = {f'class_{i}': report[str(i)] for i in range(num_classes)}
    return {
        'accuracy': report['accuracy'],
        'precision': macro['precision'],
        'recall': macro['recall'],
        'f1': macro['f1-score'],
        'per_class': per_class
    }
