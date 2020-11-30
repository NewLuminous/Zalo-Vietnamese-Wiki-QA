import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, classification_report

def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)
    
def plot_accuracy(y_test, y_pred_dict, title='Accuracy', show_values=False):
    acc_score_dict = {}
    for key in y_pred_dict.keys():
        acc_score_dict[key] = accuracy_score(y_test, y_pred_dict[key])
    
    acc_score_series = pd.Series(acc_score_dict)
    
    if show_values:
        _, ax = plt.subplots()
        for i, v in enumerate(acc_score_series):
            ax.text(i - 0.1, v, '%0.4f' % v, color='black', fontweight='bold')
    plt.title(title)
    acc_score_series.plot(kind='bar')
    plt.xticks(rotation=0)
    
def get_f1_score(y_test, y_pred, average='macro'):
    return f1_score(y_test, y_pred, average=average)
    
def plot_f1_score(y_test, y_pred_dict, average='macro', title=None, show_values=False):
    f1_score_dict = {}
    for key in y_pred_dict.keys():
        f1_score_dict[key] = f1_score(y_test, y_pred_dict[key], average=average)
    
    f1_score_series = pd.Series(f1_score_dict)
    
    if show_values:
        _, ax = plt.subplots()
        for i, v in enumerate(f1_score_series):
            ax.text(i - 0.1, v, '%0.4f' % v, color='black', fontweight='bold')
    if title is None:
        plt.title(f'{average}-F1 Score')
    else:
        plt.title(title)
    f1_score_series.plot(kind='bar')
    plt.xticks(rotation=0)
    
def get_mcc(y_test, y_pred):
    return matthews_corrcoef(y_test, y_pred)
    
def plot_mcc(y_test, y_pred_dict, title='Matthews Correlation Coefficient', show_values=False):
    mcc_score_dict = {}
    for key in y_pred_dict.keys():
        mcc_score_dict[key] = matthews_corrcoef(y_test, y_pred_dict[key])
    
    mcc_score_series = pd.Series(mcc_score_dict)
    
    if show_values:
        _, ax = plt.subplots()
        for i, v in enumerate(mcc_score_series):
            ax.text(i - 0.1, v, '%0.4f' % v, color='black', fontweight='bold')
    plt.title(title)
    mcc_score_series.plot(kind='bar')
    plt.xticks(rotation=0)
    
def get_auc_score(y_test, y_pred):
    return roc_auc_score(y_test, y_pred)
    
def plot_roc_curve(y_test, y_pred_dict, title='ROC curve'):
    plt.figure(figsize=(10, 10))
    for key in y_pred_dict.keys():
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_dict[key])
        score = roc_auc_score(y_test, y_pred_dict[key])
        plt.plot(fpr, tpr, label='%s (AUC = %0.4f)' % (key, score))

    plt.plot(
            [0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='Luck',
            alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.rcParams['font.size'] = 12
    plt.title(title)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right", fontsize=16)
    plt.tight_layout()
    plt.grid(True)

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(data=cm,
                        columns=['Predict Negative', 'Predict Positive'],
                        index=['Actual Negative', 'Actual Positive'])

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    
def get_classification_report(y_test, y_pred):
    return classification_report(y_test, y_pred)

def print_classification_report(y_test, y_pred):
    print('Classification report:\n', classification_report(y_test, y_pred))