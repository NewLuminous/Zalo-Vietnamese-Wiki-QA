import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

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