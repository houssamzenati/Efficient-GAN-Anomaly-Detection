import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve, auc

def do_roc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the ROC curve

    Args:
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the ROC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            roc_auc (float): area under the under the ROC curve
            thresholds (list): list of thresholds for the ROC
    """
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr) # compute area under the curve

    if plot: 
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('results/' + file_name + '_roc.jpg')
        plt.close()

    return roc_auc, thresholds

def do_prc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the ROC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
            pre_score (float): precision score
            rec_score (float): recall score
            F1_score (float): max F1 score according to different thresholds
    """
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    index = np.argmax(F1)
    F1_score = F1[index]
    pre_score = precision[index]
    rec_score = recall[index]
    threshold = thresholds[index]

    prc_auc = auc(recall, precision)

    if plot:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Prec=%0.4f Rec=%0.4f F1=%0.4f AUC=%0.4f' 
                            %(pre_score, rec_score, F1_score, prc_auc))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('results/' + file_name + '_prc.jpg')
        plt.close()

    return prc_auc, pre_score, rec_score, F1_score, threshold

def predict(scores, threshold):
    scores = np.array(scores)
    scores[scores>=threshold] = 1
    scores[scores<threshold] = 0 
    return scores.tolist()

def make_meshgrid(x_min,x_max,y_min,y_max, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy 
