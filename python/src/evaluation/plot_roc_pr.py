from __future__ import print_function, division
import torch_extras
import torch
from torchnet.logger import VisdomLogger
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

class Evaluation():

    def __init__(self, classes_name, labels, scores, cm,
                 roc_curve_plot, pr_curve_plot, colors, name_ext=''):
        self.classes_name = classes_name
        self.num_classes = len(classes_name)
        self.labels = labels
        self.scores = scores
        self.cm = cm
        self.roc_curve_plot = roc_curve_plot
        self.pr_curve_plot = pr_curve_plot
        self.colors = colors
        self.name_ext = name_ext

        setattr(torch, 'one_hot', torch_extras.one_hot)
        one_hot_size = self.labels.size()
        self.label_one_hot =\
            torch.one_hot((one_hot_size[0], len(self.classes_name)),
                          self.labels.cpu().view(-1,1))

    def show_confusion_matrix(self):

        if not self.cm:
            return
        confusion_logger =\
            VisdomLogger('heatmap', port=8097, opts={'title': 'Confusion matrix',
                                                    'columnnames': self.classes_name,
                                                    'rownames': self.classes_name})
        confusion_logger.log(self.cm.value())

    def plot_pr(self, precision, recall, average_precision):
        f_scores = np.linspace(0.2, 0.8, num=2)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = self.pr_curve_plot.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            self.pr_curve_plot.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        for i, color in zip(range(self.num_classes), self.colors):
            if i == 0:
                continue
            l, = self.pr_curve_plot.plot(recall[i], precision[i], color=color, lw=2,
                label='Precision-recall for class {0} (area = {1:0.2f})'.format(self.classes_name[i], average_precision[i]))
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                        ''.format(self.classes_name[i], average_precision[i]))

        self.pr_curve_plot.set_xlim([0.0, 1.0])
        self.pr_curve_plot.set_ylim([0.0, 1.05])
        self.pr_curve_plot.set_xlabel('Recall')
        self.pr_curve_plot.set_ylabel('Precision')
        self.pr_curve_plot.set_title('PR Curve '+ self.name_ext)
        self.pr_curve_plot.legend(loc='lower left')
        #prop=dict(size=14)

    def plot_roc(self, fpr, tpr, roc_auc):
        assert self.num_classes < 3, "Roc only support binary classes"
        lw = 2
        for i, color in zip(range(self.num_classes), self.colors):
            if i == 0:
                continue
            self.roc_curve_plot.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(self.classes_name[i], roc_auc[i]))
        self.roc_curve_plot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.roc_curve_plot.set_xlim([0.0, 1.0])
        self.roc_curve_plot.set_ylim([0.0, 1.05])
        self.roc_curve_plot.set_xlabel('FPR')
        self.roc_curve_plot.set_ylabel('TPR')
        self.roc_curve_plot.set_title('ROC Curve '+self.name_ext)
        self.roc_curve_plot.legend(loc="lower right")


    def plot_results(self):

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(self.num_classes):
            if self.num_classes < 3:
                fpr[i], tpr[i], _ =\
                    roc_curve(self.label_one_hot[:, i], self.scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ =\
                precision_recall_curve(self.label_one_hot[:, i],
                                       self.scores[:, i])
            average_precision[i] =\
                average_precision_score(self.label_one_hot[:, i],
                                        self.scores[:, i])
        self.plot_pr(precision, recall, average_precision)
        self.plot_roc(fpr, tpr, roc_auc)

        self.show_confusion_matrix()
