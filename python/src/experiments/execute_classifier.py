import sys
from src.utils.cfg import yfile_to_cfg
from src.tools.finetune_factory import TrainFactory
from src.tools.data_split import DataSplit
from src.evaluation.plot_roc_pr import Evaluation
import matplotlib.pyplot as plt
plt.ion()   # interactive mode

def jap_vs_all(cfg, classifier, roc_curve_plot, pr_curve_plot):
    jap_data_dir = './dataset/data_train_test_splited/one_vs_all_cropped/japanese_knotweed_vs_all'
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    classifier.datasplit = DataSplit(cfg, data_dir=jap_data_dir)
    #best_model_path = './dataset/data_train_test_splited/one_vs_all_cropped/japanese_knotweed_vs_all/trained_models/resnet18_model_best.pth.tar'
    best_model_path = classifier.finetune_model_fun()
    labels, scores, cm = classifier.test_model(best_model_path)
    evaluation = Evaluation(classifier.datasplit.classes_name,
                            labels, scores, cm,
                            roc_curve_plot,
                            pr_curve_plot,
                            colors, 'Test Split')
    evaluation.plot_results()


def phrag_vs_all(cfg, classifier, roc_curve_plot, pr_curve_plot):
    phrag_vs_all = './dataset/data_train_test_splited/one_vs_all_cropped/common_reed_vs_all'
    colors = ['red', 'green', 'cornflowerblue']
    classifier.datasplit = DataSplit(cfg, data_dir=phrag_vs_all)
    best_model_path = classifier.finetune_model_fun()
    #best_model_path = './dataset/data_train_test_splited/one_vs_all_cropped/common_reed_vs_all/trained_models/resnet18_model_best.pth.tar'
    labels, scores, cm = classifier.test_model(best_model_path)
    evaluation = Evaluation(classifier.datasplit.classes_name,
                            labels, scores, cm,
                            roc_curve_plot,
                            pr_curve_plot,
                            colors, 'Test Split')
    evaluation.plot_results()

def implementation_test(cfg, classifier, roc_curve_plot, pr_curve_plot):
    data_dir = './hymenoptera_data'
    classifier.datasplit = DataSplit(cfg, data_dir=data_dir)
    best_model_path = classifier.finetune_model_fun()
    labels, scores, cm = classifier.test_model(best_model_path)
    evaluation = Evaluation(classifier.datasplit.classes_name,
                            labels, scores, cm,
                            roc_curve_plot,
                            pr_curve_plot,
                            colors, 'Test Split')
    evaluation.plot_results()



def main():
    assert len(sys.argv) > 1, "cfg file path missing"
    cfg = yfile_to_cfg(sys.argv[1])
    network_type = cfg.NETWORK.TYPE
    curve = plt.figure()
    roc_curve_plot= curve.add_subplot(121)
    pr_curve_plot= curve.add_subplot(122)
    classifier = TrainFactory.get_trainer(network_type, cfg)

    import pdb
    pdb.set_trace()
    phrag_vs_all(cfg, classifier, roc_curve_plot, pr_curve_plot)
    jap_vs_all(cfg, classifier, roc_curve_plot, pr_curve_plot)
    #implementation_test(cfg, classifier, roc_curve_plot, pr_curve_plot)
    input()
    input()


if __name__ == '__main__':
    main()
