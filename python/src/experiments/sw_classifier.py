import sys
import matplotlib.pyplot as plt
from src.tools.finetune_resnet import TrainResent
#from src.tools.data_split import DataSplit
from src.evaluation.plot_roc_pr import Evaluation
from src.utils.cfg import yfile_to_cfg
from src.tools.data_split import DataSplit

import os
import copy
import torch
plt.ion()   # interactive mode

class SW_Classifier():
    def __init__(self, cfg, cropsubset_data_dir,
                    crop_data_dir, data_dir):
        self.sw = cfg.SW.SWF
        self.window_avg = cfg.SW.WINDOW_AVG
        #self.write_hard_negitives = cfg.SW.WRITE_HARD_NEG
        self.crop_split = cfg.SW.CROP_SPLIT
        self.__init_plots__()
        self.cropsubset_datasplit = DataSplit(cfg, data_dir=cropsubset_data_dir)
        self.crop_datasplit = DataSplit(cfg, data_dir=crop_data_dir)
        self.datasplit = DataSplit(cfg, data_dir=data_dir)
        self.train_resnet = TrainResent(cfg, self.cropsubset_datasplit,
                                        data_dir=data_dir)
        self.__init_plots__()

    def __init_plots__(self):

        curve = plt.figure()
        self.roc_curve_plot_subset = curve.add_subplot(321)
        self.pr_curve_plot_subset = curve.add_subplot(322)
        self.roc_curve_plot_crop = curve.add_subplot(323)
        self.pr_curve_plot_crop = curve.add_subplot(324)
        self.roc_curve_plot_sw = curve.add_subplot(325)
        self.pr_curve_plot_sw = curve.add_subplot(326)


    def bin_crops_results(self, crops_score, crops_label):
        image_crop_score_bins = {}
        crop_file_names = {}
        #for score, label, file_name_tup in zip(crops_score,
        #crops_label, image_datasets['test'].imgs):
        if self.crop_split == 'train':
            image_name_label_tups =\
                self.crop_datasplit.image_datasets[self.crop_split].dataset.imgs
        else:
            image_name_label_tups =\
                self.crop_datasplit.image_datasets[self.crop_split].imgs

        for score, label, file_name_tup in \
                zip(crops_score, crops_label,
                    image_name_label_tups):
            file_dir, crop_file_name = os.path.split(file_name_tup[0])
            crop_file_name, ext = os.path.splitext(crop_file_name)
            file_name = crop_file_name
            if self.sw:
                crop_file_name_parts = crop_file_name.split('_')
                file_name = '_'.join(crop_file_name_parts[:-4])
                #add extra logic here
                if file_name not in crop_file_names:
                    crop_file_names[file_name] = []
                crop_file_names[file_name].append(crop_file_name)
            if file_name not in image_crop_score_bins:
                image_crop_score_bins[file_name] = []# {}
                #if label not in image_crop_score_bins[file_name]:
                image_crop_score_bins[file_name] =\
                    copy.deepcopy(score.unsqueeze(0))
            else:
                image_crop_score_bins[file_name] =\
                    torch.cat((image_crop_score_bins[file_name],
                               score.unsqueeze(0)), 0)
        return image_crop_score_bins, crop_file_names

    def define_image_level_score(self, image_crop_score_bins, crop_file_names):
        image_level_score = None# torch.zeros(len(image_datasets['test'].imgs), 2).cuda()
        image_level_label = None #torch.LongTensor(len(image_datasets['test'].imgs)).cuda()
        if self.crop_split == 'train':
            image_name_label_tups =\
               self.datasplit.image_datasets[self.crop_split].dataset.imgs
        else:
            image_name_label_tups =\
                self.crop_datasplit.image_datasets[self.crop_split].imgs
        for idx, file_name_tup in enumerate(image_name_label_tups):
            file_dir, file_name = os.path.split(file_name_tup[0])
            file_name, ext = os.path.splitext(file_name)
            if file_name in image_crop_score_bins:
                score = image_crop_score_bins[file_name]
                val_s, idx_s = score[:,1].sort(descending=True)
                score_com = score[idx_s[0:self.window_avg]].sum(0).unsqueeze(0)/self.window_avg
                if file_name_tup[1] == 0 and self.crop_split == 'train':
                    for i in idx_s[0:10]:
                        self.write_hard_negitives(crop_file_names[file_name][i],
                                             self.datasplit.class_names[1])
                    #print(crop_file_names[file_name][idx_s[0]])
                #_, arg_idx = score[:, 1].max(0)
                if image_level_score is None:
                    image_level_score = score_com
                    image_level_label = torch.LongTensor([file_name_tup[1]])
                else:
                    image_level_score = torch.cat((image_level_score, score_com))
                    image_level_label = torch.cat((image_level_label, torch.LongTensor([file_name_tup[1]])))
            else:
                print("file name missing")
        return image_level_score, image_level_label

    def plot_results(self, crop_labels, crop_scores, crop_cm,
                     image_labels,  image_scores, colors):
        evaluation = Evaluation(self.crop_datasplit.classes_name,
                                crop_labels, crop_scores, crop_cm,
                                self.roc_curve_plot_crop,
                                self.pr_curve_plot_crop,
                                colors, 'Crop Test Set Subset')
        evaluation.plot_results()
        evaluation = Evaluation(self.crop_datasplit.classes_name,
                                image_labels, image_scores, None,
                                self.roc_curve_plot_sw,
                                self.pr_curve_plot_sw,
                                colors, 'Whole Image')
        evaluation.plot_results()

    def one_vs_all_sw(self, best_model_path, colors):

        crop_labels, crop_scores, crop_cm =\
            self.train_resnet.test_model(best_model_path, self.crop_datasplit)

        image_crop_score_bins, crop_file_names =\
            self.bin_crops_results(crop_scores, crop_labels)

        image_scores, image_labels =\
            self.define_image_level_score(image_crop_score_bins, crop_file_names)

        self.plot_results(crop_labels, crop_scores, crop_cm,
                          image_labels,  image_scores, colors)

    def one_vs_all_train(self, colors):

        #datasplit = DataSplit(cfg, data_dir=data_dir)
        #train_resnet = TrainResent(cfg, datasplit, data_dir=data_dir)
        best_model_path = self.train_resnet.finetune_model_fun()
        labels, scores, cm = self.train_resnet.test_model(best_model_path)
        evaluation = Evaluation(self.cropsubset_datasplit.classes_name,
                                labels, scores, cm,
                                self.roc_curve_plot_subset,
                                self.pr_curve_plot_subset,
                                colors, 'Crop Test Set Subset')
        evaluation.plot_results()
        return best_model_path

    def write_hard_negitives(self, name, file_suf):
        with open('hard-neg'+file_suf, "a") as fp:
            fp.write(name+'\n')

def main():
    assert len(sys.argv) > 1, "cfg file path missing"
    cfg = yfile_to_cfg(sys.argv[1])
    '''
    split = 'test'
    curve = plt.figure()
    roc_curve_plot_small = curve.add_subplot(321)
    pr_curve_plot_small = curve.add_subplot(322)
    roc_curve_plot = curve.add_subplot(323)
    pr_curve_plot = curve.add_subplot(324)
    sw_roc_curve_plot = curve.add_subplot(325)
    sw_pr_curve_plot = curve.add_subplot(326)
    '''

    jap_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all_hn'
    jap_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all_complete'
    jap_data_dir_undivided = './dataset/data_train_test_splited/one_vs_all/jap_vs_all'

    phrag_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/phrag_vs_all_hn'
    phrag_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/phrag_vs_all_complete'
    phrag_data_dir_undivided = './dataset/data_train_test_splited/one_vs_all/phrag_vs_all'


    data_dir = './hymenoptera_data'
    jap_train_data_dir = data_dir
    phrag_train_data_dir = data_dir
    jap_data_dir = data_dir
    phrag_data_dir = data_dir
    jap_data_dir_undivided = data_dir
    phrag_data_dir_undivided= data_dir
    #data_dir_undivided = './hymenoptera_data'''
    jap_best_model_path = './hymenoptera_datamodel_best.pth.tar'
    sw_classifier = SW_Classifier(cfg, data_dir, data_dir, data_dir)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    #jap_best_model_path = sw_classifier.one_vs_all_train(colors)
    '''
    colors = ['red', 'green', 'cornflowerblue']
    #data_dir = './dataset/data_train_test_splited/one_vs_all/jap_vs_all'
    phrag_best_model_path = one_vs_all_train(phrag_train_data_dir, roc_curve_plot_small, pr_curve_plot_small, colors)
    '''


    #jap_best_model_path = '/home/deepak/thesis/'+jap_train_data_dir+'model_best.pth.tar'
    #phrag_best_model_path = '/home/deepak/thesis/'+phrag_train_data_dir+'model_best.pth.tar'
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    sw_classifier.one_vs_all_sw(jap_best_model_path, colors)
    plt.show()
    input()
    return
    '''self.one_vs_all_sw(jap_data_dir, sw_roc_curve_plot, sw_pr_curve_plot, colors,
                jap_data_dir_undivided, jap_best_model_path,
                roc_curve_plot, pr_curve_plot, split)'''

    colors = ['red', 'green', 'cornflowerblue']
    one_vs_all_sw(phrag_data_dir, sw_roc_curve_plot, sw_pr_curve_plot, colors,
                phrag_data_dir_undivided, phrag_best_model_path,
                roc_curve_plot, pr_curve_plot, split)

if __name__ == '__main__':
    main()
