import matplotlib.pyplot as plt
from src.tools.finetune_densnet import TrainDensnet
#from src.tools.data_split import DataSplit
from torchnet.meter import ConfusionMeter
from src.evaluation.plot_roc_pr import Evaluation
from src.utils.geometric import check_overlapping
from src.tools.data_split import DataSplit

import os
import copy
import torch
import pickle
import datetime
import shutil
plt.ion()   # interactive mode

class SW_Classifier():
    def __init__(self, cfg, cropsubset_data_dir=None, crop_data_dir=None,
                 data_dir=None):
        self.cfg = cfg
        self.sw = cfg.SW.SWF
        self.window_avg = cfg.SW.WINDOW_AVG
        #self.write_hard = cfg.SW.WRITE_HARD_NEG
        self.crop_split = cfg.SW.CROP_SPLIT
        self.spatial_cc = cfg.SW.SPATIAL_CC
        self.pick_scores = cfg.SW.PICK_SCORES
        if cropsubset_data_dir:
            self.cropsubset_datasplit = DataSplit(cfg, data_dir=cropsubset_data_dir)
        if crop_data_dir:
            self.crop_datasplit = DataSplit(cfg, data_dir=crop_data_dir)
        if data_dir:
            self.datasplit = DataSplit(cfg, data_dir=data_dir)
        self.train_classifier = None
        if cropsubset_data_dir:
            self.train_classifier = TrainDensnet(cfg, self.cropsubset_datasplit)
        self.datasets = None
        self.__init_plots__()

    def assign_datasets(self, cropsubset_data_dir, crop_data_dir, data_dir):
        self.cropsubset_datasplit = DataSplit(self.cfg, data_dir=cropsubset_data_dir)
        self.crop_datasplit = DataSplit(self.cfg, data_dir=crop_data_dir)
        self.datasplit = DataSplit(self.cfg, data_dir=data_dir)
        self.train_classifier = TrainDensnet(self.cfg, self.cropsubset_datasplit)

    def __init_plots__(self):
        curve = plt.figure()
        self.roc_curve_plot_subset = curve.add_subplot(321)
        self.pr_curve_plot_subset = curve.add_subplot(322)
        self.roc_curve_plot_crop = curve.add_subplot(323)
        self.pr_curve_plot_crop = curve.add_subplot(324)
        self.roc_curve_plot_sw = curve.add_subplot(325)
        self.pr_curve_plot_sw = curve.add_subplot(326)


    def set_hard_neg_file(self):
        hard_neg_dir = os.path.join(*[self.crop_datasplit.data_dir,
                              'hard_neg', self.train_classifier.arch])
        if not os.path.isdir(hard_neg_dir):
            os.makedirs(hard_neg_dir)
        hard_neg_file_path = os.path.join(*[hard_neg_dir,'hard_neg'])
        if os.path.isfile(hard_neg_file_path):
            now = str(datetime.datetime.now())
            backup = hard_neg_file_path+'_backup_'+now
            shutil.copyfile(hard_neg_file_path, backup)
            os.remove(hard_neg_file_path)
        return hard_neg_file_path



    def write_hard_negitives(self, file_name_tup, idx_s, crop_file_names,
                             file_name, hard_neg_file_path=None):
        if file_name_tup[1] == 0 and self.crop_split == 'train_val':
            with open(hard_neg_file_path, "a") as fp:
                for i in idx_s[0:10]:
                    fp.write(crop_file_names[file_name][i]+'\n')
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

    def add_to_connected_group(self, score, crop_name, groups={},
                                    groups_score={}):
        #def check_spatial_connectivity(groups, crop_name):
        def crop_name_to_coor(name):
           coor = name.split('_')[-4:]
           coor = [int(val) for val in coor]
           coor[2] = coor[2] + coor[0]
           coor[3] = coor[3] + coor[1]
           coor[0] = coor[0] - 10
           coor[1] = coor[1] - 10
           coor[2] = coor[2] + 10
           coor[3] = coor[3] + 10
           return coor

        group_id = 0
        for group_id in groups:
            for member_name in groups[group_id]:
                memeber_coor = crop_name_to_coor(member_name)
                crop_coor = crop_name_to_coor(crop_name)
                overlap, _ = check_overlapping(memeber_coor, crop_coor)
                if overlap:
                    groups[group_id].append(crop_name)
                    #if score[1] > groups_score[group_id][0, 1]:
                    sum_row = groups_score[group_id] + score.unsqueeze(0)
                    #groups_score[group_id] = score.unsqueeze(0)
                    groups_score[group_id] = sum_row #score.unsqueeze(0)
                    return groups, groups_score
        groups[group_id+1] = [crop_name]
        groups_score[group_id+1] = score.unsqueeze(0)
        return groups, groups_score

    def pick_largest_group(self, groups, groups_score):
        max_num = 0
        max_group_id = 0
        max_score = torch.zeros([1, 2])
        num_largest_group = 0
        for group_id in groups:
            mems = groups[group_id]
            score = groups_score[group_id]
            #mem_nums = [len(mem) for mem in mems]
            if len(mems) > max_num :
                max_num = len(mems)
                max_group_id = group_id
                max_score = score
                num_largest_group = 1
            if len(mems) == max_num :
                max_score += score #scores core[0, 1]>max_score[0, 1]
                num_largest_group += 1
        max_score = max_score/num_largest_group
        #image_score = max_score/max_num #groups_score[max_group_id]/max_num
        return max_score#.unsqueeze(0)

    def cal_cc_score(self, idx_s, score, file_name, crop_file_names, groups,
                     groups_score):
        for i in idx_s[0:self.window_avg]:

            groups, groups_score =\
                self.add_to_connected_group(score[i],
                                            crop_file_names[file_name][i],
                                            groups, groups_score)

        return self.pick_largest_group(groups, groups_score)

    def cal_whole_image_score(self, idx_s, score, file_name, crop_file_names):
        groups = {}
        groups_score = {}
        if self.spatial_cc:
            score_com =\
                self.cal_cc_score(idx_s, score, file_name,
                                  crop_file_names, groups, groups_score)
        else:
            score_com =\
                score[idx_s[0:self.window_avg]].sum(0).unsqueeze(0)/self.window_avg
        return score_com

    def define_image_level_score(self, image_crop_score_bins, crop_file_names):
        images_score = None# torch.zeros(len(image_datasets['test'].imgs), 2).cuda()
        images_label = None #torch.LongTensor(len(image_datasets['test'].imgs)).cuda()
        if self.crop_split == 'train':
            image_name_label_tups =\
               self.datasplit.image_datasets[self.crop_split].dataset.imgs


        else:
            if self.crop_split == 'train_val':
                hard_neg_file_path = self.set_hard_neg_file()
            else:
                hard_neg_file_path = None
            image_name_label_tups =\
                self.datasplit.image_datasets[self.crop_split].imgs

        confusion_matrix = ConfusionMeter(self.datasplit.num_classes,
                                          normalized=True)
        for idx, file_name_tup in enumerate(image_name_label_tups):
            file_dir, file_name = os.path.split(file_name_tup[0])
            file_name, ext = os.path.splitext(file_name)
            if file_name in image_crop_score_bins:
                score = image_crop_score_bins[file_name]
                val_s, idx_s = score[:,1].sort(descending=True)
                image_score = self.cal_whole_image_score(idx_s, score, file_name,
                                           crop_file_names)

                if images_score is None:
                    images_score = image_score
                    images_label = torch.LongTensor([file_name_tup[1]])
                else:
                    images_score = torch.cat((images_score, image_score))
                    images_label = torch.cat((images_label,
                                              torch.LongTensor([file_name_tup[1]])))

                self.write_hard_negitives(file_name_tup, idx_s,
                                     crop_file_names, file_name, hard_neg_file_path)
            else:
                print("File name missing from bins")

        confusion_matrix.add(images_score, images_label)

        print(confusion_matrix.conf)
        return images_score, images_label, confusion_matrix

    def plot_results(self, crop_labels, crop_scores, crop_cm,
                     image_labels,  image_scores, whole_img_cm, colors):
        evaluation = Evaluation(self.crop_datasplit.classes_name,
                                crop_labels, crop_scores, crop_cm,
                                self.roc_curve_plot_crop,
                                self.pr_curve_plot_crop,
                                colors, 'Crop Test Set complete')
        evaluation.plot_results()
        evaluation = Evaluation(self.crop_datasplit.classes_name,
                                image_labels, image_scores, whole_img_cm,
                                self.roc_curve_plot_sw,
                                self.pr_curve_plot_sw,
                                colors, 'Whole Image')
        evaluation.plot_results()

    def get_pick_path(self):
        pick_dir = os.path.join(*[self.crop_datasplit.data_dir,
                              'scores_labels', self.train_classifier.arch],
                                self.crop_split)
        if not os.path.isdir(pick_dir):
            os.makedirs(pick_dir)
        score_pick_path = os.path.join(*[pick_dir,'score.p'])
        label_pick_path = os.path.join(*[pick_dir,'label.p'])
        return score_pick_path, label_pick_path

    def write_matrix(self, scores, labels):
        def write(pick_path, obj):
            with open(pick_path, 'wb') as fp:
                pickle.dump(obj, fp)

        score_pick_path, label_pick_path = self.get_pick_path()
        write(score_pick_path, scores)
        write(label_pick_path, labels)

    def read_matrix(self):
        def read(path):
            with open(path, 'rb') as fp:
                obj = pickle.load(fp)
            return obj
        score_pick_path, label_pick_path = self.get_pick_path()
        scores = read(score_pick_path)
        labels = read(label_pick_path)
        return scores, labels



    def one_vs_all_sw(self, best_model_path, colors):
        assert self.train_classifier, "Datasets are not assigned"
        if not self.pick_scores:
            crop_labels, crop_scores, crop_cm =\
                self.train_classifier.test_model(best_model_path, self.crop_datasplit, self.crop_split)
            self.write_matrix(crop_scores, crop_labels)
        else:
            crop_scores, crop_labels = self.read_matrix()
            crop_cm = None

        image_crop_score_bins, crop_file_names =\
            self.bin_crops_results(crop_scores, crop_labels)

        image_scores, image_labels, whole_img_cm =\
            self.define_image_level_score(image_crop_score_bins, crop_file_names)

        self.plot_results(crop_labels, crop_scores, crop_cm,
                          image_labels,  image_scores, whole_img_cm, colors)

    def one_vs_all_train(self, colors, best_model_path=None):
        assert self.train_classifier, "Datasets are not assigned"
        if not best_model_path:
            best_model_path = self.train_classifier.finetune_model_fun()
        labels, scores, cm = self.train_classifier.test_model(best_model_path)
        evaluation = Evaluation(self.cropsubset_datasplit.classes_name,
                                labels, scores, cm,
                                self.roc_curve_plot_subset,
                                self.pr_curve_plot_subset,
                                colors, 'Crop Test Set Subset')
        evaluation.plot_results()
        return best_model_path
