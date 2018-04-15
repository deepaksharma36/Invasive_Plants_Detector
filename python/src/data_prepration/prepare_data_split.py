import os
import sys
import os
from shutil import copyfile
from PIL import Image

from src.utils.cfg import yfile_to_cfg
class PrepareDataSplit():

    def __init__(self, cfg):
        self.source_train_split_file =\
            os.path.expandvars(cfg.DATA_IN.TRAIN_SPLIT_FILE_INV_NON_INV)
        self.source_test_split_file =\
            os.path.expandvars(cfg.DATA_IN.TEST_SPLIT_FILE_INV_NON_INV)
        self.destination_train_split_file =\
            os.path.expandvars(cfg.SPLIT_FILE_IN_OUT.TRAIN_SPLIT_FILE_PHRAG_JAP_NON_INV)
        self.destination_test_split_file  =\
            os.path.expandvars(cfg.SPLIT_FILE_IN_OUT.TEST_SPLIT_FILE_PHRAG_JAP_NON_INV)
        self.data_out_dir = os.path.expandvars(cfg.PREPED_DATA_IN_OUT.DATA_OUT_DIR)
        self.split_write = cfg.SPLIT_FILE_IN_OUT.WRITE
        self.bb_file = os.path.expandvars(cfg.DATA_IN.BB_FILE)
        self.data_in_dir = os.path.expandvars(cfg.DATA_IN.DATA_IN_DIR)
        #self.invasive_dir = cfg.DATA_IN.INVASIVE_DIR
        #self.non_invasive_dir = cfg.DATA_IN.NON_INVASIVE_DIR
        self.class_id = {1: 'common_reed', 2: 'japanese_knotweed'}
        #self.class_names = list(self.class_id.values())
        self.class_names = ['common_reed',  'japanese_knotweed']
        self.crop = [2700, 0, 5500, -1]
        self.one_vs_all_dir = ['common_reed_vs_all', 'japanese_knotweed_vs_all']

    def get_sample_name_class_mapping(self):
        name_class_map = {}
        with open(self.bb_file) as fp:
            for line in fp:
                line = line.strip().split(',')
                name_class_map[line[0]] = self.class_id[int(line[1])]
        return name_class_map

    def invasive_class_to_sub_classes(self, source_split_path, name_class_map):
        with open(source_split_path) as fp:
            for line in fp:
                path_parts = line.split(os.sep)
                path_parts_extra = path_parts[1].strip().split(',')
                if path_parts_extra[0] in name_class_map:
                    path_parts[0] = name_class_map[path_parts_extra[0]]
                    #path_extra = ','.join(path_parts_extra)
                    line = os.sep.join(path_parts)
                yield line

    def write_file(lines, file_path):
        with open(file_path, 'w') as fp:
            for line in lines:
                fp.write(line)

    def prepare_data_split_files(self):
        split_files = {}
        source_split_files = {'train': self.source_train_split_file , 'test': self.source_test_split_file}
        destination_split_files = {'train': self.destination_train_split_file , 'test': self.destination_test_split_file}
        name_class_map = self.get_sample_name_class_mapping()

        for split_file_type in source_split_files:
            file_entries = list(self.invasive_class_to_sub_classes(source_split_files[split_file_type], name_class_map))
            split_files[split_file_type] = file_entries
            if self.split_write:
                self.write_file(file_entries,
                                destination_split_files[split_file_type])
        return split_files

    def copy_img_files(self, split_file_entries,  split_type):
        def get_destination_paths(desti_img_rel_path, desti_img_rel_path_other,
                                  class_id, other_class_id):
            dest_img_abs_paths = []
            dest_img_abs_paths.append(os.path.join(*[self.data_out_dir,
                                                     self.one_vs_all_dir[class_id],
                                                     split_type,
                                                     desti_img_rel_path]))
            dest_img_abs_paths.append(os.path.join(*[self.data_out_dir,
                                                     self.one_vs_all_dir[other_class_id],
                                                     split_type,
                                                     desti_img_rel_path_other]))
            return dest_img_abs_paths

        for entry in split_file_entries:
            entry_l = entry.strip().split(',')
            source_img_path_l = entry_l[0].split(os.sep)
            class_name = source_img_path_l[0]
            desti_img_rel_path = os.sep.join(source_img_path_l)+'_z5.jpg'
            desti_img_rel_path_l = desti_img_rel_path.split(os.sep)
            desti_img_rel_path_l[-2] = 'all'
            desti_img_rel_path_other = os.path.join(*desti_img_rel_path_l)
            if class_name in self.class_names:
                source_img_path_l[0] = 'z5'
            source_img_rel_path = os.sep.join(source_img_path_l)+'_z5.jpg'
            source_img_abs_path =\
                os.path.join(*[self.data_in_dir, source_img_rel_path])
            if class_name in self.class_names:
                source_img_path_l[0] = 'z5'
                source_img_rel_path = os.sep.join(source_img_path_l)+'_z5.jpg'
                class_id = self.class_names.index(class_name)
                other_class_id = (class_id + 1)%len(self.class_names)
                dest_img_abs_paths =\
                    get_destination_paths(desti_img_rel_path,
                                          desti_img_rel_path_other,
                                          class_id, other_class_id)
            else:
                class_id = 0
                other_class_id = 1
                desti_img_rel_path = desti_img_rel_path_other
                dest_img_abs_paths =\
                    get_destination_paths(desti_img_rel_path,
                                          desti_img_rel_path_other,
                                          class_id, other_class_id)


            for dest_img_abs_path in dest_img_abs_paths:
                desti_dir_path = os.path.dirname(dest_img_abs_path)
                if not os.path.isdir(desti_dir_path):
                    os.makedirs(desti_dir_path)
                try:
                    if not self.crop or source_img_path_l[0] == 'z5_extracted':
                        copyfile(source_img_abs_path, dest_img_abs_path)
                    else:
                        img = Image.open(source_img_abs_path)
                        img = img.crop((self.crop[1], self.crop[0],
                                        img.width-1, self.crop[2]))
                        img.save(dest_img_abs_path)
                except Exception as e:
                    print(e)

def main():
    assert len(sys.argv) > 1, "Config file path as argument missing"
    cfg = yfile_to_cfg(sys.argv[1])
    prepare_data_split = PrepareDataSplit(cfg)
    split_files =  prepare_data_split.prepare_data_split_files()

    for split_file_type in split_files:
        prepare_data_split.copy_img_files(split_files[split_file_type], split_file_type)

if __name__ == '__main__':
    main()
