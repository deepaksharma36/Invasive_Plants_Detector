import sys
from src.utils.cfg import yfile_to_cfg
from src.tools.sw_classifier import SW_Classifier

def jap_vs_all(cfg, sw_classifier):
    jap_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all_hn'
    #jap_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all'
    jap_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all_complete'
    jap_data_dir_undivided = './dataset/data_train_test_splited/one_vs_all/jap_vs_all'
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    sw_classifier.assign_datasets(jap_train_data_dir,
                                  jap_data_dir, jap_data_dir_undivided)
    #sw_classifier.pick_scores = False
    #sw_classifier.pick_scores = True
    jap_best_model_path =\
            '/home/deepak/invasive_detector/dataset/data_train_test_splited/one_vs_all_sub_images_take2//jap_vs_all_hn/trained_models/densenet121_model_best.pth.tar'
    #        '/home/deepak/invasive_detector/dataset/data_train_test_splited/one_vs_all_sub_images_take2/jap_vs_all/trained_models/dense_model_best.pth.tar'
    sw_classifier.window_avg = 6
    #jap_best_model_path = sw_classifier.train_classifier(colors, jap_best_model_path)
    #jap_best_model_path = sw_classifier.train_classifier(colors)
    sw_classifier.test_classifier(jap_best_model_path, colors)

def phrag_vs_all(cfg, sw_classifier):
    phrag_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/phrag_vs_all_hn'
    #phrag_train_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/phrag_vs_all'
    phrag_data_dir = './dataset/data_train_test_splited/one_vs_all_sub_images_take2/phrag_vs_all_complete'
    phrag_data_dir_undivided = './dataset/data_train_test_splited/one_vs_all/phrag_vs_all'
    sw_classifier.assign_datasets(phrag_train_data_dir,
                                  phrag_data_dir, phrag_data_dir_undivided)
    phrag_best_model_path =\
            '/home/deepak/invasive_detector/dataset/data_train_test_splited/one_vs_all_sub_images_take2//phrag_vs_all_hn/trained_models/densenet121_model_best.pth.tar'
    colors = ['red', 'green', 'cornflowerblue']
    sw_classifier.window_avg = 8
    #pharg_best_model_path = sw_classifier.train_classifier(colors, phrag_best_model_path)
    #phrag_best_model_path = sw_classifier.train_classifier(colors)
    sw_classifier.test_classifier(phrag_best_model_path, colors)

def implementation_test(cfg, sw_classifier):
    data_dir = './hymenoptera_data'
    colors = ['red', 'green', 'cornflowerblue']
    sw_classifier.assign_datasets(data_dir, data_dir, data_dir)
    test_best_model_path = sw_classifier.train_classifier(colors)
    sw_classifier.test_classifier(test_best_model_path, colors)



def main():
    assert len(sys.argv) > 1, "cfg file path missing"
    cfg = yfile_to_cfg(sys.argv[1])
    sw_classifier = SW_Classifier(cfg)
    #  implementation_test(cfg, sw_classifier)
    phrag_vs_all(cfg, sw_classifier)
    jap_vs_all(cfg, sw_classifier)
    input()
    input()


if __name__ == '__main__':
    main()
