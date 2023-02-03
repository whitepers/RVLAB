import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce
from tqdm import tqdm
import matplotlib.image

plt.style.use('dark_background')

base_path = r'C:/Users/오용석/Desktop/celeba-dataset'
img_base_path = os.path.join(base_path, 'img_align_celeba')
target_img_path = os.path.join(base_path, 'processed')

# eval_list : 라벨값을 txt로 저장 / Train : o, Validation:1, Test:2
eval_list = np.loadtxt(os.path.join(base_path,'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)

""" NEW CODE 1"""
# 아래 코드 (NEW CODE 2)는 한 sample마다 한거고 이건 모든 sample 한번에 하는거
downscale = 4
n_train =162770
n_val = 19867
n_test = 19962


save_path = r'C:/Users/오용석/Desktop/celeba-dataset/changeimage'  # TODO: 여기 경로 수정
for cnt, s_path in enumerate(tqdm(eval_list, desc="Data preprocessing for resizing and normalization", disable=False)):
    filename, ext = s_path[0], s_path[1]  # img_name with label

    img_sample = np.fromfile(os.path.join(img_base_path, filename), np.uint8)
    img_sample = cv2.imdecode(img_sample, cv2.IMREAD_COLOR)
    h, w, _ = img_sample.shape

    # crop_sample은 이미지 크기 조정 -> 이미지 정사각형으로 만듬
    crop = img_sample[int((h-w)/2):int(-(h-w)/2), :]
    crop = cv2.resize(crop, dsize=(176, 176))

    # 4배로 down sample
    resized = pyramid_reduce(crop, downscale=4, channel_axis=-1)
    # 0~1사이로 정규화 과정
    norm = cv2.normalize(crop.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)

    #  split train set based on label information
    if ext == "0":
        trn_x_path = os.path.join(target_img_path, 'x_train')
        if not os.path.exists(trn_x_path): os.makedirs(trn_x_path)

        trn_y_path = os.path.join(target_img_path, 'y_train')
        if not os.path.exists(trn_y_path): os.makedirs(trn_y_path)

        np.save(os.path.join(trn_x_path, filename.split(".")[0] + '.npy'), resized)
        np.save(os.path.join(trn_y_path, filename.split(".")[0] + '.npy'), norm)

    elif ext == "1":
        val_x_path = os.path.join(target_img_path, 'x_valid')
        if not os.path.exists(val_x_path): os.makedirs(val_x_path)

        val_y_path = os.path.join(target_img_path, 'y_valid')
        if not os.path.exists(val_y_path): os.makedirs(val_y_path)

        np.save(os.path.join(val_x_path, filename.split(".")[0] + '.npy'), resized)
        np.save(os.path.join(val_y_path, filename.split(".")[0] + '.npy'), norm)

    elif ext == "2":
        tst_x_path = os.path.join(target_img_path, 'x_test')
        if not os.path.exists(tst_x_path): os.makedirs(tst_x_path)

        tst_y_path = os.path.join(target_img_path, 'y_test')
        if not os.path.exists(tst_y_path): os.makedirs(tst_y_path)

        np.save(os.path.join(tst_x_path, filename.split(".")[0] + '.npy'), resized)
        np.save(os.path.join(tst_y_path, filename.split(".")[0] + '.npy'), norm)

    else:
        raise Exception("Wrong the label description!!")


    # pad = int((crop.shape[0]-resized.shape[0]/2))
    # padded = cv2.copyMakeBorder(resized, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))


    """ DATA Saving """  # TODO: 불필요
    #저장하기
    # matplotlib.image.imsave(save_path + "/cropped_%s" % s_path, crop)
    # matplotlib.image.imsave(save_path + "/resized_%s" % s_path, resized)
    # matplotlib.image.imsave(save_path + "/padded_%s" % s_path, padded)
""" END """

# """ NEW CODE 2"""
# img_sample = np.fromfile(os.path.join(img_base_path, eval_list[0][0]), np.uint8)
# img_sample = cv2.imdecode(img_sample, cv2.IMREAD_COLOR)
# h, w, _ = img_sample.shape
#
# crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2),:]
# resized_sample = pyramid_reduce(crop_sample, downscale=3, channel_axis=-1)
#
# pad = int((crop_sample.shape[0]-resized_sample.shape[0]/2))
# padded_sample = cv2.copyMakeBorder(resized_sample, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
# """ END """






