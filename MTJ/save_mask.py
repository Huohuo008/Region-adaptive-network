import os
import sys
import random
import pylab
import numpy as np
import tensorflow as tf
import skimage.io
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("../../")
import cv2
# Import Mask RCNN
from ran import utils
import ran.model as modellib
from sklearn.metrics import confusion_matrix
from MTJ import Muscles
sys.path.append(ROOT_DIR)  # To find local version of the library
from sklearn.metrics import confusion_matrix
from MTJ.Different_loss import models

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = Muscles.MuscleConfig()
# MUSCLE_DIR = 'E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\'
MUSCLE_DIR = 'E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\wangkainibiaozhu\\previous data version\\'
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

def sensitiv(y_true,y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    cm=confusion_matrix(y_true_f,y_pred_f)
    print(cm)
    return cm[0][0]/(cm[0][0]+cm[1][0])

def specifiv(y_true,y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    cm=confusion_matrix(y_true_f,y_pred_f)
    return cm[1][1]/(cm[1][1]+cm[0][1])

def dice_acc(y_true,y_pred):
    smooth = 0.
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)



dataset = Muscles.MuscleDataset()
dataset.load_muscle(MUSCLE_DIR, "val")

dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
print(MODEL_DIR)
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path,by_name=True)

model1=models()
model1.load_weights(ROOT_DIR+"\\mask_rcnn_totalp2_rgb_epoch.h5")
model2=models()
model2.load_weights(ROOT_DIR+"\\mask_rcnn_crossp_rgb_epoch.h5")
model3=models()
model3.load_weights(ROOT_DIR+"\\mask_rcnn_dicep_rgb_epoch.h5")

layer_name = "seg_branch"
layer = model.get_model(layer_name,config.SMALL_CHECK_PATH)  #总
layer1 = model.get_model(layer_name,config.SMALL_CHECK_PATH1)  #总的
layer2= model.get_model(layer_name,config.SMALL_CHECK_PATH2)  #总的
# global_model=models()
# global_model.load_weights(os.path.join(ROOT_DIR, "mask_rcnn_global_rgb_epoch.h5"))
y_trues=[]
ypred1=[]
ypred2=[]

for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask, jiedian =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    Imgname=dataset.image_info[image_id]["id"]
    zb_name = Imgname.split('.')[:-1][0]
    true_box_i = utils.extract_bboxes(gt_mask)
    tr_y1, tr_x1, tr_y2, tr_x2 = true_box_i[0]
    zhen_yz = int(0.5 * (tr_y1 + tr_y2))
    zhen_xz = int(0.5 * (tr_x1 + tr_x2))      #真实的中心点啊
    info = dataset.image_info[image_id]
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    print(image.shape)
    # r_img=cv2.resize(image,(256,256),interpolation=1)
    # r_img=np.expand_dims(r_img,axis=0)
    # r_mask=layer.predict(r_img)[0]
    # r_mask=cv2.resize(r_mask,(512,512),interpolation=1)
    results = model.detect([image], verbose=1)

    r = results[0]
    # r["f2"],r["masks"]
    box_list=r["rois"]            #回归的box框
    p_x=r['x']                    #坐标值
    if len(box_list) and p_x>120:
        booox = r["rois"][0]
        yo_center = int(0.5 * (booox[0] + booox[2]))
        xo_center = int(0.5 * (booox[1] + booox[3]))
        y_refine = int(0.5 * (r['y'] + yo_center))
        x_refine = int(0.5 * (r['x'] + xo_center))
        x_zuo=np.max([x_refine-32,tr_x1])
        x_you=np.min([x_refine+32,tr_x2])
        if x_zuo<x_you-32:
            expanded_img = image[yo_center - 32:yo_center + 32, xo_center - 96:xo_center + 96, :].copy()
            # expanded_img2 = image[int(r['y']) - 32:int(r['y']) + 32, int(r['x']) - 108:int(r['x']) + 84, :].copy()
            expanded_img2 = image[int(y_refine) - 32:int(y_refine) + 32, int(x_refine) - 96:int(x_refine) + 96, :].copy()
            expanded_img3 = image[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96, :].copy()
            expanded_img4 = image[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96, :].copy()
            expanded_img5 = image[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96, :].copy()

            expanded_img = np.expand_dims(expanded_img, axis=0)
            expanded_img2 = np.expand_dims(expanded_img2, axis=0)
            expanded_img3 = np.expand_dims(expanded_img3, axis=0)
            expanded_img4 = np.expand_dims(expanded_img4, axis=0)
            expanded_img5 = np.expand_dims(expanded_img5, axis=0)

            expanded_mask = layer.predict(expanded_img)  # total
            expanded_mask = np.where(expanded_mask > 0.5, 1, 0)
            expanded_mask1 = layer.predict(expanded_img2)  # total2_keypoint
            expanded_mask1 = np.where(expanded_mask1 > 0.5, 1, 0)
            expanded_mask2 = model1.predict(expanded_img3)  # true  box
            expanded_mask2 = np.where(expanded_mask2 > 0.5, 1, 0)
            expanded_mask3 = model2.predict(expanded_img4)  # true  box   cross
            expanded_mask3 = np.where(expanded_mask3 > 0.5, 1, 0)
            expanded_mask4 = model3.predict(expanded_img5)  # true  box   dice
            expanded_mask4 = np.where(expanded_mask4 > 0.5, 1, 0)

            zao_mask = np.zeros([512, 512], dtype=np.bool)
            zao_mask1 = np.zeros([512, 512], dtype=np.bool)
            zao_mask2 = np.zeros([512, 512], dtype=np.bool)
            zao_mask3 = np.zeros([512, 512], dtype=np.bool)
            zao_mask4 = np.zeros([512, 512], dtype=np.bool)
            zao_mask[yo_center - 32:yo_center + 32, xo_center - 96:xo_center + 96] = expanded_mask[0][:, :, 0]
            zao_mask1[int(y_refine) - 32:int(y_refine) + 32, int(x_refine) - 96:int(x_refine) + 96] = expanded_mask1[0][:, :, 0]
            zao_mask2[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96] = expanded_mask2[0][:, :, 0]
            zao_mask3[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96] = expanded_mask3[0][:, :, 0]
            zao_mask4[zhen_yz - 32:zhen_yz + 32, zhen_xz - 96:zhen_xz + 96] = expanded_mask4[0][:, :, 0]
            bian_list = image_meta[7:11]
            # zao_mask[yo_center - 32:yo_center + 32, xo_center - 96:xo_center + 96] = expanded_mask[0][:, :, 0]
            # zao_mask1[int(r['y']) - 32:int(r['y']) + 32, int(r['x']) - 108:int(r['x']) + 84] = expanded_mask1[0][:, :, 0]
            y_true = gt_mask[tr_y1:tr_y2,tr_x1:tr_x2, 0]
            # # #
            y_true=np.where(y_true==1,255,0)
            # # #
            y_true_dst=MUSCLE_DIR+"segresult999\\"+zb_name+"_ytrue.jpg"
            skimage.io.imsave(y_true_dst,y_true)
            # # y_true1= gt_mask[int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48,0]
            # # #
            # # y_true1=np.where(y_true1==1,255,0)
            # # #
            # # y_true1_dst = MUSCLE_DIR + "segresult1\\" + zb_name + "_ytruepoint.jpg"
            # # skimage.io.imsave(y_true1_dst,y_true1)
            # y_maskrcnn = r["masks"][tr_y1:tr_y2,x_zuo:x_you, 0]
            # # #
            # y_maskrcnn=np.where(y_maskrcnn==1,255,0)
            # # #
            # y_maskrcnn_dst = MUSCLE_DIR + "segresult930\\" + zb_name + "_ymaskrcnn.jpg"
            # skimage.io.imsave(y_maskrcnn_dst,y_maskrcnn)
            # # y_maskrcnn1 =r["masks"][int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48,0]
            # # #
            # # y_maskrcnn1=np.where(y_maskrcnn1==1,255,0)
            # # #
            # # y_maskrcnn1_dst = MUSCLE_DIR + "segresult1\\" + zb_name + "_ymaskrcnnpoint.jpg"
            # # skimage.io.imsave(y_maskrcnn1_dst,y_maskrcnn1)
            # y_zhongxin=zao_mask[tr_y1:tr_y2,x_zuo:x_you]
            # # #
            # y_zhongxin=np.where(y_zhongxin==1,255,0)
            # # #
            # y_zhongxin_dst = MUSCLE_DIR + "segresult930\\" + zb_name + "_yzhongxin.jpg"
            # skimage.io.imsave(y_zhongxin_dst,y_zhongxin)
            # # y_zhongxin1=zao_mask[int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48]
            # # #
            # # y_zhongxin1=np.where(y_zhongxin1==1,255,0)
            # # #
            # # y_zhongxin1_dst = MUSCLE_DIR + "segresult1\\" + zb_name + "_yzhongxinpoint.jpg"
            # # skimage.io.imsave(y_zhongxin1_dst,y_zhongxin1)
            # y_key=zao_mask1[tr_y1:tr_y2,x_zuo:x_you]
            # # #
            # y_key=np.where(y_key==1,255,0)
            # # #
            # y_key_dst = MUSCLE_DIR + "segresult930\\" + zb_name + "_ykey.jpg"
            # skimage.io.imsave(y_key_dst, y_key)

            y_shiji=zao_mask2[tr_y1:tr_y2,tr_x1:tr_x2]
            y_shiji=np.where(y_shiji==1,255,0)
            y_shiji_dst = MUSCLE_DIR + "segresult999\\" + zb_name + "_yshiji.jpg"
            skimage.io.imsave(y_shiji_dst, y_shiji)

            y_shiji1=zao_mask3[tr_y1:tr_y2,tr_x1:tr_x2]
            y_shiji1=np.where(y_shiji1==1,255,0)
            y_shiji1_dst = MUSCLE_DIR + "segresult999\\" + zb_name + "_yshijicross.jpg"
            skimage.io.imsave(y_shiji1_dst, y_shiji1)

            y_shiji2=zao_mask4[tr_y1:tr_y2,tr_x1:tr_x2]
            y_shiji2=np.where(y_shiji2==1,255,0)
            y_shiji_dst2 = MUSCLE_DIR + "segresult999\\" + zb_name + "_yshijidice.jpg"
            skimage.io.imsave(y_shiji_dst2, y_shiji2)
        # y_key1=zao_mask1[int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48]
        # #
        # y_key1=np.where(y_key1==1,255,0)
        # #
        # y_key1_dst = MUSCLE_DIR + "segresult1\\" + zb_name + "_ykeypoint.jpg"
        # skimage.io.imsave(y_key1_dst,y_key1)

        ##########################
        #global model 截出来的
        #############################
        # y_key = r_mask[tr_y1:tr_y2, tr_x1:tr_x2]
        # #
        # y_key = np.where(y_key == 1, 255, 0)
        # #
        # y_key_dst = MUSCLE_DIR + "segresult929\\" + zb_name + "_rkey.jpg"
        # skimage.io.imsave(y_key_dst, y_key)
        # y_key1 = r_mask[int(r['y']) - 15:int(r['y']) + 15, int(r['x']) - 48:int(r['x']) + 48]
        # #
        # y_key1 = np.where(y_key1 == 1, 255, 0)
        # #
        # y_key1_dst = MUSCLE_DIR + "segresult929\\" + zb_name + "_rkeypoint.jpg"
        # skimage.io.imsave(y_key1_dst, y_key1)
    else:
        continue
# dicea1 = dice_acc(y_trues, ypred1)
# dicea2 = dice_acc(y_trues, ypred2)
# speca1 = specifiv(y_trues, ypred1)
# speca2 = specifiv(y_trues, ypred2)
# sensa1 = sensitiv(y_trues, ypred1)
# sensa2 = sensitiv(y_trues, ypred2)
# print("依次是", [dicea1, dicea2, speca1, speca2, sensa1, sensa2])
