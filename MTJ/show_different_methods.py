import os
import sys
import random
import pylab
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import skimage
import scipy
import cv2
from skimage import measure

import matplotlib.pyplot as plt
from MTJ import Different_loss
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from ran import utils
import ran.model as modellib
from sklearn.metrics import confusion_matrix
from MTJ import Muscles


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
# MUSCLE_WEIGHTS_PATH = MODEL_DIR+"mask_rcnn_muscle_0030.h5"

config = Muscles.MuscleConfig()
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


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

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

# Must call before using the dataset
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
layer_name = "seg_branch"
layer = model.get_model(layer_name,config.SMALL_CHECK_PATH)  #总的
# layer1 = model.get_model(layer_name,config.SMALL_CHECK_PATH1)#cross
# layer2 = model.get_model(layer_name,config.SMALL_CHECK_PATH2)#dice
image_id = 20

image, image_meta, gt_class_id, gt_bbox, gt_mask,jiedian =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)    #没有补零呵呵，这里补了，之后就不用了。
gtt_mask, _ = dataset.load_mask(image_id)
print("zhijie read shape",gtt_mask.shape)
print("chuli yihou shape",gt_mask.shape)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))
print(image_meta[7:11])
print(image.shape)

if image.ndim != 3:
    image = skimage.color.gray2rgb(image)
# If has an alpha channel, remove it for consistency
if image.shape[-1] == 4:
    image = image[..., :3]
print(image.shape)

# Run object detection
results = model.detect([image], verbose=1)
r = results[0]
booox=r["rois"][0]
true_box_i=utils.extract_bboxes(gt_mask)

tr_y1,tr_x1,tr_y2,tr_x2=true_box_i[0]
print("dhfdisfhj",tr_y1,tr_x1,tr_y2,tr_x2)
print("dsfgdsf",booox)
zhen_yz=int(0.5*(tr_y1+tr_y2))
zhen_xz=int(0.5*(tr_x1+tr_x2))
y_center=int(0.5*(booox[0]+booox[2]))
x_center=int(0.5*(booox[1]+booox[3]))
expanded_img=image[y_center-32:y_center+32,x_center-96:x_center+96,:].copy()
y_refine=0.5*(r['y']+y_center)
x_refine=0.5*(r['x']+x_center)
expanded_img2=image[int(y_refine)-32:int(y_refine)+32, int(x_refine)-96:int(x_refine)+96,:].copy()

expanded_img3=image[zhen_yz-32:zhen_yz+32,zhen_xz-96:zhen_xz+96,:].copy()

expanded_img = np.expand_dims(expanded_img, axis=0)
expanded_img2 = np.expand_dims(expanded_img2, axis=0)
expanded_img3 = np.expand_dims(expanded_img3, axis=0)
expanded_mask = layer.predict(expanded_img)#total
expanded_mask=np.where(expanded_mask>0.5,1,0)
expanded_mask1 = layer.predict(expanded_img2)#total2_keypoint
expanded_mask1=np.where(expanded_mask1>0.5,1,0)
expanded_mask2 = layer.predict(expanded_img3)#true  box
expanded_mask2=np.where(expanded_mask2>0.5,1,0)
zao_mask=np.zeros([512,512],dtype=np.bool)
zao_mask1=np.zeros([512,512],dtype=np.bool)
zao_mask2=np.zeros([512,512],dtype=np.bool)
zao_mask[y_center-32:y_center+32,x_center-96:x_center+96]=expanded_mask[0][:,:,0]
zao_mask1[int(y_refine)-32:int(y_refine)+32, int(x_refine)-96:int(x_refine)+96]=expanded_mask1[0][:,:,0]
zao_mask2[zhen_yz-32:zhen_yz+32,zhen_xz-96:zhen_xz+96]=expanded_mask2[0][:,:,0]
bian_list=image_meta[7:11]

# #mrcnn 与box fenge 比
# g_img=image.copy()
# m_img=image.copy()
# b_img=image.copy()
# bk_img=image.copy()
#
# plt.figure()
# ax1=plt.subplot(221)
# plt.title(r"$(a)$")
# image1, i_contours, hierarchy = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ii_img = cv2.drawContours(g_img, i_contours, -1, (255, 0, 0), 1)  # img为三通道才能显示轮廓
# ax1.imshow(g_img)
# plt.xticks([])
# plt.yticks([])
#
# ax1=plt.subplot(222)
# plt.title(r"$(b)$")
# print(r["masks"].shape)
# print(np.max(r["masks"]))
# image1, i_contours, hierarchy = cv2.findContours(r["masks"].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ii_img = cv2.drawContours(m_img, i_contours, -1, (255, 255, 0), 1)  # img为三通道才能显示轮廓
# ax1.imshow(m_img)
# plt.xticks([])
# plt.yticks([])
#
# ax2=plt.subplot(223)
# plt.title(r"$(c)$")
# image1, i_contours, hierarchy = cv2.findContours(zao_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ii_img = cv2.drawContours(b_img, i_contours, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓
# ax2.imshow(b_img)
# plt.xticks([])
# plt.yticks([])
#
# ax2=plt.subplot(224)
# plt.title(r"$(d)$")
# image1, i_contours, hierarchy = cv2.findContours(zao_mask1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ii_img = cv2.drawContours(bk_img, i_contours, -1, (0, 0, 255), 1)  # img为三通道才能显示轮廓
# ax2.imshow(bk_img)
# plt.xticks([])
# plt.yticks([])
#
# plt.show()

###############################################

#全部 对比，重点选取
g_img=image.copy()  #真实
gt_img=image.copy()
m_img=image.copy()
b_img=image.copy()
bk_img=image.copy()

plt.figure()
ax1=plt.subplot(231)
# plt.title(r"$(a)$")
image1, gt_contours, hierarchy = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ii_img = cv2.drawContours(gt_img, gt_contours, -1, (255, 0, 0), 1)  # img为三通道才能显示轮廓
# ax1.imshow(gt_img[bian_list[0]:bian_list[2],bian_list[1]:bian_list[3]])
ax1.imshow(gt_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])

ax5=plt.subplot(232)
# plt.title(r"$(e)$")
image1, z1_contours, hierarchy = cv2.findContours(zao_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ii_img = cv2.drawContours(bk_img, z1_contours, -1, (0, 255, 255), 1)  # img为三通道才能显示轮廓

########################

ax5.imshow(bk_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])

ax3=plt.subplot(233)
# plt.title(r"$(c)$")
image1, r_contours, hierarchy = cv2.findContours(r["masks"].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ii_img = cv2.drawContours(m_img, r_contours, -1, (255, 255, 0), 1)  # img为三通道才能显示轮廓
ax3.imshow(m_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])

ax4=plt.subplot(234)
# plt.title(r"$(d)$")
image1, z_contours, hierarchy = cv2.findContours(zao_mask1.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ii_img = cv2.drawContours(b_img, z_contours, -1, (255, 0, 255), 1)  # img为三通道才能显示轮廓
ax4.imshow(b_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])



ax2=plt.subplot(235)
# plt.title(r"$(f)$")

ii_img = cv2.drawContours(g_img, gt_contours, -1, (255, 0, 0), 1)  # img为三通道才能显示轮廓
# ii_img = cv2.drawContours(g_img, z_contours, -1, (0, 255, 0), 1)  # img为三通道才能显示轮廓
ii_img = cv2.drawContours(g_img, r_contours, -1, (255, 255, 0), 1)  # img为三通道才能显示轮廓
ii_img = cv2.drawContours(g_img, z1_contours, -1, (0, 255, 255), 1)  # img为三通道才能显示轮廓
ax2.imshow(g_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])

ax2=plt.subplot(236)
# plt.title(r"$(b)$")
ii_img = cv2.drawContours(bk_img, z_contours, -1, (255, 0, 255), 1)  # img为三通道才能显示轮廓
ax2.imshow(bk_img[int(y_refine)-40:int(y_refine)+40, int(x_refine)-90:int(x_refine)+90])
plt.xticks([])
plt.yticks([])

plt.show()
# true_box_i=utils.extract_bboxes(gt_mask)
# tr_y1,tr_x1,tr_y2,tr_x2=true_box_i[0]
# # Display results
# ax = get_ax(1)
# r = results[0]
# #r["f2"],r["masks"]
# booox=r["rois"][0]
# print("houxuankuang",booox)
# y_center=int(0.5*(booox[0]+booox[2]))
# x_center=int(0.5*(booox[1]+booox[3]))
# expanded_img=image[y_center-32:y_center+32,x_center-96:x_center+96,:]
# expanded_img2=image[int(r['y'])-32:int(r['y'])+32, int(r['x'])-96:int(r['x'])+96,:]
# expanded_img = np.expand_dims(expanded_img, axis=0)
# expanded_img2 = np.expand_dims(expanded_img2, axis=0)
# expanded_mask = layer.predict(expanded_img)#total
# expanded_mask=np.where(expanded_mask>0.5,1,0)
# expanded_mask1 = layer.predict(expanded_img2)#total2_keypoint
# expanded_mask1=np.where(expanded_mask1>0.5,1,0)
# expanded_mask2 = layer2.predict(expanded_img)#dice
# expanded_mask2=np.where(expanded_mask2>0.5,1,0)
# zao_mask=np.zeros([512,512],dtype=np.bool)
# zao_mask1=np.zeros([512,512],dtype=np.bool)
# zao_mask2=np.zeros([512,512],dtype=np.bool)
# zao_mask[y_center-32:y_center+32,x_center-96:x_center+96]=expanded_mask[0][:,:,0]
# zao_mask1[int(r['y'])-32:int(r['y'])+32, int(r['x'])-96:int(r['x'])+96]=expanded_mask1[0][:,:,0]
# zao_mask2[y_center-32:y_center+32,x_center-96:x_center+96]=expanded_mask2[0][:,:,0]
# # y_true=gtt_mask[booox[0]:booox[2],booox[1]:booox[3],0]
# # plt.imshow(y_true)
# # pylab.show()
# y_true=gt_mask[booox[0]:booox[2],booox[1]:booox[3],0]   #真实值回归框
# yy_true=gt_mask[tr_y1:tr_y2,tr_x1:tr_x2,0]   #真实框真值
# y_biaozhun=scipy.misc.imresize(yy_true,(booox[2]-booox[0],booox[3]-booox[1]))  #真是框插值
# yy_pre1=zao_mask[tr_y1:tr_y2,tr_x1:tr_x2]   #候选框中心点为中心，真实框预测
# yy_pre2=zao_mask1[tr_y1:tr_y2,tr_x1:tr_x2]  #关键点为中心，真实框预测
#
#
# y_pred1=zao_mask[booox[0]:booox[2],booox[1]:booox[3]]
#
# # y_pred11=r["f2"][booox[0]:booox[2],booox[1]:booox[3],0]
# y_pred2=r["masks"][booox[0]:booox[2],booox[1]:booox[3],0]
# dicea1=dice_acc(y_true,y_pred1)
# dicea2=dice_acc(y_true,y_pred2)
# speca1=specifiv(y_true,y_pred1)
# speca2=specifiv(y_true,y_pred2)
# sensa1=sensitiv(y_true,y_pred1)
# sensa2=sensitiv(y_true,y_pred2)
# print("依次是",[dicea1,dicea2,speca1,speca2,sensa1,sensa2])
# plt.figure()
# plt.subplot(231)
# plt.title("(a)")
# plt.imshow(y_true)
# plt.subplot(232)
# plt.title("(b)")
# plt.imshow(y_pred1)
# plt.subplot(233)
# plt.title("(c)")
# plt.imshow(y_pred2)
# plt.subplot(234)
# plt.title("(d)")
# plt.imshow(zao_mask1[tr_y1:tr_y2,tr_x1:tr_x2])
# plt.subplot(235)
# plt.title("(e)")
# plt.imshow(zao_mask[tr_y1:tr_y2,tr_x1:tr_x2])
# plt.subplot(236)
# plt.title("(f)")
# plt.imshow(yy_true)
# plt.show()
#
# plt.figure()
# plt.subplot(231)
# plt.title("(a)")
# plt.imshow(gt_mask[int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48,0])
# plt.subplot(232)
# plt.title("(b)")
# plt.imshow(zao_mask[int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48])
# plt.subplot(233)
# plt.title("(c)")
# plt.imshow(r['masks'][int(r['y'])-15:int(r['y'])+15, int(r['x'])-48:int(r['x'])+48,0])
# plt.subplot(234)
# plt.title("(d)")
# plt.imshow(yy_true)
# plt.subplot(235)
# plt.title("(e)")
# plt.imshow(zao_mask[tr_y1:tr_y2,tr_x1:tr_x2])
# plt.subplot(236)
# plt.title("(f)")
# plt.imshow(r['masks'][tr_y1:tr_y2,tr_x1:tr_x2,0])
# plt.show()
#
#
# visualize.display_instances(image, r['rois'], r['f2'], r['class_ids'],
#                             dataset.class_names, r['scores'], ax=ax,
#                             title="Predictions")
# pylab.show()
#
# print("y:",r['y'])
# print("x:",r['x'])
# print("zheny",jiedian[0])
# mm=r['masks'][:,:,0]
# plt.imshow(mm)
# pylab.show()


# # visualize.display_instances(image, r['rois'], r['f2'], r['class_ids'],
# #                             dataset.class_names, r['scores'], ax=ax,
# #                             title="Predictions")
# # pylab.show()
#
# nn=r['f2'][:,:,0]
# plt.imshow(nn)
# pylab.show()














# target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
#     image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
# log("target_rpn_match", target_rpn_match)
# log("target_rpn_bbox", target_rpn_bbox)
#
# positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
# negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
# neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
# positive_anchors = model.anchors[positive_anchor_ix]
# negative_anchors = model.anchors[negative_anchor_ix]
# neutral_anchors = model.anchors[neutral_anchor_ix]
# log("positive_anchors", positive_anchors)
# log("negative_anchors", negative_anchors)
# log("neutral anchors", neutral_anchors)
# print(positive_anchors[0])
# # small_img=image[np.floor(positive_anchors[0]):np.floor(positive_anchors[0])+45,np.floor(positive_anchors[1]):np.floor(positive_anchors[1])+90]
# # print(np.min(image))
# # print(np.min(small_img))
# # Apply refinement deltas to positive anchors
# # refined_anchors = utils.apply_box_deltas(
# #     positive_anchors,
# #     target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
# # log("refined_anchors", refined_anchors, )
# # print(positive_anchors)
# # print(refined_anchors)
# # visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())
# # pylab.show()
# #
# # pillar = model.keras_model.get_layer("ROI").output
# # nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
# # if nms_node is None:
# #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
# # if nms_node is None: #TF 1.9-1.10
# #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")
# #
# # rpn = model.run_graph([image], [
# #     ("rpn_class", model.keras_model.get_layer("rpn_class").output),
# #     ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
# #     ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
# #     ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
# #     ("post_nms_anchor_ix", nms_node),
# #     ("proposals", model.keras_model.get_layer("ROI").output),
# # ])
# # limit = 100
# # sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
# # visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())
# # pylab.show()
# #
# # # Show top anchors by score (before refinement)
# # limit = 50
# # ax = get_ax(1, 2)
# # pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
# # refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
# # refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
# # visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
# #                      refined_boxes=refined_anchors[:limit], ax=ax[0])
# # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])
# # pylab.show()
#
#
# # Get input and output to classifier and mask heads.
# mrcnn = model.run_graph([image], [
#     ("proposals", model.keras_model.get_layer("ROI").output),
#     ("probs", model.keras_model.get_layer("mrcnn_class").output),
#     ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
#     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
#     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
#     # ("rough",model.keras_model.get_layer("rough_mask").output),
# ])
#
# det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
# det_count = np.where(det_class_ids == 0)[0][0]
# det_class_ids = det_class_ids[:det_count]
# detections = mrcnn['detections'][0, :det_count]
#
# print("{} detections: {}".format(
#     det_count, np.array(dataset.class_names)[det_class_ids]))
#
# captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
#             for c, s in zip(detections[:, 4], detections[:, 5])]
# visualize.draw_boxes(
#     image,
#     refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
#     visibilities=[2] * len(detections),
#     captions=captions, title="Detections",
#     ax=get_ax())
# pylab.show()
#
# # Proposals are in normalized coordinates. Scale them
# # to image coordinates.
# h, w = config.IMAGE_SHAPE[:2]
# proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)
#
# # Class ID, score, and mask per proposal
# roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
# roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
# roi_class_names = np.array(dataset.class_names)[roi_class_ids]
# roi_positive_ixs = np.where(roi_class_ids > 0)[0]
#
# # How many ROIs vs empty rows?
# print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
# print("{} Positive ROIs".format(len(roi_positive_ixs)))
#
# # Class counts
# print(list(zip(*np.unique(roi_class_names, return_counts=True))))
#
# roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
# log("roi_bbox_specific", roi_bbox_specific)
#
# # Apply bounding box transformations
# # Shape: [N, (y1, x1, y2, x2)]
# refined_proposals = utils.apply_box_deltas(
#     proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
# log("refined_proposals", refined_proposals)


# keep = np.where(roi_class_ids > 0)[0]
# print("Keep {} detections:\n{}".format(keep.shape[0], keep))
# keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
# print("Remove boxes below {} confidence. Keep {}:\n{}".format(
#     config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))
#
#
# pre_nms_boxes = refined_proposals[keep]
# pre_nms_scores = roi_scores[keep]
# pre_nms_class_ids = roi_class_ids[keep]
#
# nms_keep = []
# for class_id in np.unique(pre_nms_class_ids):
#     # Pick detections of this class
#     ixs = np.where(pre_nms_class_ids == class_id)[0]
#     # Apply NMS
#     class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
#                                             pre_nms_scores[ixs],
#                                             config.DETECTION_NMS_THRESHOLD)
#     # Map indicies
#     class_keep = keep[ixs[class_keep]]
#     nms_keep = np.union1d(nms_keep, class_keep)
#     print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20],
#                                    keep[ixs], class_keep))
#
# keep = np.intersect1d(keep, nms_keep).astype(np.int32)
# print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))
#
# ixs = np.arange(len(keep))  # Display all
# # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
# captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
#             for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
# visualize.draw_boxes(
#     image, boxes=proposals[keep][ixs],
#     refined_boxes=refined_proposals[keep][ixs],
#     visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
#     captions=captions, title="Detections after NMS",
#     ax=get_ax())
# pylab.show()
#
# activations = model.run_graph([image], [
#     ("input_image",        model.keras_model.get_layer("input_image").output),
#     ("res2c_out",          model.keras_model.get_layer("res2c_out").output),
#     ("res3c_out",          model.keras_model.get_layer("res3c_out").output),
#     ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
#     ("rpn_bbox",           model.keras_model.get_layer("rpn_bbox").output),
#     ("roi",                model.keras_model.get_layer("ROI").output),
# ])
#
# _ = plt.imshow(modellib.unmold_image(activations["input_image"][0],config))
# # Backbone feature map
# display_images(np.transpose(activations["res2c_out"][0,:,:,:4], [2, 0, 1]), cols=4)