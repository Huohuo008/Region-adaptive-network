'''
比较U-net分割网络在不同loss下的表现，以及显示。
包括，训练，测试，显示，保存。
'''
import os
import random
import datetime
import re
from sklearn.metrics import confusion_matrix
import numpy as np
from glob import glob
import skimage
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import pylab
import cv2
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from ran import utils

ROOT_DIR = os.path.abspath("../../")

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def build_rough_mask_graph(simage,num_classes=1,train_bn=True):

    conv1=KL.Conv2D(32,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv1')(simage)
    conv1=KL.Conv2D(32,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv1_2')(conv1)
    bn1 = BatchNorm(name='rough_mask_bn1')(conv1, training=train_bn)
    pool1=KL.MaxPooling2D((2,2))(bn1)

    conv2=KL.Conv2D(64,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv2')(pool1)
    conv2=KL.Conv2D(64,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv2_2')(conv2)
    bn2 = BatchNorm(name='rough_mask_bn2')(conv2, training=train_bn)
    pool2 = KL.MaxPooling2D((2, 2))(bn2)

    conv3=KL.Conv2D(128,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv3')(pool2)
    conv3 =KL.Conv2D(128, (3, 3), strides=1, padding="same", activation='relu',name='rough_mask_conv3_2')(conv3)
    bn3 = BatchNorm(name='rough_mask_bn3')(conv3, training=train_bn)
    pool3 =KL.MaxPooling2D((2, 2))(bn3)

    conv4=KL.Conv2D(128,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv4')(pool3)
    conv4=KL.Conv2D(128,(3,3),strides=1,padding="same",activation='relu',name='rough_mask_conv4_2')(conv4)
    bn4 = BatchNorm(name='rough_mask_bn4')(conv4, training=train_bn)
    pool4 = KL.Dropout(0.25)(bn4)

    deconv1 = KL.Conv2DTranspose(128, (2, 2), strides=2, activation="relu",name="rough_mask_deconv1")(pool4)
    uco1 = KL.concatenate([deconv1,conv3])
    conv5=KL.Conv2D(128,(3,3),padding="same",activation="relu",name='rough_mask_conv5')(uco1)
    conv5 =KL.Conv2D(128, (3, 3), padding="same", activation="relu",name='rough_mask_conv5_2')(conv5)
    bn5 = BatchNorm(name='rough_mask_bn5')(conv5, training=train_bn)

    deconv2=KL.Conv2DTranspose(64, (2, 2), strides=2, activation="relu",name="rough_mask_deconv2")(bn5)
    co2=KL.concatenate([deconv2,conv2])
    conv6 =KL.Conv2D(64, (3, 3),padding="same", activation="relu",name='rough_mask_conv6')(co2)
    conv6 =KL.Conv2D(64, (3, 3), padding="same", activation="relu",name='rough_mask_conv6_2')(conv6)
    bn6 = BatchNorm(name='rough_mask_bn6')(conv6, training=train_bn)

    deconv3=KL.Conv2DTranspose(32, (2, 2), strides=2, activation="relu",name="rough_mask_deconv3")(bn6)
    co3=KL.concatenate([deconv3,conv1])
    conv7 =KL.Conv2D(32, (3, 3), padding="same", activation="relu",name='rough_mask_conv7')(co3)
    conv7 =KL.Conv2D(32, (3, 3), padding="same", activation="relu",name='rough_mask_conv7_2')(conv7)
    bn7 = BatchNorm(name='rough_mask_bn7')(conv7, training=train_bn)
    # aux0=KL.TimeDistributed(KL.Conv2D())

    pre_final=KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid",name="rough_mask")(bn7)

    return pre_final

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

def dice_met(y_true,y_pred):
    smooth = 0.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_acc(y_true,y_pred):
    smooth = 0.
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)

def dice_coeff(y_true,y_pred):
    # smooth = 0.
    # intersection = K.sum(y_true * y_pred)
    # dice=(2. * intersection + smooth) / (K.sum(y_true*y_true) + K.sum(y_pred* y_pred) + smooth)
    return 1. - dice_met(y_true, y_pred)

def cross_loss(y_true,y_pred):
    loss = K.binary_crossentropy(target=y_true, output=y_pred)
    return loss

def rough_mask_loss_gragh(predmask,acc_smallmask):
    mask_shape = tf.shape(acc_smallmask)
    y_true=K.reshape(acc_smallmask, (-1, mask_shape[2], mask_shape[3]))
    y_pred = K.reshape(predmask, (-1, mask_shape[2], mask_shape[3]))
    loss1=dice_coeff(y_true,y_pred)
    loss2=K.binary_crossentropy(target=y_true, output=y_pred)
    loss2 = K.mean(loss2)
    loss=0.5*loss1+0.5*loss2
    return loss


def total_loss(y_true,y_pred):
    return rough_mask_loss_gragh(y_pred,y_true)

def get_small(img_list,mask_list,box_list):
    small_img_list=[]
    small_mask_list=[]
    bankuan=96
    banchang=32
    for i in range(len(img_list)):
        image=img_list[i]
        mask=mask_list[i]
        y1,x1,y2,x2=box_list[i][0]
        # cent_x=int(box[1]+box[3])
        # cent_y=int(box[0]+box[2])
        cent_x=int(0.5*(x1+x2))
        cent_y=int(0.5*(y1+y2))
        if cent_x<bankuan:
            cent_x=bankuan
        if cent_y<banchang:
            cent_y=banchang
        small_img=image[cent_y-banchang:cent_y+banchang,cent_x-bankuan:cent_x+bankuan,:]
        small_mask = mask[cent_y - banchang:cent_y + banchang, cent_x - bankuan:cent_x + bankuan, :]
        small_img_list.append(small_img)
        small_mask_list.append(small_mask)
    small_mask_list=np.stack(small_mask_list,axis=0)
    small_img_list=np.stack(small_img_list,axis=0)

    return small_img_list,small_mask_list

def load_small_images(dataset_dir):
    Images=[]
    Masks=[]
    boxes=[]
    r_Images=[]
    r_Masks=[]
    # dataset_dir="E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\train"
    Imgspath = glob(dataset_dir + '\\*.jpg')
    for i, a in enumerate(Imgspath):
        Imgname = a.split('\\')[-1]
        M = re.split('[_.]', Imgname)[-2]
        if M != 'Mask':
            image_path = os.path.join(dataset_dir, Imgname)

            image = skimage.io.imread(image_path)
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
        # else:
            Head = ''
            Head2 = ''
            head = a
            routelists = head.split('\\')[:-1]
            Bing=re.split('[_.]', Imgname)[:-1]
            for i, Zifuc in enumerate(routelists):
                if i != 0:
                    Head = os.path.join(Head, Zifuc)
                else:
                    Head = os.path.join(Head, Zifuc) + '\\'
            for zifu in Bing:
                Head2 = Head2 + zifu + '_'
            maskpath = os.path.join(Head, Head2) + "Mask.jpg"
            mask = skimage.io.imread(maskpath)
            mask1=np.where(mask>100,1,0).astype(np.uint8)
            mask=np.expand_dims(mask1,axis=-1)
            bbox=utils.extract_bboxes(mask)
            Images.append(image)
            Masks.append(mask)
            boxes.append(bbox)
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            r_Images.append(image)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
            mask = np.expand_dims(mask, axis=-1)
            r_Masks.append(mask)
    Small_img,Small_mask=get_small(Images,Masks,boxes)
    r_Images=np.array(r_Images)
    r_Masks=np.array(r_Masks)
    i_Images=np.array(Images)
    i_Masks=np.array(Masks)
    return Small_img,Small_mask,r_Images,r_Masks,i_Images,i_Masks,boxes


def models():
    input_small_image = KL.Input(shape=(None,None,3), name="input_small_image")
    # input_small_image = KL.Input(shape= config.SMALL_IMAGE_SHAPE,
    #                              name="input_small_image")
    # input_small_mask = KL.Input(shape=(1,48,144,1), name="input_small_mask")
    rough_mask=build_rough_mask_graph(input_small_image)
    # rough_mask_loss = KL.Lambda(lambda x: rough_mask_loss_gragh(*x), name="rough_mask_loss")(
    #     [rough_mask, rough_aux0, rough_aux1, input_small_mask])
    model = KM.Model(inputs=[input_small_image],outputs=[rough_mask])
    model.summary()

    return model

def train():
    dataset_dir = "E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\train"
    model=models()
    # optimizer = keras.optimizers.SGD(
    #     lr=0.01, momentum=0.99,
    #     clipnorm=1)
    optimizer = keras.optimizers.Adam(lr=0.001, clipvalue=1., clipnorm=1.)
    print("start Training")
    log_dir="E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset"
    checkpoint_path = os.path.join(ROOT_DIR, "mask_rcnn_global_rgb_epoch.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0, save_weights_only=True),
    ]

    model.compile(optimizer=optimizer,loss=total_loss,metrics=[dice_met])
    # model.metrics_tensors.append(loss)
    train_img,train_mask,g_image,g_mask,_,_,_,_=load_small_images(dataset_dir)
    print(train_img.shape)

    model.fit(g_image,g_mask,batch_size=16,epochs=50,callbacks=callbacks,validation_split=0.1)


def show():
    # dataset_dir = "E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\train"
    new_data_dir="E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\wangkainibiaozhu\\previous data version\\val"
    train_img,train_mask,r,g_mask,i_im,i_mask,boxes=load_small_images(new_data_dir)
    box=boxes[5]
    print(box[0])

    c_y=int(0.5*(box[0][0]+box[0][2]))
    c_x = int(0.5 * (box[0][1] + box[0][3]))
    #i init  c:cross, t:total d:dice
    i_img=i_im[5].copy()
    j_img = i_im[5].copy()
    c_img = i_im[5].copy()
    t_img = i_im[5].copy()
    d_img = i_im[5].copy()
    img=train_img[5]
    mask=i_mask[5]
    s_mask=train_mask[5]
    print(s_mask.shape)


    # train_img=img.transpose(2,0,1)
    # train_mask=mask.transpose(2,0,1)
    img = np.expand_dims(img, axis=0)
    model1=models()
    model1.load_weights(ROOT_DIR+"\\mask_rcnn_totalp2_rgb_epoch.h5")
    model2=models()
    model2.load_weights(ROOT_DIR+"\\mask_rcnn_crossp_rgb_epoch.h5")
    model3=models()
    model3.load_weights(ROOT_DIR+"\\mask_rcnn_dicep_rgb_epoch.h5")
    pred_total=model1.predict(img)
    pred_total=np.where(pred_total>0.5,1,0)
    i_pred_total=np.zeros(mask.shape,dtype=np.uint8)
    i_pred_total[c_y-32:c_y+32,c_x-96:c_x+96,:]=pred_total[0]
    print(i_pred_total.shape)
    print(np.max(i_pred_total))

    pred_cross=model2.predict(img)
    pred_cross = np.where(pred_cross > 0.5, 1, 0)
    i_pred_cross=np.zeros(mask.shape,dtype=np.uint8)
    i_pred_cross[c_y-32:c_y+32,c_x-96:c_x+96,:]=pred_cross[0]

    pred_dice=model3.predict(img)
    pred_dice = np.where(pred_dice > 0.5, 1, 0)
    i_pred_dice=np.zeros(mask.shape,dtype=np.uint8)
    i_pred_dice[c_y-32:c_y+32,c_x-96:c_x+96,:]=pred_dice[0]

    plt.figure()
    ax1=plt.subplot(231)
    # plt.title(r"$(a)$")
    # i_pred = np.zeros(mask.shape, dtype=np.uint8)
    # i_pred[c_y - 32:c_y + 32, c_x - 96:c_x + 96, :] = s_mask
    image, i_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(j_img, i_contours, -1, (255, 0, 0), 1)  # img为三通道才能显示轮廓  绿色
    image, i_contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(i_img, i_contours, -1, (255, 0, 0), 1)  # img为三通道才能显示轮廓
    # ax1.imshow(i_img[c_y-50:c_y+50,c_x-100:c_x+100])
    ax1.imshow(j_img)
    plt.xticks([])
    plt.yticks([])
    # plt.subplot(232)
    # plt.title("mask")
    # plt.imshow(mask[:,:,0])
    ax2=plt.subplot(232)
    # plt.title(r"$(b)$")
    iimage, pt_contours, ihierarchy = cv2.findContours(i_pred_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(t_img, pt_contours, -1, (255, 255, 255), 1)  # img为三通道才能显示轮廓 白色
    ax2.imshow(t_img[c_y-50:c_y+50,c_x-100:c_x+100])
    plt.xticks([])
    plt.yticks([])

    ax3=plt.subplot(233)
    # plt.title(r"$(c)$")
    image, pc_contours, hierarchy = cv2.findContours(i_pred_cross, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(c_img, pc_contours, -1, (255, 255, 0), 1)  # img为三通道才能显示轮廓 黄
    ax3.imshow(c_img[c_y-50:c_y+50,c_x-100:c_x+100])
    plt.xticks([])
    plt.yticks([])

    ax4=plt.subplot(234)
    # plt.title(r"$(d)$")
    image, pd_contours, hierarchy = cv2.findContours(i_pred_dice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(d_img, pd_contours, -1, (0, 255, 255), 1)  # img为三通道才能显示轮廓
    ax4.imshow(d_img[c_y-50:c_y+50,c_x-100:c_x+100])
    plt.xticks([])
    plt.yticks([])

    ax6=plt.subplot(236)
    # plt.title(r"$(f)$")
    ii_img = cv2.drawContours(t_img, pd_contours, -1, (0, 255, 255), 1)  # img为三通道才能显示轮廓
    ax6.imshow(t_img[c_y-50:c_y+50,c_x-100:c_x+100])
    plt.xticks([])
    plt.yticks([])

    ax5=plt.subplot(235)
    # plt.title(r"$(e)$")
    # image, pd_contours, hierarchy = cv2.findContours(i_pred_dice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ii_img = cv2.drawContours(i_img, pc_contours, -1, (255, 255, 0), 1)  # img为三通道才能显示轮廓
    ii_img = cv2.drawContours(i_img, pt_contours, -1, (255, 255, 255), 1)  # img为三通道才能显示轮廓
    # ii_img = cv2.drawContours(i_img, pd_contours, -1, (0, 0, 255), 1)  # img为三通道才能显示轮廓
    ax5.imshow(i_img[c_y-50:c_y+50,c_x-100:c_x+100])
    plt.xticks([])
    plt.yticks([])
    plt.show()



def mas_test():
    dataset_dir = "E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\val"
    test_img, test_mask,L_img,L_mask = load_small_images(dataset_dir)
    model = models()
    log_dir = "E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset"
    checkpoint_path = os.path.join(ROOT_DIR, "mask_rcnn_global_rgb_epoch.h5")
    model.load_weights(checkpoint_path)

    # 测试总体
    out_puts = model.predict(L_img, batch_size=16, verbose=1)
    out_puts = np.where(out_puts > 0.5, 1, 0)
    acc_out = dice_acc(test_mask, out_puts)
    senti_out=sensitiv(test_mask,out_puts)
    spe_out=specifiv(test_mask,out_puts)
    print("dice准确率是：",acc_out)
    print("senti准确率是：", senti_out)
    print("speci准确率是：", spe_out)



if __name__=="__main__":
    mode="show"
    assert mode in ["train","show","test"]
    dict_m={"train":train,"show":show,"test":mas_test}
    f=dict_m[mode]
    f()
# train()
# show()
# mas_test()
