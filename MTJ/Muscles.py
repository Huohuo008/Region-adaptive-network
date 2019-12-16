import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import csv
from glob import glob
import pandas as pd
import re
import cv2
import xlrd
import pylab as pl
from skimage import measure
import random
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import RAn
sys.path.append(ROOT_DIR)  # To find local version of the library

from ran.config import Config
from ran import model as modellib, utils
from ran import visualize
# Path to trained weights file
#加载训练权重
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# aa=DEFAULT_LOGS_DIR.split('\\')#调试

############################################################
#  Configurations
############################################################


class MuscleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "muscle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + muscle

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    ROUGH_IMG_DIM = 48
    ROUGH_IMG_DIM2 = 144
    RPN_ANCHOR_RATIOS = [1, 2, 3]
    SMALL_IMAGE_SHAPE = (64, 192, 3)
    SMALL_MASK_SHAPE = (64, 192, 1)
    EXTEND_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2, 0.5, 0.5])
    SMALL_CHECK_PATH = "mask_rcnn_totalp2_rgb_epoch.h5"
    SMALL_CHECK_PATH1 = "mask_rcnn_crossp_rgb_epoch.h5"
    SMALL_CHECK_PATH2 = "mask_rcnn_dicep_rgb_epoch.h5"



############################################################
#  Dataset
############################################################

class MuscleDataset(utils.Dataset):

    def load_muscle(self, dataset_dir, subset):
        """Load a subset of the Muscle dataset.
        dataset_dir: Root directory of the dataset.E:\project_huo\zhangyi\张熠_毕设论文实验整理\DL方法实验\dataset
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("muscle", 1, "muscle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)     #连接路径名称
        Imgspath=glob(dataset_dir+'\\*.jpg')
        Zuobiaopath=dataset_dir+'\\zuobiao.xlsx'
        Zuobiao=pd.read_excel(Zuobiaopath)
        for i,a in enumerate(Imgspath):
            Imgname = a.split('\\')[-1]
            M = re.split('[_.]', Imgname)[-2]
            if M != 'Mask':
                zb_name=Imgname.split('.')[:-1]
                pre_axis=Zuobiao.ix[zb_name]  #行索引
                if pre_axis is not None:
                    image_path = os.path.join(dataset_dir, Imgname)
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]
                    zuobiao_y=int(pre_axis['y'])
                    zuobiao_x=int(pre_axis['x'])
                    #参与构成数据对应字典列表的一些项目，包括类别，image_id(路径)，path，关键坐标。
                    #后续在load这些信息可以通过查询路径，找到对应。
                    self.add_image(
                        "muscle",
                        image_id=Imgname,  # use file name as a unique image id
                        path=image_path,
                        zuo_y=zuobiao_y, zuo_x=zuobiao_x,
                        width=width, height=height,
                        polygons={})


    def load_mask(self,image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a muscle dataset image, delegate to parent class.
        #调用父类的大字典
        image_info = self.image_info[image_id]
        if image_info["source"] != "muscle":   #基类的load——mask返回一个空mask
            return super(self.__class__, self).load_mask(image_id)
        #路径字符串常规操作
        info=self.image_info[image_id]
        Bing= re.split('[_.]', info['id'])[:-1]
        Head=''
        Head2=''
        head=info['path']
        routelists=head.split('\\')[:-1]
        #返回路径相关的mask。
        for i,Zifuc in enumerate(routelists):
            if i!=0:
                Head=os.path.join(Head,Zifuc)
            else:
                Head=os.path.join(Head,Zifuc)+'\\'
        for zifu in Bing:
            Head2=Head2+zifu+'_'

        maskpath=os.path.join(Head,Head2)+"Mask.jpg"
        mask=skimage.io.imread(maskpath)
        mask=np.where(mask>100, 1, 0)
        mask=np.expand_dims(mask,axis=-1)
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "muscle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def create_csv(path):
    #创建以一定头格式的csv文件，用来存放数据和后续数据处理。
    with open(path,'w',newline="") as f:
        csv_write = csv.writer(f,dialect='excel')
        csv_head = ["up_y","left_x","down_y","right_x","y","x","kuandu",'h_x',"num_zhen"]
        csv_write.writerow(csv_head)

def write_csv(path,list):
    #横向存储，csv是含有逗号的文件，传入格式必须是含逗号的列表格式。
    with open(path,'a+',newline="") as f:
        csv_write = csv.writer(f)
        data_row = list
        csv_write.writerow(data_row)


def train(model):
    """Train the model."""
    # Training dataset.
    # 构建一个大的dataset字典。
    dataset_train = MuscleDataset()
    dataset_train.load_muscle(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    # 构建一个大的dataset字典。
    dataset_val = MuscleDataset()
    dataset_val.load_muscle(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def BuildGaborKernels(ksize = 5,lamda = 1.5,sigma = 1.0):
    '''
    @description:生成多尺度，多方向的gabor特征
    @参数参考opencv
    @return:多个gabor卷积核所组成的
    @author:SXL
    '''
    filters = []
    for theta in np.array([0,np.pi/4, np.pi/2,np.pi*3/4]):
        kern = cv2.getGaborKernel((ksize,ksize),sigma,
                theta,lamda,0.5,0,ktype=cv2.CV_32F)
        filters.append(kern)
    pl.figure(1)
    for temp in range(len(filters)):
        pl.subplot(4, 4, temp + 1)
        pl.imshow(filters[temp], cmap='gray')
    pl.show()
    return filters

def GaborFeature(image):
    '''
    @description:提取字符图像的gabor特征
    @image:灰度字符图像
    @return:滤波后的图
    @author:SXL
    '''
    # retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    kernels = BuildGaborKernels(ksize = 7,lamda = 8,sigma = 4)
    dst_imgs = []
    for kernel in kernels:
        img = np.zeros_like(image)
        tmp = cv2.filter2D(image,cv2.CV_8UC3,kernel)
        img = np.maximum(img,tmp,img)
        dst_imgs.append(img)

    pl.figure(2)
    for temp in range(len(dst_imgs)):
        pl.subplot(4,1,temp+1) #第一个4为4个方向，第二个4为4个尺寸
        pl.imshow(dst_imgs[temp], cmap='gray' )
    pl.show()
    return dst_imgs

def GetImageFeatureGabor(image):
    '''
    @description:提取经过Gabor滤波后字符图像的网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:SXL
    '''
    #----------------------------------------
    #图像大小归一化
    image = cv2.resize(image,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]
    resImg=GaborFeature(image)
    #-----Gabor滤波--------------------------

    #-----对滤波后的图逐个网格化提取特征-------
    feature = np.zeros(64)        # 定义特征向量
    grid_size=4
    imgcount=0
    for img in resImg:
        # 二值化
        retval, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        imgcount+=1

        # 计算网格大小
        grid_h = binary.shape[0] / grid_size
        grid_w = binary.shape[1] / grid_size
        for j in range(grid_size):
            for i in range(grid_size):
                # 统计每个网格中黑点的个数
                grid = binary[int(j * grid_h):int((j + 1) * grid_h), int(i * grid_w):int((i + 1) * grid_w)]
                feature[j * grid_size + i+(imgcount-1)*grid_size*grid_size] = grid[grid == 0].size

    return feature


def square_detect(image, boxes, masks, class_ids,y,x, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,show_point=True,
                      colors=None, captions=None,count=-1):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    import cv2
    N = boxes.shape[0]
    # if not N:
    #     print("\n*** Nothing to Save ***l2 \n")
    # else:
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]   #保证框和mask一一对应。
    # write_csv(count, boxes)
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (int(x1), int(y2)), (int(x2), int(y1)), (0, 255, 0), 3)
        if not captions:
            score = scores[i] if scores is not None else None
            caption = "{:.3f}".format(score)
        else:
            caption = captions[i]
        cv2.putText(masked_image, caption, (int(x1), int(y1 - 6)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255))
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        if show_point:
            masked_image[int(y-5):int(y+5),int(x-5):int(x+5),:]=255

    return masked_image

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def simple_init(mask,y1,y2,x1,x2):
    s_mask = mask[y1 - 1:y2 + 1, x1 - 1:x2 + 1]
    testline1 = x2 - x1 - 40
    cite1 = s_mask[:, testline1]
    tidu1 = np.diff(cite1.transpose(1, 0))
    num1 = np.sum(tidu1, axis=1)
    print(num1)
    testline2 = 30
    cite2 = s_mask[:, testline2]
    tidu2 = np.diff(cite2.transpose(1, 0))
    num2 = np.sum(tidu2, axis=1)
    print(num2)
    return num1,num2

def get_refine_mask(layer,detection_result,image,layer_name = "seg_branch"):
    r = detection_result
    booox = r["rois"][0]
    y_center = int(0.5 * (booox[0] + booox[2]))
    x_center = int(0.5 * (booox[1] + booox[3]))
    y_refine = 0.5 * (r['y'] + y_center)
    x_refine = 0.5 * (r['x'] + x_center)
    if x_refine<96:
        x_refine=96
    expanded_img2 = image[int(y_refine) - 32:int(y_refine) + 32, int(x_refine) - 96:int(x_refine) + 96, :].copy()
    expanded_img2 = np.expand_dims(expanded_img2, axis=0)
    expanded_mask1 = layer.predict(expanded_img2)  # total2_keypoint
    expanded_mask1=np.where(expanded_mask1>0.5,1,0)

    return expanded_img2[0],expanded_mask1[0,:,:,0],y_refine,x_refine

def get_xiankuan(mask,k_x):
    t1=np.sum(mask[:,int(k_x)-20,0],axis=0)
    t2 = np.sum(mask[:, int(k_x) - 30,0], axis=0)
    # t3 = np.sum(mask[:, k_x - 40], axis=0)
    if abs(t1-t2)<5:
        kuandu=int((t1+t2)//2)
    else:
        kuandu=np.min([t1,t2])
    return kuandu

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # draw the box of the image
        splash = square_detect(image,r['rois'], r['masks'], r['class_ids'],
                            class_names=2, scores=r['scores'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

    elif video_path:
        import cv2
        input_path='S79301.csv'
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        #创一个带指定表头的表格文件
        create_csv(input_path)
        count = 0
        success = True
        flag=0
        last_list=[]
        layer_name="seg_branch"
        layer = model.get_model(layer_name, config.SMALL_CHECK_PATH)  # 总的
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                N = r['rois'].shape[0]

                if flag==0:   #尚未出现首帧时，读取判断是不是首帧。
                    if not N:
                        print("\n*** Nothing to Save ***11 \n")
                        splash=image
                    else:
                        y1,x1,y2,x2=r['rois'][0][0:4]
                        if x2-x1>70 and y2-y1>10:
                            refined_img, refined_mask, c_y, c_x = get_refine_mask(layer, r, image)
                            zao_mask1 = np.zeros(image.shape[:2], dtype=np.bool)
                            zao_mask1[int(c_y) - 32:int(c_y) + 32, int(c_x) - 96:int(c_x) + 96] = refined_mask
                            zao_mask1=np.expand_dims(zao_mask1,axis=-1)
                            k_mode = utils.get_harries(refined_img, refined_mask)
                            print("首帧出现了")
                            dy, dx = utils.get_nearest(k_mode)  # 小方块里面的坐标
                            if dy == -100 and dx == -100:
                                h_x = int(r['x'])
                            else:
                                h_x =int(c_x - 96 + dx)
                            splash = square_detect(image, r['rois'], zao_mask1, r['class_ids'],r['y'],r['x'],class_names=2, scores=r['scores'], count=count)
                            kuandu=get_xiankuan(zao_mask1,r['x'])
                        # RGB -> BGR to save image to video
                            splash = splash[..., ::-1]
                            box_list = r['rois'][0]
                            pro_list = box_list.tolist()
                            pro_list.append(float(r['y']))
                            pro_list.append(float(r['x']))
                            pro_list.append(kuandu)
                            pro_list.append(h_x)
                            pro_list.append(count)
                            write_csv(input_path, pro_list)
                            last_list=pro_list
                            flag=1
                        else:
                            print("大小不满足特点，非首帧")
                            flag = 0
                            splash = image

                elif flag:
                   # Color splash
                    if not N:
                        print("此帧之前未检测出，补帧中。。。。。。")
                        full_mask= np.zeros((height, width,1))
                        full_mask2 = np.zeros((height, width, 1))
                        pro_list=last_list
                        k_y=int(pro_list[4])
                        k_x=int(pro_list[5])
                        if k_x<config.SMALL_IMAGE_SHAPE[1]//2:
                            init_img = image[k_y - config.SMALL_IMAGE_SHAPE[0]//2:k_y + config.SMALL_IMAGE_SHAPE[0]//2, 0:config.SMALL_IMAGE_SHAPE[1], :]
                            expanded_img = np.expand_dims(init_img, axis=0)
                            expanded_mask = layer.predict(expanded_img)
                            full_mask[k_y - config.SMALL_IMAGE_SHAPE[0]//2:k_y + config.SMALL_IMAGE_SHAPE[0]//2, 0:config.SMALL_IMAGE_SHAPE[1], :] = expanded_mask[0]
                            kuangzuo=0 #框左坐标

                        else:
                            if k_x+config.SMALL_IMAGE_SHAPE[1]//2>480:
                                k_x=479-config.SMALL_IMAGE_SHAPE[1]//2
                            init_img=image[k_y-config.SMALL_IMAGE_SHAPE[0]//2:k_y+config.SMALL_IMAGE_SHAPE[0]//2,k_x-config.SMALL_IMAGE_SHAPE[1]//2:k_x+config.SMALL_IMAGE_SHAPE[1]//2,:]
                            expanded_img=np.expand_dims(init_img,axis=0)
                            expanded_mask=layer.predict(expanded_img)
                            expanded_mask = np.where(expanded_mask>0.5,1,0)
                            full_mask[k_y-config.SMALL_IMAGE_SHAPE[0]//2:k_y+config.SMALL_IMAGE_SHAPE[0]//2,k_x-config.SMALL_IMAGE_SHAPE[1]//2:k_x+config.SMALL_IMAGE_SHAPE[1]//2,:]=expanded_mask[0]
                            kuangzuo=k_x-config.SMALL_IMAGE_SHAPE[1]//2

                        init_mask=expanded_mask[0,:,:,0]
                        k_mode=utils.get_harries(init_img,init_mask)
                        dy,dx=utils.get_nearest(k_mode)
                        if dy==-100 and dx==-100:
                            print("补帧失败")
                            splash=image
                            pro_list[8] = count
                            write_csv(input_path, pro_list)
                            last_list = pro_list
                        else:
                            print("补帧成功")
                            pro_list[1]=kuangzuo
                            pro_list[3]=kuangzuo+192
                            pro_list[5]=kuangzuo+dx-20
                            kuandu=get_xiankuan(full_mask,pro_list[5])  ##################dx
                            pro_list[6] =kuandu
                            h_x=int(kuangzuo+dx)
                            pro_list[7]=h_x
                            pro_list[8]=count
                            y1,x1,y2,x2=[int(i) for i in pro_list[0:4]]
                            full_mask2[y1:y2,x1:x2,:]= full_mask[y1:y2,x1:x2,:]
                            write_csv(input_path, pro_list)
                            last_list=pro_list
                            box_array = np.expand_dims(np.array(pro_list[0:4]), axis=0)
                            id_array = np.array(['Muscles'])
                            score_array = np.array([1])
                            splash = square_detect(image, box_array, full_mask2, id_array, pro_list[4], pro_list[5], class_names=2, scores=score_array, count=count)
                    else:
                        refined_img,refined_mask,c_y,c_x = get_refine_mask(layer, r, image)
                        k_mode = utils.get_harries(refined_img, refined_mask)
                        dy, dx = utils.get_nearest(k_mode)
                        zao_mask1 = np.zeros(image.shape[:2], dtype=np.bool)
                        zao_mask1[int(c_y) - 32:int(c_y) + 32, int(c_x) - 96:int(c_x) + 96] = refined_mask
                        zao_mask1 = np.expand_dims(zao_mask1, axis=-1)
                        splash = square_detect(image,r['rois'], zao_mask1, r['class_ids'],r['y'],r['x'],
                                    class_names=2, scores=r['scores'],count=count)
                        # RGB -> BGR to save image to video
                        kuandu=get_xiankuan(zao_mask1,r['x'])
                        splash = splash[..., ::-1]
                        box_list = r['rois'][0]
                        pro_list=box_list.tolist()
                        pro_list.append(float(r['y']))
                        pro_list.append(float(r['x']))
                        pro_list.append(kuandu)
                        if dy == -100 and dx == -100:
                            h_x = int(r['x'])
                        else:
                            h_x = int(c_x - 96 + dx)
                        pro_list.append(h_x)
                        pro_list.append(count)
                        write_csv(input_path,pro_list)
                        last_list=pro_list
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def refine_detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # draw the box of the image
        splash = square_detect(image,r['rois'], r['masks'], r['class_ids'],
                            class_names=2, scores=r['scores'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        num_frames=int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        slice_info=pd.read_csv("S2_bi5.csv")
        count = 0
        success = True
        layer_name = "seg_branch"
        layer=model.get_model(layer_name)
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                if count<(num_frames-len(slice_info)):  #未到首帧，输出原图
                    splash=image
                else:
                    full_mask2=np.zeros((height,width,1))
                    full_mask = np.zeros((height, width,1))
                    slice_series=slice_info.iloc[count-5]
                    y1,x1,y2,x2=slice_series[["up_y", "left_x", "down_y", "right_x"]]
                    key_y,key_x=slice_series[[ "y", "x"]]
                    small_image=np.expand_dims(image[key_y-config.SMALL_IMAGE_SHAPE[0]//2:key_y+config.SMALL_IMAGE_SHAPE[0]//2,key_x-config.SMALL_IMAGE_SHAPE[1]//2:key_x+config.SMALL_IMAGE_SHAPE[1]//2,:],axis=0)
                    small_mask=layer.predict(small_image)
                    full_mask2[key_y-config.SMALL_IMAGE_SHAPE[0]//2:key_y+config.SMALL_IMAGE_SHAPE[0]//2,key_x-config.SMALL_IMAGE_SHAPE[1]//2:key_x+config.SMALL_IMAGE_SHAPE[1]//2,:]=small_mask[0]
                    full_mask[y1:y2,x1:x2,:]=full_mask2[y1:y2,x1:x2,:]
                    box_array=np.expand_dims(np.array([y1,x1,y2,x2]),axis=0)
                    id_array=np.array(['Muscles'])
                    score_array=np.array([1])
                    splash = square_detect(image, box_array, full_mask, id_array, key_y, key_x,
                                           class_names=2, scores=score_array, count=count)
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Muscles.')
    parser.add_argument("command",
                        metavar="command",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="dataset",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="\weights\\",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MuscleConfig()
    else:
        class InferenceConfig(MuscleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                 video_path=args.video)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

