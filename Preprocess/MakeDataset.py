'''
将已有的标注文件名字解析，在视频中找到对应的帧。解析文件。
复制到相应对应的文件夹作为训练集和验证集。
'''


import cv2
import numpy
from glob import glob
import numpy as np
import re
import shutil

DataPath="E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\wangkainibiaozhu"
VideosPath="E:\\project_huo\\zhangyi\\ultrasound_data\\"
sublist=[str(a) for a in range(1,10)]
print(len(sublist))
newFiletrain = DataPath + 'test_original'
for Mtrain in sublist:
    MasksPath=glob(DataPath+'train_and_val_original\\'+'S'+Mtrain+'_'+'*Mask.jpg')
    if len(MasksPath) == 0:
        print("不包含该系列的mask")
    elif len(MasksPath)>0:
        Videopath=VideosPath+'S'+Mtrain+'_pG.avi'
        # assert VideosPath in Videos
        print("正在读取.."+Videopath)
        cap=cv2.VideoCapture(Videopath)
        for Mask in MasksPath:
            Maskname=Mask.split('\\')[-1]
            M=int(re.split('[_.]',Maskname)[-3])
            cap.set(cv2.CAP_PROP_POS_FRAMES, M)
            success, frame = cap.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                newMask=DataPath+'train'+'\\S'+Mtrain +'_Img_'+'%04d'%M+'.jpg'
                cv2.imencode('.jpg', gray)[1].tofile(newMask)
                aa=gray.shape
                shutil.copy(Mask,DataPath+'train')
            else:
                print("读取文件失败")
        cap.release()