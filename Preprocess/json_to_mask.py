''''
将手动标注的label me 生成的jason 文件提取为mask图片。
初始文件已经在cmd中处理，成为文件包格式。
每个文件包内的mask是0,1二值图像。
'''


from glob import glob
import numpy as np
import re
import shutil
import skimage.io
import PIL.Image


DataPath="E:\\project_huo\\zhangyi\\张熠_毕设论文实验整理\\DL方法实验\\dataset\\wangkainibiaozhu"

json_list=glob(DataPath+"\\muscle_val_json\\*_json")
for labels in json_list:
    llist=labels.split('\\')[-1]
    pnamelist=llist.split('_')
    label=PIL.Image.open(labels+"\\label.png")
    la_arr=np.array(label)
    la_arr=np.where(la_arr>0,255,0)
    skimage.io.imsave(DataPath+"\\val_labels\\"+pnamelist[0]+'_'+pnamelist[1]+'_'+pnamelist[2]+'_Mask.jpg',la_arr)
