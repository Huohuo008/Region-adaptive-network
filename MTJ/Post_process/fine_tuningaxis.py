import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from MTJ.Post_process.Smooth_method import build_model,build_bidirection
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from scipy.io import loadmat
import glob
import csv
import re

save_path="myweights_lstm5.h5"
bi_save_path="myweights_bilstm5.h5"

def load_testdata(path,sequence_length=5):
    df = pd.read_csv(path)
    train_list=[]
    scaler_list=[]
    # df['x'] = df['x'].fillna(0)
    # # 把数据转换为数组
    test_list=["up_y","left_x","down_y","right_x","y","x"]
    # test_all = np.array(df[test_list]).astype(float)
    for project in test_list:
        data = []
        data_all = np.array(df[project]).astype(float)
        # y = loadmat("E:\\project_huo\\zhangyi\\ultrasound_data\\siny.mat")
        # data_all =np.transpose(y['y'], (1, 0))
        # 将数据缩放至给定的最小值与最大值之间，这里是０与１之间，数据预处理
        scaler = MinMaxScaler()
        data_all = scaler.fit_transform(data_all.reshape(-1, 1))
        # test_all = scaler.fit_transform(test_all.reshape(-1, 1))
        print(len(data_all))
        # 构造送入lstm的3D数据：(133, 11, 1)
        for i in range(len(data_all) - sequence_length - 1):
            data.append(data_all[i: i + sequence_length])
        # for i in range(len(test_all) - sequence_length - 1):
        #     tt.append(test_all[i + sequence_length])
        reshaped_data = np.array(data).astype('float64')
        # reshaped_test = np.array(tt).astype('float64')
        print(reshaped_data.shape)
        # # 打乱第一维的数据
        # np.random.shuffle(reshaped_data)
        # print('reshaped_data:', reshaped_data[0])
        # 这里对133组数据进行处理，每组11个数据中的前10个作为样本集：(133, 10, 1)
        # x = reshaped_data
        # y = reshaped_test
        # print('samples:', x.shape)
        # # 133组样本中的每11个数据中的第11个作为样本标签
        # y = reshaped_data[:, -1]
        # print('labels:', y.shape)
        # 构建训练集(训练集占了80%)
        # split_boundary = int(reshaped_data.shape[0] * split)
        train_list.append(reshaped_data)
        scaler_list.append(scaler)
    # 构建测试集(原数据的后20%)
    # test_x = x[split_boundary:]
    # 训练集标签
    # train_y = (y)
    # 测试集标签
    # test_y = y[split_boundary:]
    # 返回处理好的数据
    print("测试集的大小", len(train_list))
    # print("训练标签的大小", train_y.shape)
    return train_list, scaler_list

def create_csv(path):
    with open(path,'w',newline="") as f:
        csv_write = csv.writer(f,dialect='excel')
        csv_head = ["up_y","left_x","down_y","right_x","y","x"]
        csv_write.writerow(csv_head)

def lstm_axis_test(path,to):
    model=build_model()
    model.load_weights(save_path)
    train_list,scaler_list=load_testdata(path)
    # create_csv("S2_1.csv")
    df=pd.DataFrame()
    df_list = ["up_y", "left_x", "down_y", "right_x", "y", "x"]
    for num,data in enumerate(train_list):
        predict_data=model.predict(data)
        predicted=predict_data
        scaler=scaler_list[num]
        aa= scaler.inverse_transform([i for i in predicted])
        df[df_list[num]] = [int(b) for b in aa]
        # df=df.applymap(f)
    print("change successfully")
    df.to_csv(to, index=False, sep=',')

def bilstm_axis_test(path,to):
    model=build_bidirection()
    model.load_weights(bi_save_path)
    train_list,scaler_list=load_testdata(path)
    # create_csv("S2_1.csv")
    df=pd.DataFrame()
    df_list = ["up_y", "left_x", "down_y", "right_x", "y", "x"]
    for num,data in enumerate(train_list):
        predict_data=model.predict(data)
        predicted=predict_data
        scaler=scaler_list[num]
        aa= scaler.inverse_transform([i for i in predicted])
        df[df_list[num]] = [int(b) for b in aa]
        # df=df.applymap(f)
    print("change successfully")
    df.to_csv(to, index=False, sep=',')


def tri_mid(array):
    array2=np.zeros(array.shape)
    array2[0]=0.5*(array[0]+array[1])
    array2[1] = 0.5 * (array[1] + array[2])
    array2[-1]=0.5*(array[-1]+array[-2])
    array2[-2] = 0.5 * (array[-2] + array[-3])
    for z in range(2,array.shape[0]-2):
        array2[z]=(array[z-2]+array[z-1]+array[z]+array[z+1]+array[z+2])/5
    return array2

def median_axis_test(path,to):
    init_axis = pd.read_csv(path)
    df_list = ["up_y", "left_x", "down_y", "right_x", "y", "x"]
    df=pd.DataFrame()
    for df_project in df_list:
        axis_df=init_axis[df_project]
        after_axis=tri_mid(np.array(axis_df))
        df[df_project]=[int(b) for b in after_axis]
    print("change successfully")
    df.to_csv(to, index=False, sep=',')





if __name__ =="__main__":
    name='LSTM'
    init_path='C:\\Users\\HP\\Desktop\\S39301.csv'

    dict_f={"median":("S2_3.csv",median_axis_test),'LSTM':("C:\\Users\\HP\\Desktop\\S3_1.csv",lstm_axis_test),"Bi-lstm":("S2_bi5.csv",bilstm_axis_test)}
    cunchulujing,f=dict_f[name]
    # lstm_axis_test()
    # median_axis_test()
    f(init_path,cunchulujing)