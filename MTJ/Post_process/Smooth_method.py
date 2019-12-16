import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.layers.wrappers import Bidirectional
from scipy.io import loadmat
import glob


save_path="myweights_bilstm5.h5"
def load_data(path, train=True,sequence_length=5, split=0.8):
    if train:
        file_name=["S1.xls", "S3.xls", "S7.xls"]
        data = []
        tt = []
    else:
        file_name=["S9.xls"]
        data = []
        tt = []
    for ii in file_name:
        path_list = path+ii
        # 提取数据一列
        df = pd.read_excel(path_list)
        df['x']=df['x'].fillna(0)
        # # 把数据转换为数组
        test_all = np.array(df['x']).astype(float)
        data_all=np.array(df['p']).astype(float)
        # y = loadmat("E:\\project_huo\\zhangyi\\ultrasound_data\\siny.mat")
        # data_all =np.transpose(y['y'], (1, 0))
        # 将数据缩放至给定的最小值与最大值之间，这里是０与１之间，数据预处理
        scaler = MinMaxScaler()
        data_all = scaler.fit_transform(data_all.reshape(-1,1))
        test_all=scaler.fit_transform(test_all.reshape(-1,1))
        print(len(data_all))
        # 构造送入lstm的3D数据：(133, 11, 1)
        for i in range(len(data_all) - sequence_length - 1):
            data.append(data_all[i: i + sequence_length])
        for i in range(len(test_all) - sequence_length - 1):
            tt.append(test_all[i+sequence_length])
    reshaped_data = np.array(data).astype('float64')
    reshaped_test = np.array(tt).astype('float64')
    print(reshaped_data.shape)
    # # 打乱第一维的数据
    # np.random.shuffle(reshaped_data)
    # print('reshaped_data:', reshaped_data[0])
    # 这里对133组数据进行处理，每组11个数据中的前10个作为样本集：(133, 10, 1)
    x = reshaped_data
    y=reshaped_test
    # print('samples:', x.shape)
    # # 133组样本中的每11个数据中的第11个作为样本标签
    # y = reshaped_data[:, -1]
    # print('labels:', y.shape)
    # 构建训练集(训练集占了80%)
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x=(x)
    # 构建测试集(原数据的后20%)
    # test_x = x[split_boundary:]
    # 训练集标签
    train_y=(y)
    # 测试集标签
    # test_y = y[split_boundary:]
    # 返回处理好的数据
    print("训练集的大小",train_x.shape)
    print("训练标签的大小", train_y.shape)
    return train_x, train_y, scaler


# 模型建立
def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model

def build_bidirection():
    model = Sequential()
    model.add(Bidirectional(LSTM(input_dim=1, output_dim=20, return_sequences=True), merge_mode='concat',input_shape=(5, 1)))
    model.add(Bidirectional(LSTM(40, return_sequences=False), merge_mode='concat'))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def train_model(train_x, train_y, test_x, test_y,train=False):
    model = build_bidirection()
    callback=[keras.callbacks.ModelCheckpoint('{}_bilstm5.h5'.format('myweights'),
                                    monitor='val_loss',
                                    verbose=0, save_best_only=False),]
    try:
        if train:
            model.fit(train_x, train_y, batch_size=512, epochs=30, validation_split=0.1,callbacks=callback)
        else:
            print("pre to load weights")
            model.load_weights(save_path)
            print("weights_loaded")
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size,))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print('predict:\n', predict)
    print('test_y:\n', test_y)
    # 预测的散点值和真实的散点值画图
    try:
        fig1 = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict, test_y


if __name__ == '__main__':
    # 加载数据
    train_x, train_y, scaler = load_data('E:\\project_huo\\zhangyi\\ultrasound_data\\')
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    test_x, test_y, scaler = load_data('E:\\project_huo\\zhangyi\\ultrasound_data\\',train=False)
    # 模型训练

    predict_y, test_y = train_model(train_x, train_y, test_x, test_y,train=False)
    # 对标准化处理后的数据还原
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    test_y = scaler.inverse_transform(test_y)
    # 把预测和真实数据对比
    fig2 = plt.figure(2)
    plt.plot(predict_y, 'g:')
    plt.plot(test_y, 'r-')
    plt.legend(['predict', 'true'])
    plt.show()
