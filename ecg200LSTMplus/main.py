# Input data files are available in the "../dataset/" directory.

# import os
# for dirname, _, filenames in os.walk('./dataset'):
#     for filename in filenames:
#         # 连接两个或更多的路径名组件
#         print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(42)

import matplotlib.pyplot as plt
np.random.seed(42)
import tensorflow as tf
import tensorflow.keras as keras

def DATA_ACQUISITION():
    mit_test_data = pd.read_csv("dataset/mitbih_test.csv", header=None)
    mit_train_data = pd.read_csv("dataset/mitbih_train.csv", header=None)
    return mit_test_data, mit_train_data
# Any results you write to the current directory are saved as output.

def PRODUCE_BALANCED_DATASET():
    # There is a huge difference in the balanced of the classes.
    # Better choose the resample technique more than the class weights for the algorithms.
    from sklearn.utils import resample

    mit_test_data, mit_train_data = DATA_ACQUISITION()

    # classify
    df_1 = mit_train_data[mit_train_data[187] == 1]
    df_2 = mit_train_data[mit_train_data[187] == 2]
    df_3 = mit_train_data[mit_train_data[187] == 3]
    df_4 = mit_train_data[mit_train_data[187] == 4]
    df_0 = (mit_train_data[mit_train_data[187] == 0]).sample(n=20000, random_state=42) #  #从中随机获取42个元素，作为一个片断

    # 增大采样频率
    # minority
    # sample with replacement
    # to match majority class
    # reproducible results
    df_1_upsample = resample(df_1, replace=True, n_samples=20000, random_state=123)
    df_2_upsample = resample(df_2, replace=True, n_samples=20000, random_state=124)
    df_3_upsample = resample(df_3, replace=True, n_samples=20000, random_state=125)
    df_4_upsample = resample(df_4, replace=True, n_samples=20000, random_state=126)

    train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

    df_11 = mit_test_data[mit_test_data[187] == 1]
    df_22 = mit_test_data[mit_test_data[187] == 2]
    df_33 = mit_test_data[mit_test_data[187] == 3]
    df_44 = mit_test_data[mit_test_data[187] == 4]
    df_00 = (mit_test_data[mit_test_data[187] == 0]).sample(replace=True, n=20000, random_state=42)

    df_11_upsample = resample(df_11, replace=True, n_samples=20000, random_state=123)
    df_22_upsample = resample(df_22, replace=True, n_samples=20000, random_state=124)
    df_33_upsample = resample(df_33, replace=True, n_samples=20000, random_state=125)
    df_44_upsample = resample(df_44, replace=True, n_samples=20000, random_state=126)

    test_df = pd.concat([df_00, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])
    # 一种查看表格某列中有多少个不同值的快捷方法，并计算每个不同值有在该列中有多少重复值。
    equilibre = train_df[187].value_counts()
    print(equilibre)

    print("ALL Train data")
    print("Type\tCount")
    print((mit_train_data[187]).value_counts())
    print("-------------------------")
    print("ALL Test data")
    print("Type\tCount")
    print((mit_test_data[187]).value_counts())

    print("ALL Balanced Train data")
    print("Type\tCount")
    print((train_df[187]).value_counts())
    print("-------------------------")
    print("ALL Balanced Test data")
    print("Type\tCount")
    print((test_df[187]).value_counts())

    return train_df, test_df


def LSTMModel():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM
    from keras.layers.embeddings import Embedding
    # fix random seed for reproducibility
    np.random.seed(7)
    # Separate features and targets

    train_df, test_df = PRODUCE_BALANCED_DATASET()
    mit_test_data, mit_train_data = DATA_ACQUISITION()

#    from keras.utils import to_categorical

    print("--- X ---")
    # X = mit_train_data.loc[:, mit_train_data.columns != 187]
    X = train_df.loc[:, mit_train_data.columns != 187]
    print(X.head())
    print(X.info())

    print("--- Y ---")
    # y = mit_train_data.loc[:, mit_train_data.columns == 187]
    y = train_df.loc[:, mit_train_data.columns == 187]
    y = keras.utils.to_categorical(y)

    print("--- testX ---")
    # testX = mit_test_data.loc[:, mit_test_data.columns != 187]
    testX = test_df.loc[:, mit_test_data.columns != 187]
    print(testX.head())
    print(testX.info())

    print("--- testy ---")
    # testy = mit_test_data.loc[:, mit_test_data.columns == 187]
    testy = test_df.loc[:, mit_test_data.columns == 187]
    testy = keras.utils.to_categorical(testy)

    # create the model.
    # History对象即为fit方法的返回值,可以使用history中的存储的acc和loss数据对训练过程进行可视化画图
    from keras.callbacks import History
    history = History()
    embedding_vecor_length = 187

    # test the code
    model = Sequential()
    # 嵌入层将正整数（下标）转换为具有固定大小的向量
    # input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
    # output_dim：大于0的整数，代表全连接嵌入的维度
    # 输入序列的长度，就是向量的长度
    #model.add(Embedding(100000, embedding_vecor_length, input_length=187))
    model.add(LSTM(187))
    model.add(Dense(5, activation='softmax'))
    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X, y, validation_data=(testX, testy), epochs=3, batch_size=8)

    #Dropout is a powerful technique for combating overfitting in your LSTM models
    # model = Sequential()
    # model.add(Embedding(1000, embedding_vecor_length, input_length=187))
    # model.add(LSTM(50, dropout=0.001, recurrent_dropout=0.001))
    # model.add(Dense(5, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    # history = model.fit(X, y, validation_data=(testX, testy), epochs=50, batch_size=128)

    ## SAVE MODEL ##
    # serialize model to JSON
    model_json = model.to_json()
    with open("1model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("1model.h5")
    print("Saved model to disk")

    return model, testX, testy, history

def EvaluateModel():
    model, testX, testy, history = LSTMModel()
    mse, acc = model.evaluate(testX, testy)
    print('mean_squared_error :', mse)
    print('accuracy:', acc)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # The history for the validation dataset is labeled test by convention as it is indeed a test dataset for the model.
    # The plots can provide an indication of useful things about the training of the model, such as:
    # *It’s speed of convergence over epochs (slope).
    # *Whether the model may have already converged (plateau of the line).
    # *Whether the mode may be over-learning the training data (inflection for validation line)

def Αccuracy_and_prediction_scores():
    model, testX, testy, history = LSTMModel()
    y_pred = model.predict(testX, batch_size=1000)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error
    print(classification_report(testy.argmax(axis=1), y_pred.argmax(axis=1)))


if __name__ == '__main__':
    EvaluateModel()
    Αccuracy_and_prediction_scores()
