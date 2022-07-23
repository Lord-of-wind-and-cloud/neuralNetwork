import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils



import matplotlib.pyplot as plt

def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = 38
    f = open(os.path.join('output', dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join('output', dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join('output', dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        train_data = preprocess(train_data)
        test_data = preprocess(test_data)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')

    return df

def build_model(x_train, y_train, x_test, y_test):
    # create the model.
    # History对象即为fit方法的返回值,可以使用history中的存储的acc和loss数据对训练过程进行可视化画图
    from keras.callbacks import History
    history = History()

    # test the code
    model = Sequential()
    # 嵌入层将正整数（下标）转换为具有固定大小的向量
    # input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
    # output_dim：大于0的整数，代表全连接嵌入的维度
    # 输入序列的长度，就是向量的长度
    model.add(Embedding(100000, 38, input_length=38))
    model.add(LSTM(38))
    model.add(Dense(2, activation='softmax'))
    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=256, batch_size=32)

    ## SAVE MODEL ##
    # serialize model to JSON
    model_json = model.to_json()
    with open("1model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("1model.h5")
    print("Saved model to disk")

    return model, history

def EvaluateModel(x_train, y_train, x_test, y_test):
    model, history = build_model(x_train, y_train, x_test, y_test)
    mse, acc = model.evaluate(x_test, y_test)
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

def Αccuracy_and_prediction_scores(x_train, y_train, x_test, y_test):
    model, history = build_model(x_train, y_train, x_test, y_test)

    y_pred = model.predict(x_test, batch_size=1000)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error

    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))



if __name__ == '__main__':
    (x_train, _), (x_test, y_test) = get_data("machine-1-1", max_train_size = None, max_test_size = None, train_start=0, test_start=0)
    y_train = np.zeros(28479, None)
    y_train[15849-1:16368-1] = 1
    y_train[16963-1:17517-1] = 1
    y_train[18071-1:18528-1] = 1
    y_train[19367-1:20088-1] = 1
    y_train[20786-1:21195-1] = 1
    y_train[24679-1:24682-1] = 1
    y_train[26114-1:26116-1] = 1
    y_train[27554-1:27556-1] = 1
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    EvaluateModel(x_train, y_train, x_test, y_test)
    Αccuracy_and_prediction_scores(x_train, y_train, x_test, y_test)
