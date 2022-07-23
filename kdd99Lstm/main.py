import pandas as pd
from sklearn import preprocessing
from keras.utils.np_utils import *
from keras import Sequential, callbacks
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, RepeatVector, Dense, LSTM
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.callbacks import CSVLogger
import tensorflow as tf
# import data
def preProcess(filename):
    traindata = pd.read_csv('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/dataset/NSL-KDD/KDDTrain+.txt', header=None)
    testdata = pd.read_csv('/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/dataset/NSL-KDD/KDDTest+.csv', header=None)
    # train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'] )
    # train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'])
    # train_set.remove(['count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_serror_rate'])
    # label_set.pop(0)
    # label_set.pop(0)
    # label_set.pop(0)
    # train_set = [[float(y) for y in x] for x in train_set]
    # label_set = [float(m) for m in label_set]
    X = traindata.iloc[:, 1:42]
    Y = traindata.iloc[:, 0]
    C = testdata.iloc[:, 0]
    T = testdata.iloc[:, 1:42]

    scaler = preprocessing.Normalizer().fit(X)
    trainX = scaler.transform(X)

    scaler = preprocessing.Normalizer().fit(T)
    testT = scaler.transform(T)

    y_train1 = np.array(Y)
    y_test1 = np.array(C)

    y_train = to_categorical(y_train1)
    y_test = to_categorical(y_test1)

    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    X_test = np.reshape(testT, (testT.shape[0], testT.shape[1], 1))

    return X_train, y_train, y_test, X_test

# train the model
# define parameters
def train(X_train, y_train):
    verbose, epochs, batch_size = 1, 10, 1000
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], 1  # 122,1,1

    # reshape output into [samples, timesteps, features]
    y_train = y_train.reshape((y_train.shape[0], 1, 1))

    # define model
    lstm_cnn = Sequential()
    lstm_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    lstm_cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    lstm_cnn.add(MaxPooling1D(pool_size=2))
    lstm_cnn.add(Flatten())
    lstm_cnn.add(RepeatVector(n_outputs))
    Model.add(LSTM(200, activation='relu', return_sequences=True))
    lstm_cnn.add(tf.nn.dropout(0.1))
    lstm_cnn.add(Dense(1, activation='sigmoid'))
    lstm_cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # set checkpoint
    checkpointer = callbacks.ModelCheckpoint(filepath="results/lstm_cnn_results/checkpoint-{epoch:02d}.hdf5", verbose=1,
                                             save_best_only=True, monitor='val_acc', mode='max')

    # set logger
    csv_logger = CSVLogger('results/lstm_cnn_results/cnntrainanalysis1.csv', separator=',', append=False)

    # fit network
    lstm_cnn.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=0.1,
        callbacks=[checkpointer, csv_logger])
    return lstm_cnn


def predict(X_test, y_test, lstm_cnn):

    y_pred = lstm_cnn.predict_classes(X_test)

    y_pred = y_pred[:, 0]
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="binary")
    precision = precision_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    print("Performance over the testing data set n")
    print("Accuracy : {} nRecall : {} nPrecision : {} nF1 : {}n".format(accuracy, recall, precision, f1))


if __name__ == '__main__':
    X_train, y_train, y_test, X_test = preProcess("/Users/university/learningBoardly/scientificResearch/RSpartII/KDD99/dataset/NSL-KDD")
    lstm_cnn = train(X_train, y_train)
    predict(X_test, y_test, lstm_cnn)