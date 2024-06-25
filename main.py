import os
import random
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from model import snn
from FCSNN_1 import snn_1
# from FCSNN_2 import snn_2
# from FCSNN_3 import snn_3
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def eval_cnn(predicted, y_test, n_class):
    if n_class == 2:
        prediction = [1 if p >= 0.5 else 0 for p in predicted]
        label = y_test
    else:
        prediction = [np.argmax(p) for p in predicted]
        label = [np.argmax(l) for l in y_test]

    acc = accuracy_score(label, prediction)
    fm = f1_score(label, prediction, average='weighted')
    prec = precision_score(label, prediction, average='weighted')
    rec = recall_score(label, prediction, average='weighted')
    confus = confusion_matrix(label, prediction)

    return acc, fm, prec, rec, confus


def set_seeds(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=1):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    keras.backend.set_floatx('float32')


def train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test, n_class, loss_fn, metrics, opt, lr, batch, sqr, fold):
    print(lr, batch)

    if experiment == '1':
        classifier = snn_1(n_class)
    elif experiment == '2':
        classifier = snn_1(n_class)
    # elif experiment == '3':
    #     classifier = snn_3(n_class)
    else:
        classifier = snn(n_class)

    model = classifier.get_model(input_shape=(64, 64, 1), residual = True, sqr= sqr)


    print(model.summary())
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001)
    early_s = EarlyStopping(monitor='val_loss', patience=100)

    if(str(opt) == "<class 'keras.optimizer_v2.gradient_descent.SGD'>"):
        optimizer = opt(learning_rate=lr, momentum=0.9)
        # print("hoi")
    else:
        optimizer = opt(learning_rate=lr)


    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    model.fit([X_train_fold[:, 0], X_train_fold[:, 1]], y_train_fold[:], batch_size=batch, epochs=1000,
              validation_data=([X_val_fold[:, 0], X_val_fold[:, 1]], y_val_fold[:]), callbacks = [reduce_lr, early_s],
              verbose=1)


    # #------------------Validation
    # predicted = model([X_val[:, 0], X_val[:, 1]], training = False)
    #
    # acc, fm, prec, rec, confus = eval_cnn(predicted, y_val, n_class)
    #
    # print("Accuracy: ", acc)
    # print("F-Measure: ",fm)
    # print("Precision: ",prec)
    # print("Recall: ",rec)
    # print(confus)
    #
    # val_results = pd.DataFrame({
    #     "sqrt": [sqrt],
    #     "batch": [batch],
    #     "lr": [lr],
    #     "Optimization": [opt],
    #     "Fold": [fold],
    #     "val F1": [fm],
    #     "val Accuracy": [acc],
    #     "val Precision": [prec],
    #     "val Recall": [rec],
    #     # "Test Specificity": [test_specificity],
    #     # "Test AUC": [test_auc]
    # })
    #
    # # Determine Excel file path
    # excel_file_path = f"val_results.xlsx"
    #
    # # Check if Excel file exists
    # if os.path.isfile(excel_file_path):
    #     # If file exists, open it and append new data
    #     existing_data = pd.read_excel(excel_file_path)
    #     combined_data = pd.concat([existing_data, val_results], ignore_index=True)
    #     combined_data.to_excel(excel_file_path, index=False)
    #     print("Test results appended to existing Excel file:", excel_file_path)
    # else:
    #     # If file doesn't exist, create a new Excel file and save the data
    #     val_results.to_excel(excel_file_path, index=False)
    #     print("Test results saved to new Excel file:", excel_file_path)



    #------------------Testing
    print("--------------------------------------------------------------------------------")
    predicted = model([X_test[:, 0], X_test[:, 1]], training = False)

    acc, fm, prec, rec, confus = eval_cnn(predicted, y_test, n_class)

    print("Accuracy: ", acc)
    print("F-Measure: ",fm)
    print("Precision: ",prec)
    print("Recall: ",rec)
    print(confus)

    test_results = pd.DataFrame({
        "sqr": [sqr],
        "batch": [batch],
        "lr": [lr],
        "Optimization": [opt],
        "Fold": [fold],
        "Test F1": [fm],
        "Test Accuracy": [acc],
        "Test Precision": [prec],
        "Test Recall": [rec],
        # "Test Specificity": [test_specificity],
        # "Test AUC": [test_auc]
    })

    # Determine Excel file path
    excel_file_path = f"test_results.xlsx"

    # Check if Excel file exists
    if os.path.isfile(excel_file_path):
        # If file exists, open it and append new data
        existing_data = pd.read_excel(excel_file_path)
        combined_data = pd.concat([existing_data, test_results], ignore_index=True)
        combined_data.to_excel(excel_file_path, index=False)
        print("Test results appended to existing Excel file:", excel_file_path)
    else:
        # If file doesn't exist, create a new Excel file and save the data
        test_results.to_excel(excel_file_path, index=False)
        print("Test results saved to new Excel file:", excel_file_path)

    return fm, model


def run(experiment, sqrt, data):
    # os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5, 6, 7"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    SEED=1

    set_global_determinism(seed=SEED)

    # +
    # h5f = h5py.File('Data/3blur_all_data64x64.h5', 'r')
    # X_train = h5f['X_train'][:]
    # y_train = h5f['Y_train'][:]
    # X_val = h5f['X_val'][:]
    # y_val = h5f['Y_val'][:]
    # X_test = h5f['X_test'][:]
    # y_test = h5f['Y_test'][:]
    # h5f.close()
    # n_class = len(set(y_train))
    with open('Data/3blur_all_data64x64.npy', 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)
        X_val = np.load(f)
        y_val = np.load(f)
        X_test = np.load(f)
        y_test = np.load(f)
    f.close()

    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)

    if data == 1:
        #only 0 and 4
        # train_mask = np.isin(y_train, [0, 4])
        # val_mask = np.isin(y_val, [0, 4])
        train_mask = np.isin(y_combined, [0, 4])
        test_mask = np.isin(y_test, [0, 4])

        # X_train = X_train[train_mask]
        # y_train = y_train[train_mask]
        # X_val = X_val[val_mask]
        # y_val = y_val[val_mask]
        X_train = X_combined[train_mask]
        y_train = y_combined[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        y_train[y_train == 4] = 1
        # y_val[y_val == 4] = 1
        y_test[y_test == 4] = 1
        pass
    elif data  == 2:
        y_train = y_combined

        #0-3 is 0 and 4 is 1
        y_train[y_train == 1] = 0
        y_train[y_train == 2] = 0
        y_train[y_train == 3] = 0
        y_train[y_train == 4] = 1

        y_test[y_test == 1] = 0
        y_test[y_test == 2] = 0
        y_test[y_test == 3] = 0
        y_test[y_test == 4] = 1

        # y_val[y_val == 1] = 0
        # y_val[y_val == 2] = 0
        # y_val[y_val == 3] = 0
        # y_val[y_val == 4] = 1

    elif data == 3:
        #all individual classes

        # TODO: No class weight
        pass

    n_class = len(set(y_train))

    # -

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if(n_class > 2):
        #     y_train = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_train)
        #     y_val = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_val)
        #     y_test = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_test)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']
    else:
        loss_fn = keras.losses.BinaryCrossentropy()
        metrics = ['binary_accuracy']


    fm_ = -999
    best_fold = 100
    count = 0
    if int(experiment) <= 3 and int(experiment) > 0:
        learn_rate = [0.005, 0.001, 0.0001]
        learn_batch = [512, 64, 16]
        opt_learn = [Adamax, Nadam, Adam]

        # fm, model = train_model(X_train, y_train, X_val, y_val, X_test, y_test, n_class, loss_fn, metrics,
        #                         opt_learn[int(experiment)-1], learn_rate[int(experiment)-1], learn_batch[int(experiment)-1],
        #                         sqrt)
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

            fm, model = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test,
                                    n_class, loss_fn, metrics, opt_learn[int(experiment)-1], learn_rate[int(experiment)-1],
                                    learn_batch[int(experiment)-1], sqrt, count)

            print("Fold F-Measure: ", fm)
            if fm_ < fm:
                best_lr = learn_rate[int(experiment)-1]
                best_batch = learn_batch[int(experiment)-1]
                fm_ = fm
                best_model = model
                opt_ = opt_learn[int(experiment)-1]
                best_fold = count
            count += 1

        # best_lr = learn_rate[int(experiment)-1]
        # best_batch = learn_batch[int(experiment)-1]
        # fm_ = fm
        # best_model = model
        # opt_ = opt_learn[int(experiment)-1]

    else:

        # # #Grid
        # opt_learn = [Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
        # # fm_ = -999
        # learn_rate = [0.0001,0.0005,0.001,0.005]
        #
        # learn_batch = [512, 256, 128, 64, 32, 16, 8]

        opt_learn = [Nadam, Adam, Adamax]
        # fm_ = -999
        learn_rate = [0.0001,0.0005,0.001,0.005]

        learn_batch = [512, 256, 128, 64, 32, 16, 8]

        # #Grid
        # learn_rate = [0.001]
        # learn_batch = [256]
        # opt_learn =  [Adam]
        #learn_rate = [0.001]
        #learn_batch = [100]
        #opt_learn =  [SGD]


        for opt in opt_learn:
            for lr in learn_rate:
                for batch in learn_batch:
                    # fm, model = train_model(X_train, y_train, X_val, y_val, X_test, y_test, n_class, loss_fn, metrics, opt, lr, batch, sqrt)
                    # print("LR: ", lr, " Batch: ", batch," F-Measure test: ", fm)
                    # if(fm_ < fm):
                    #     best_lr = lr
                    #     best_batch = batch
                    #     fm_ = fm
                    #     best_model = model
                    #     opt_ = opt
                    count = 1
                    for train_index, test_index in skf.split(X_train, y_train):
                        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
                        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

                        fm, model = train_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test, y_test,
                                                n_class, loss_fn, metrics, opt, lr, batch, sqrt, count)
                        print("Opt: ", opt, " LR: ", lr, " Batch: ", batch, " Fold F-Measure: ", fm)
                        if fm_ < fm:
                            best_lr = lr
                            best_batch = batch
                            fm_ = fm
                            best_model = model
                            opt_ = opt
                            best_fold = count
                        count += 1


    print("Best learning_rate: ",best_lr)
    print("Best batch: ",best_batch)
    print("Best accuracy: ",fm_)
    print("Best fold: ",best_fold)
    print(best_model.summary())
    fm_ = str(fm_)
    fm_ = fm_[0:6]
    best_model.save('saved_model/'+str(fm_)+'_'+str(best_batch)+'_'+str(opt_)+'_lr_'+str(best_lr)+'_exp_'+str(experiment)+'_sqrt_'+str(sqrt)+'_blur16x16_.h5')

if __name__ == '__main__':

    excel_file_path = f"test_results.xlsx"

    # Check if Excel file exists
    if os.path.isfile(excel_file_path):
        os.remove(excel_file_path)

    experiment = sys.argv[1]

    if sys.argv[2] == 'None':
        sqrt = False
    else:
        sqrt = True

    if len(sys.argv) == 4:
        data = int(experiment)
    else:
        data = int(sys.argv[4])
    run(experiment, sqrt, data)