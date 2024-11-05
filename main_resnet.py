import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = warnings, 2 = errors
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import h5py

SEED=1
def set_seeds(seed=SEED):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # random.seed(42)
    tf.random.set_seed(1234)
    np.random.seed(0)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.backend.set_floatx('float32')

set_global_determinism(seed=SEED)
from tf_keras.layers import CategoryEncoding
from tf_keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Lion

from model_resnet import snn

import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from tf_keras.callbacks import ReduceLROnPlateau, EarlyStopping

# +
h5f = h5py.File('Data/3blur_all_data16x16.h5', 'r')
X_train = h5f['X_train'][:]
y_train = h5f['Y_train'][:]
X_val = h5f['X_val'][:]
y_val = h5f['Y_val'][:]
X_test = h5f['X_test'][:]
y_test = h5f['Y_test'][:]
h5f.close()

# Function to filter and modify the labels
def process_data(X, y):
    # Filter out labels 1, 2, 3
    mask = ~np.isin(y, [1, 2, 3])
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Change label 4 to 1
    y_filtered[y_filtered == 4] = 1

    return X_filtered, y_filtered

# Apply the function to train, validation, and test sets
X_train, y_train = process_data(X_train, y_train)
X_val, y_val = process_data(X_val, y_val)
X_test, y_test = process_data(X_test, y_test)

# y_train[y_train == 1] = 0
# y_train[y_train == 2] = 0
# y_train[y_train == 3] = 0
# y_train[y_train == 4] = 1

# y_test[y_test == 1] = 0
# y_test[y_test == 2] = 0
# y_test[y_test == 3] = 0
# y_test[y_test == 4] = 1

# y_val[y_val == 1] = 0
# y_val[y_val == 2] = 0
# y_val[y_val == 3] = 0
# y_val[y_val == 4] = 1

n_class = len(set(y_train))
print("y_train ", n_class)
n_class = len(set(y_val))
print("y_val ", n_class)
n_class = len(set(y_test))
print("y_test ", n_class)

# -

if(n_class > 2):
    #     y_train = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_train)
    #     y_val = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_val)
    #     y_test = CategoryEncoding(num_tokens=5, output_mode="one_hot")(y_test)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']
else:
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = ['binary_accuracy']



def eval_cnn(predicted, y_test, n_class):
    if n_class == 2:
        prediction = [1 if p >= 0.5 else 0 for p in predicted]
        label = y_test
    else:
        prediction = [np.argmax(p) for p in predicted]
        label = [np.argmax(l) for l in y_test]

    acc = accuracy_score(label, prediction)
    fm = f1_score(label, prediction)
    prec = precision_score(label, prediction)
    rec = recall_score(label, prediction)
    # fm = f1_score(label, prediction, average='weighted')
    # prec = precision_score(label, prediction, average='weighted')
    # rec = recall_score(label, prediction, average='weighted')
    confus = confusion_matrix(label, prediction)

    return acc, fm, prec, rec, confus, prediction

# opt_learn =  [Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
# opt_learn = [Lion, RMSprop, SGD, Adam, Adagrad, Adadelta]
# fm_ = -999
# learn_rate = [0.0001,0.0005,0.001]
# learn_batch = [512, 256, 128]


fm_ = -999
#Grid
# learn_rate = [0.001]
# learn_batch = [64]
# opt_learn =  [Lion, RMSprop, Adam]
learn_rate = [0.005]
learn_batch = [512]
opt_learn =  [Lion]
for opt in opt_learn:
    for lr in learn_rate:
        for batch in learn_batch:
            print(lr, batch)
            classifier = snn(n_class)
            model = classifier.get_model(input_shape=(16, 16, 1), residual = True)


            print(model.summary())
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001)
            early_s = EarlyStopping(monitor='val_loss', patience=100)

            if(str(opt) == "<class 'keras.optimizer_v2.gradient_descent.SGD'>"):
                optimizer = opt(learning_rate=lr, momentum=0.9)
                print("hoi")
            else:
                optimizer = opt(learning_rate=lr)


            model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics, jit_compile=False)
            model.fit([X_train[:, 0], X_train[:, 1]], y_train[:], batch_size=batch, epochs=100, validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]), callbacks = [reduce_lr, early_s], verbose=1)



            #------------------Training
            # predicted = model([X_train[:, 0], X_train[:, 1]], training = False)

            #

            print("-------------------------------------------Training------------------------------------------")
            # predicted = model.predict([X_train[:, 0], X_train[:, 1]], batch_size=batch)
            predicted = model([X_train[:, 0], X_train[:, 1]], training = False)
            acc, fm, prec, rec, confus, prediction = eval_cnn(predicted, y_train, n_class)
            # print("Predicted: ", prediction)
            # print("Y_train: ", y_train)
            print("n_class: ", n_class)
            print()
            print("Accuracy: ", acc)
            print("F-Measure: ",fm)
            print("Precision: ",prec)
            print("Recall: ",rec)
            print(confus)



            #------------------Testing
            print("-------------------------------------------Testing-------------------------------------------")
            # predicted = model.predict([X_test[:, 0], X_test[:, 1]], batch_size=batch)
            predicted = model([X_test[:, 0], X_test[:, 1]], training = False)

            acc, fm, prec, rec, confus, prediction = eval_cnn(predicted, y_test, n_class)
            # print("Predicted: ", prediction)
            # print("Y_test: ", y_test)
            print("n_class: ", n_class)
            print()
            print("Accuracy: ", acc)
            print("F-Measure: ",fm)
            print("Precision: ",prec)
            print("Recall: ",rec)
            print(confus)
            # Append results to list
            result_df = pd.DataFrame([{
                "Optimizer": opt.__name__,
                "Learning Rate": lr,
                "Batch Size": batch,
                "Accuracy": acc,
                "F-Measure": fm,
                "Precision": prec,
                "Recall": rec,
                "Confusion Matrix": confus.tolist()
            }])
            # Check if the results file exists
            if os.path.exists("training_results.xlsx"):
                # Read the existing Excel file
                existing_df = pd.read_excel("training_results.xlsx")
                # Append new results
                updated_df = pd.concat([existing_df, result_df], ignore_index=True)
            else:
                # If the file does not exist, use the new results directly
                updated_df = result_df
            # Save the updated DataFrame back to the Excel file
            updated_df.to_excel("training_results.xlsx", index=False)

            print("LR: ", lr, " Batch: ",batch," F-Measure test: ", fm)
            if(fm_ < fm):
                best_lr = lr
                best_batch = batch
                fm_ = fm
                best_model = model
                opt_ = opt

print("Best learning_rate: ",best_lr)
print("Best batch: ",best_batch)
print("Best accuracy: ",fm_)
print(best_model.summary())
fm_ = str(fm_)
fm_ = fm_[0:6]
best_model.save('saved_model/'+str(fm_)+'_'+str(best_batch)+'_'+str(opt_)+'_lr_'+str(best_lr)+'_3blur_64x64_.h5')


