import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = warnings, 2 = errors
import random
import numpy as np
import tensorflow as tf
SEED=1
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.keras.backend.set_floatx('float32')

set_global_determinism(seed=SEED)
from tensorflow.keras.layers import CategoryEncoding
from tf_keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Lion
from snn_vit import snn_vit
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from tf_keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
    fm = f1_score(label, prediction, average='weighted')
    prec = precision_score(label, prediction, average='weighted')
    rec = recall_score(label, prediction, average='weighted')
    confus = confusion_matrix(label, prediction)

    return acc, fm, prec, rec, confus, prediction

# opt_learn =  [Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
# opt_learn = [Lion, RMSprop, SGD, Adam, Adagrad, Adadelta]
# fm_ = -999
# learn_rate = [0.0001,0.0005,0.001]
# learn_batch = [512, 256, 128]

fm_ = -999
#Grid
learn_rate = [0.001]
learn_batch = [64]
opt_learn =  [Lion, RMSprop, Adam]
#learn_rate = [0.001]
#learn_batch = [100]
#opt_learn =  [SGD]

for opt in opt_learn:
    for lr in learn_rate:
        for batch in learn_batch:
            print(lr, batch)
            classifier = snn_vit(n_class)
            model = classifier.get_model(input_shape=(64, 64, 1), residual = True)


            print(model.summary())
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001)
            early_s = EarlyStopping(monitor='val_loss', patience=100)

            if(str(opt) == "<class 'keras.optimizer_v2.gradient_descent.SGD'>"):
                optimizer = opt(learning_rate=lr, momentum=0.9)
                print("hoi")
            else:
                optimizer = opt(learning_rate=lr)


            model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics, jit_compile=False)
            model.fit([X_train[:, 0], X_train[:, 1]], y_train[:], batch_size=batch, epochs=1, validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]), callbacks = [reduce_lr, early_s], verbose=1)



            #------------------Training
            # predicted = model([X_train[:, 0], X_train[:, 1]], training = False)
            #
            print("-------------------------------------------Training------------------------------------------")
            predicted = model.predict([X_train[:, 0], X_train[:, 1]], batch_size=batch)
            # predicted = model([X_train[:, 0], X_train[:, 1]], training = False)

            acc, fm, prec, rec, confus, prediction = eval_cnn(predicted, y_train, n_class)

            print("Predicted: ", prediction)
            print("Y_train: ", y_train)
            print("n_class: ", n_class)
            print()

            print("Accuracy: ", acc)
            print("F-Measure: ",fm)
            print("Precision: ",prec)
            print("Recall: ",rec)
            print(confus)

            #------------------Testing
            print("-------------------------------------------Testing-------------------------------------------")
            predicted = model.predict([X_test[:, 0], X_test[:, 1]], batch_size=batch)
            # predicted = model([X_test[:, 0], X_test[:, 1]], training = False)

            acc, fm, prec, rec, confus, prediction = eval_cnn(predicted, y_test, n_class)

            print("Predicted: ", prediction)
            print("Y_test: ", y_test)
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

# Save all results to an Excel file
# results_df = pd.DataFrame(results)
# results_df.to_excel("training_results.xlsx", index=False)

print("Best learning_rate: ",best_lr)
print("Best batch: ",best_batch)
print("Best accuracy: ",fm_)
print(best_model.summary())
fm_ = str(fm_)
fm_ = fm_[0:6]
best_model.save('saved_model/'+str(fm_)+'_'+str(best_batch)+'_'+str(opt_)+'_lr_'+str(best_lr)+'_classweighted_blur16x16_.h5')