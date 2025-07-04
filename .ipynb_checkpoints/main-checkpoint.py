# +
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5, 6, 7"
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
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from model import snn
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
    if(n_class==2):
      prediction = []
      label = []
      for i in range(len(predicted)):
          if(predicted[i]<0.5):
              temp = 0
          else:
              temp = 1
          prediction.append(temp)
          label.append(y_test[i])
    else:
        prediction = []
        label = []
        for i in range(len(predicted)):
            prediction.append(np.argmax(predicted[i]))
            label.append(np.argmax(y_test[i]))

    return label, prediction

# opt_learn =  [Adam, RMSprop, SGD, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
# fm_ = -999
# learn_rate = [0.0001,0.0005,0.001,0.005]

# learn_batch = [512, 256, 128, 64, 32, 16, 8]




fm_ = -999
#Grid
learn_rate = [0.001]
learn_batch = [256]
opt_learn =  [Adam]
#learn_rate = [0.001]
#learn_batch = [100]
#opt_learn =  [SGD]
for opt in opt_learn:
    for lr in learn_rate:
        for batch in learn_batch:
            print(lr, batch)
            classifier = snn(n_class)
            model = classifier.get_model(input_shape=(64, 64, 1), residual = True)
            

            print(model.summary())
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=0.00001)
            early_s = EarlyStopping(monitor='val_loss', patience=100)
            
            if(str(opt) == "<class 'keras.optimizer_v2.gradient_descent.SGD'>"):
                optimizer = opt(learning_rate=lr, momentum=0.9)
                print("hoi")
            else:
                optimizer = opt(learning_rate=lr)
            

            model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
            model.fit([X_train[:, 0], X_train[:, 1]], y_train[:], batch_size=batch, epochs=100, validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]), callbacks = [reduce_lr, early_s], verbose=1)       
       
            
            
            #------------------Training
            predicted = model([X_train[:, 0], X_train[:, 1]], training = False)
    
            acc, fm, prec, rec, confus = eval_cnn(predicted, y_train, n_class)    

            print("Accuracy: ", acc)
            print("F-Measure: ",fm)
            print("Precision: ",prec)
            print("Recall: ",rec)
            print(confus)
            
            
            
            #------------------Testing
            print("--------------------------------------------------------------------------------")
            predicted = model([X_test[:, 0], X_test[:, 1]], training = False)

            acc, fm, prec, rec, confus = eval_cnn(predicted, y_test, n_class)    

            print("Accuracy: ", acc)
            print("F-Measure: ",fm)
            print("Precision: ",prec)
            print("Recall: ",rec)
            print(confus)
            
            
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
best_model.save('saved_model/'+str(fm_)+'_'+str(best_batch)+'_'+str(opt_)+'_lr_'+str(best_lr)+'_classweighted_blur16x16_.h5')

