import os
import argparse
import random
import numpy as np
import tensorflow
import pandas as pd
import h5py

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models.model import snn
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.utils import class_weight

SEED = 1
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)
    tensorflow.keras.backend.set_floatx('float32')

def process_data(X, y):
    # Filter out labels 1, 2, 3
    mask = ~np.isin(y, [1, 2, 3])
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Change label 4 to 1
    y_filtered[y_filtered == 4] = 1

    return X_filtered, y_filtered

set_global_determinism(seed=SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--residual', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dense', action='store_true')
parser.add_argument('--substraction', action='store_true')
parser.add_argument('--shared', action='store_true')
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--optimizer', type=str, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--experiment', type=int, required=True)
parser.add_argument('--num_of_layer', type=int, required=True)
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()

filename_prefix = f"experiment_{args.experiment}_subs_{args.substraction}_shared_{args.shared}_weighted_{args.weighted}_res_{args.residual}_drop_{args.dropout}_dense_{args.dense}"

h5f = h5py.File('Data/3blur_all_data16x16.h5', 'r')
X_train = h5f['X_train'][:]
y_train = h5f['Y_train'][:]
X_val = h5f['X_val'][:]
y_val = h5f['Y_val'][:]
X_test = h5f['X_test'][:]
y_test = h5f['Y_test'][:]
h5f.close()

# Label processing
if args.experiment == 1:
    # Apply the function to train, validation, and test sets
    X_train, y_train = process_data(X_train, y_train)
    X_val, y_val = process_data(X_val, y_val)
    X_test, y_test = process_data(X_test, y_test)
elif args.experiment == 2:
    y_train[y_train < 4] = 0
    y_train[y_train == 4] = 1
    y_test[y_test < 4] = 0
    y_test[y_test == 4] = 1
    y_val[y_val < 4] = 0
    y_val[y_val == 4] = 1
elif args.experiment == 3:
    pass
else:
    raise ValueError("Invalid experiment number")

n_class = len(set(y_train))
loss_fn = tensorflow.keras.losses.BinaryCrossentropy() if n_class == 2 else tensorflow.keras.losses.SparseCategoricalCrossentropy()
metrics = ['binary_accuracy'] if n_class == 2 else ['accuracy']

if args.weighted:
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Convert class weights into a dictionary
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print("Class weights:", class_weight_dict)


# Model setup
classifier = snn(n_class, residual=args.residual, dropout=args.dropout, dense=args.dense, input_shape=(16, 16, 1),
                 num_of_layer=args.num_of_layer, substraction=args.substraction, shared=args.shared)

model = classifier.get_model()

optimizer_class = getattr(tensorflow.keras.optimizers, args.optimizer)
optimizer = optimizer_class(learning_rate=args.learning_rate)

model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics, jit_compile=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, min_lr=0.00001)
early_s = EarlyStopping(monitor='val_loss', patience=25, min_delta=0.0001)
if args.weighted:
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=args.batch_size, epochs=args.epoch,
                        validation_data=([X_val[:, 0], X_val[:, 1]], y_val), callbacks=[reduce_lr, early_s],
                        verbose=1, class_weight=class_weight_dict)
else:
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=args.batch_size, epochs=args.epoch,
                    validation_data=([X_val[:, 0], X_val[:, 1]], y_val), callbacks=[reduce_lr, early_s], verbose=1)

# Save training and validation loss
loss_df = pd.DataFrame({'Epoch': range(1, len(history.history['loss']) + 1),
                        'Train Loss': history.history['loss'],
                        'Validation Loss': history.history['val_loss']})
loss_df.to_csv(f'csv/{filename_prefix}_training_loss.csv', index=False)

# Model evaluation
predicted = model([X_test[:, 0], X_test[:, 1]], training=False)
if n_class == 2:
    prediction = [1 if p >= 0.5 else 0 for p in predicted]
    acc = accuracy_score(y_test, prediction)
    fm = f1_score(y_test, prediction)
    prec = precision_score(y_test, prediction)
    rec = recall_score(y_test, prediction)
    confus = confusion_matrix(y_test, prediction)
else:
    prediction = [np.argmax(p) for p in predicted]
    acc = accuracy_score(y_test, prediction)
    fm = f1_score(y_test, prediction, average='weighted')
    prec = precision_score(y_test, prediction, average='weighted')
    rec = recall_score(y_test, prediction, average='weighted')
    confus = confusion_matrix(y_test, prediction)


print(f'Accuracy: {acc}\nF-Measure: {fm}\nPrecision: {prec}\nRecall: {rec}\nConfusion Matrix:\n{confus}')

# Save model
model.save(f'saved_model/{filename_prefix}_model.h5')

# Save evaluation results
result_df = pd.DataFrame([{
    "Optimizer": args.optimizer,
    "Learning Rate": args.learning_rate,
    "Batch Size": args.batch_size,
    "Accuracy": acc,
    "F-Measure": fm,
    "Precision": prec,
    "Recall": rec,
    "Test Loss": history.history['val_loss'][-1],
    "Confusion Matrix": confus.tolist()
}])
result_df.to_csv(f'csv/{filename_prefix}_training_results.csv', index=False)
