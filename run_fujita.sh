#!/bin/bash

# Define hyperparameter values
optimizers=(Adam Momentum RMSprop Nadam Adamax)
learning_rates=(0.0001 0.0005 0.001 0.005)
batch_sizes=(512 256 128 64 32 16 8)
experiments=(1 2 3)
epochs=100

# Output log file
echo "Starting grid search..." > grid_search.log

# Loop through all hyperparameter combinations in reversed order
for experiment in "${experiments[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            for optimizer in "${optimizers[@]}"; do
                # Run the Python script with the current combination of parameters
                python run_fujita.py \
                    --optimizer "$optimizer" \
                    --learning_rate "$learning_rate" \
                    --batch_size "$batch_size" \
                    --experiment "$experiment" \
                    --epoch "$epochs" \

                    >> grid_search.log 2>&1
            done
        done
    done
done


echo "Grid search completed." >> grid_search.log
