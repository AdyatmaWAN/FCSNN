#!/bin/bash

# Define hyperparameter values
#optimizers=(Adam RMSprop Nadam Adamax)
optimizers=(Adam SGD RMSprop Nadam Adamax)
learning_rates=(0.0001 0.0005 0.001 0.005)
batch_sizes=(512 256 128 64 32 16 8)
experiments=(1 2 3)
num_of_layers=(3)
#num_of_layers=(1 3)
epochs=100

# Boolean flags
residual_flags=(true false)
dropout_flags=(true false)
dense_flags=(true false)
substraction_flags=(true false)
shared_flags=(true false)
weighted_flags=(true false)

# Output log file
echo "Starting grid search..." > grid_search.log


# Loop through all hyperparameter combinations in reversed order
for weighted in "${weighted_flags[@]}"; do
    for shared in "${shared_flags[@]}"; do
        for substraction in "${substraction_flags[@]}"; do
            for dense in "${dense_flags[@]}"; do
                for dropout in "${dropout_flags[@]}"; do
                    for residual in "${residual_flags[@]}"; do
                        for experiment in "${experiments[@]}"; do
                            for num_of_layer in "${num_of_layers[@]}"; do
                                for batch_size in "${batch_sizes[@]}"; do
                                    for learning_rate in "${learning_rates[@]}"; do
                                        for optimizer in "${optimizers[@]}"; do
                                            # Run the Python script with the current combination of parameters
                                            python run.py \
                                                --optimizer "$optimizer" \
                                                --learning_rate "$learning_rate" \
                                                --batch_size "$batch_size" \
                                                --experiment "$experiment" \
                                                --num_of_layer "$num_of_layer" \
                                                --epoch "$epochs" \
                                                $( [ "$residual" == "true" ] && echo "--residual" ) \
                                                $( [ "$dropout" == "true" ] && echo "--dropout" ) \
                                                $( [ "$dense" == "true" ] && echo "--dense" ) \
                                                $( [ "$substraction" == "true" ] && echo "--substraction" ) \
                                                $( [ "$shared" == "true" ] && echo "--shared" ) \
                                                $( [ "$weighted" == "true" ] && echo "--weighted" ) \
                                                >> grid_search.log 2>&1
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Grid search completed." >> grid_search.log
