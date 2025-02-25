experiments=(1 2 3)

# Output log file
echo "Starting grid search..."

# Loop through all hyperparameter combinations in reversed order
for experiment in "${experiments[@]}"; do
    # Run the Python script with the current combination of parameters
    python models/Hajeb.py \
        --experiment "$experiment" \

done


echo "Grid search completed."