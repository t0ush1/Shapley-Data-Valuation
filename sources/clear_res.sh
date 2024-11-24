#!/bin/bash

# Specify the directories containing the files to move
DIRS_TO_MOVE=("./rec_fed_sample_time" "./rec_fed_grad" "./rec_fed_tmr_res")

# Specify the directory to move the files to
TRASH_DIR="_trash"

# Create the trash directory if it doesn't exist
mkdir -p "$TRASH_DIR"

# Get the current time in the format YYYYMMDD_HHMMSS
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# Loop through each directory
for DIR in "${DIRS_TO_MOVE[@]}"
do
    # Check if the directory exists
    if [[ -d $DIR ]]; then
        # Loop through each file in the directory
        for FILE in "$DIR"/*
        do
            # Check if the file exists to avoid errors in case the glob doesn't match any files
            if [[ -e $FILE ]]; then
                # Extract the filename from the path
                FILENAME=$(basename -- "$FILE")
                
                # Construct the new file name by appending the current time
                NEW_FILE_NAME="${FILENAME%.*}_$CURRENT_TIME.${FILENAME##*.}"
                
                # Move and rename the file
                mv "$FILE" "$TRASH_DIR/$NEW_FILE_NAME"
                
                # Print a message to the terminal
                echo "Moved $FILE to $TRASH_DIR/$NEW_FILE_NAME"
            fi
        done
    else
        echo "Directory not found: $DIR"
    fi
done