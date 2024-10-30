#!/bin/bash

# Function to remove .ckpt files and create new files with the same names
remove_and_create_func() {
    local rmpath=$1

    # Create new files with the same names as the removed .ckpt files
    for file in $(find "$rmpath" -type f -name "*.ckpt" -printf "%f\n"); do
        touch "$rmpath/$file"
        echo "Created new file: $rmpath/$file"
    done
    
    # Find and remove .ckpt files
    find "$rmpath" -type f -name "*.ckpt" -exec rm -f {} +


    echo "Processed .ckpt files in $rmpath"
}

# Check if a path was provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path>"
    exit 1
fi

# Call the function with the provided path
remove_and_create_func "$1"

