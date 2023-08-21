#!/bin/bash

source_dir="./dataset/trainset"
dest_dirt="./dataset/testset"
dest_dirv="./dataset/validation"

for subdir in "$source_dir"/*; do
    subdir_name=$(basename "$subdir")
    dest_subdirv="$dest_dirv/$subdir_name"
    dest_subdirt="$dest_dirt/$subdir_name"
    mkdir -p "$dest_subdirv"
    mkdir -p "$dest_subdirt"
    count=0
    countv=0
    nfile=$(find "$subdir" -type f | wc -l)
    nfilev=$((nfile*20/100))
    nfilet=$((nfile*10/100))
    shuf_files=$(find "$subdir" -type f | shuf)
    for file in $shuf_files; do
        if [ "$count" -eq "$nfilet" ]; then
            break
        fi
        if [ -f "$file" ]; then
            mv "$file" "$dest_subdirt/"
            count=$((count + 1))
        fi
    done
    shuf_files=$(find "$subdir" -type f | shuf)
    for file in $shuf_files; do
        if [ "$countv" -eq "$nfilev" ]; then
            break
        fi
        if [ -f "$file" ]; then
            mv "$file" "$dest_subdirv/"
            countv=$((countv + 1))
        fi
    done
done

