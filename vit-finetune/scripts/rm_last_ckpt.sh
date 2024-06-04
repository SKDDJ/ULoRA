#!/bin/bash

remove_func(){
    date
    local rmpath=$1
    find $rmpath -type f -name "last.ckpt" -exec rm -f {} +
    echo "Removed last.ckpt files in $rmpath"
    date
}

remove_func "/root/shiym_proj/Sara/vit-finetune/output/"