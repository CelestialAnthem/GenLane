#!/bin/bash

# 假设输入文件名为 input.txt
input_file="/mnt/share_disk/songyuhao/seg_cleanlab/test/160.txt"

# 创建文件夹 0 和 1
mkdir -p 0 1

# 计算每个文件的行数
lines_per_file=$((160 / 16))

# 分割文件并移动到对应的文件夹
split -l $lines_per_file $input_file temp_

# 重命名并移动文件
counter=0
for file in temp_*
do
  folder=$((counter / 8))
  filename=$((counter % 8)).txt
  mv "$file" "$folder/$filename"
  ((counter++))
done

# 清理临时文件
# rm temp_*
