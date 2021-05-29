#!/bin/bash

# 第一遍
# img_path="data/plate/all_images/"
# xml_path="data/plate/all_xml/"
# file=`ls ${img_path}`
# for filename in $file
# do
# name=${filename%.*}
# if ! [ -e "${xml_path}${name}.xml" ];then
# echo "未找到与 ${img_path}${filename} 相对应的xml文件"
# fi
# done

# 第二遍
img_path="data/plate/all_images/"
xml_path="data/plate/all_xml/"
file=`ls ${xml_path}`
for filename in $file
do
name=${filename%.*}
if ! [ -e "${img_path}${name}.jpg" ];then
echo "未找到与 ${xml_path}${filename} 相对应的img文件"
fi
done

