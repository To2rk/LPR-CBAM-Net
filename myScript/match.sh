#!/bin/bash

img_path="data/plate/vertical/img/"
from_path="data/plate/all_xml/"
target_path="data/plate/vertical/xml/"
file=`ls ${img_path}`
for filename in $file
do
name=${filename%.*}
if [ -e "${target_path}${name}.xml" ];then
cp ${from_path}${name}.xml ${target_path}new_${name}.xml
else
cp ${from_path}${name}.xml ${target_path}${name}.xml
fi
done