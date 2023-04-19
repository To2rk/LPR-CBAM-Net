#!/bin/bash

from_path="data/temp/xml/"
target_path="data/plate/all_xml/"
file=`ls ${from_path}`
for filename in $file
do
if [ -e "${target_path}${filename}" ];then
cp ${from_path}${filename} ${target_path}new_${filename}
else
cp ${from_path}${filename} ${target_path}${filename}
fi
done