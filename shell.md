nvidia-docker run -itd --init --gpus all --volume="/home/sakura/文档/GitHub/Yolov5-ER:/workspace" -p 2443:22 ultralytics/yolov5:v2 /start.sh



python3.8 train.py --weights "" --cfg models/yolov5s.yaml --data data/ER/ER.yaml --batch-size 6 --epochs 20

python3.8 train.py --weights "runs/train/exp4/weights/best.pt" --data data/ER/ER.yaml --batch-size 6

python3.8 detect.py --weights weights/weights/best.pt --source data/images/val --save-crop

# 视频检测
python3.8 detect.py --weights "runs/train/exp4/weights/best.pt" --source "data/video/IMG_0366.mp4" --save-crop




# 复制时以时间命名

cp -r old copy_test/new_`date '+%Y%m%d_%H.%M.%S'`


## 批量重命名，在需要重命名的目录下执行

i=1; for x in *; do mv $x $i.jpg; let i=i+1; done





## 数据集制作过程

1. 首先获得image文件和xml文件
  
  - 其中可能存在有些image没有对应的xml

2. 获取那些没有xml的图片名

  - 执行 myScript/match.sh ，

3. 获取xml

  - myScript/match.sh

4. 获取image

  - myScript/rename.sh


5. 压缩

  - myScript/prePics.py

6. 划分数据集

  - myScript/prePics.py

7. 获得训练数据

  - myScript/make_dataset.py

8. 创建exp.yaml


9. 训练

5x: 优化结果最好
5s: 实时

python3.8 train.py --weights "" --cfg models/yolov5x.yaml --data data/exp/exp.yaml --batch-size 6 --epochs 100



