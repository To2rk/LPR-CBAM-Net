nvidia-docker run -itd --init --gpus all --volume="/home/sakura/文档/GitHub/Yolov5-ER:/workspace" -p 2443:22 ultralytics/yolov5:v2 /start.sh

# 训练

python3 train.py --weights "" --cfg models/yolov5s.yaml --data data/exp1/exp1.yaml --batch-size 6 --epochs 5000 --workers 2
python train.py --weights /home/cuckoo/Public/WDisk/Yolov5-ER/runs/train/exp39/weights/last.pt --cfg models/yolov5s.yaml --data data/YOLO_CCPD/ccpd_base/ccpd_base.yaml --batch-size 24 --epochs 20 --workers 6
python train.py --weights /home/cuckoo/Public/WDisk/Yolov5-ER/yolov5s.pt --cfg models/yolov5s.yaml --data data/YOLO_CCPD/ccpd_base/ccpd_base.yaml --batch-size 24 --epochs 100 --workers 6


python3 train.py --data data/ER/ER.yaml --batch-size 6 --epochs 20

python3.8 train.py --weights "runs/train/exp4/weights/best.pt" --data data/ER/ER.yaml --batch-size 6

python3 detect-m-copy.py --weights /home/cuckoo/Public/WDisk/Yolov5-ER/runs/train/exp35/weights/best.pt --source /home/cuckoo/Public/WDisk/Yolov5-ER/data/YOLO_CCPD/ccpd_base/images/val/ 


## 4月1日更新

正确放置数据集后运行 python  detect-m-copy.py 即可



# 视频检测
python detect.py --weights "/home/cuckoo/Public/WDisk/Yolov5-ER/runs/train/exp35/weights/best.pt" --source "/home/cuckoo/Public/WDisk/Yolov5-ER/data/YOLO_CCPD/ccpd_base/images/val/" --save-crop

python3.8 detect.py --weights "weights/weights/best.pt" --source "data/videos/3.mp4" --save-crop



# 复制时以时间命名

cp -r old copy_test/new_`date '+%Y%m%d_%H.%M.%S'`


## 批量重命名，在需要重命名的目录下执行

i=1; for x in *; do mv $x $i.jpg; let i=i+1; done





## 数据集制作过程

1. 合并数据（不同分辨率合并）

 - 将img和xml文件复制到data/temp目录
 * 执行 myScript/merge.sh 两次，先合并img，再合并xml

 **获得了所有的img和xml**

2. 检查img和xml是否对应

  - 执行 myScript/check.sh ，注意执行两次

 **获得了所有的img和xml，并且一一对应**

3. 压缩图像，并放到指定文件夹

  创建目录结构如下
        - data/plate/exp1/
          - img
          - label
          - txt
          - xml

 - 执行 myScript/prePics.py，按上述目录更改好目录参数
 - 将所有的xml也复制到 data/plate/exp1/xml下

 **获得了压缩后的图片和所有的xml，以及随机分割的数据集**

4. 获得训练数据

  创建目录结构如下
        - data/exp1/
          - images
            - train
            - val
          - labels
            - train
            - val

  - 执行 myScript/make_dataset.py，注意修改路径的地方有九个

 **获得了可用于训练的数据集**

5. 创建 data/exp1/exp1.yaml

6. 训练

5x: 优化结果最好
5s: 实时

python3.8 train.py --weights "" --cfg models/yolov5s.yaml --data data/exp1/exp1.yaml --batch-size 6 --epochs 5000 --workers 4


5.29

将裁剪出的字符打上标签

- [ ] data/plate/all_images
- [ ] 使用CNN模型进行训练

不需要区分分辨率，直接使用全部图片进行分析




wandb:         train/obj_loss █▃▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:           val/box_loss █▅▄▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:           val/cls_loss ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:           val/obj_loss █▄▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                  x/lr0 ▂▅███▇▇▇▆▆▅▅▄▃▃▂▂▂▁▁
wandb:                  x/lr1 ▂▅███▇▇▇▆▆▅▅▄▃▃▂▂▂▁▁
wandb:                  x/lr2 █▅▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:        metrics/mAP_0.5 0.9929
wandb:   metrics/mAP_0.5:0.95 0.89977
wandb:      metrics/precision 0.99379
wandb:         metrics/recall 0.99558
wandb:         train/box_loss 0.01163
wandb:         train/cls_loss 0.0
wandb:         train/obj_loss 0.00362
wandb:           val/box_loss 0.0091
wandb:           val/cls_loss 0.0
wandb:           val/obj_loss 0.00271
wandb:                  x/lr0 0.0022
wandb:                  x/lr1 0.0022
wandb:                  x/lr2 0.0022
wandb:
wandb: Synced 6 W&B file(s), 337 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Synced exp26: https://wandb.ai/torrk/YOLOv5/runs/3guw90yj
wandb: Find logs at: ./wandb/run-20211114_173832-3guw90yj/logs/debug.log
wandb:




## 11月16日

随机移动指定数量的文件到指定目录

shuf -n 10 -e * | xargs -i mv {} path-to-new-folder




