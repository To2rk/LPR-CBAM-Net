# 车牌检测与识别

## **1. 检测模块**

- **存放目录结构及命名**

```bash 
Yolov5-Attention-Detection/
    - data
      -- YOLO_CCPD
        --- ccpd_base
          ---- images
            ----- train
              ****** 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg
              ****** 01-89_90-302&482_487&539-497&538_295&550_294&490_496&478-0_0_30_25_29_31_24-130-26.jpg
              ****** 0116127873564-90_91-366&550_554&621-563&618_372&615_376&552_567&555-0_0_2_2_27_30_24-90-21.jpg
              ****** 0116127873564-90_88-284&485_480&550-489&542_294&550_279&483_474&475-0_0_19_30_33_30_26-122-33.jpg

            ----- val
              ****** 01-90_85-274&361_472&420-475&416_277&422_271&357_469&351-0_0_25_29_33_26_30-165-31.jpg
              ****** 021875-85_94-268&364_483&464-493&435_271&459_276&373_498&349-0_0_24_24_2_27_25-138-17.jpg
              ****** 019375-90_88-223&491_503&573-503&575_225&570_225&493_503&498-0_0_15_24_17_32_33-104-30.jpg
              ****** 020625-89_89-204&468_487&544-499&548_191&556_197&464_505&456-0_0_15_27_25_33_31-147-44.jpg

          ---- labels
            ----- train
              ****** 01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.txt
              ****** 01-89_90-302&482_487&539-497&538_295&550_294&490_496&478-0_0_30_25_29_31_24-130-26.txt
              ****** 0116127873564-90_91-366&550_554&621-563&618_372&615_376&552_567&555-0_0_2_2_27_30_24-90-21.txt
              ****** 0116127873564-90_88-284&485_480&550-489&542_294&550_279&483_474&475-0_0_19_30_33_30_26-122-33.txt

            ----- val
              ****** 01-90_85-274&361_472&420-475&416_277&422_271&357_469&351-0_0_25_29_33_26_30-165-31.txt
              ****** 021875-85_94-268&364_483&464-493&435_271&459_276&373_498&349-0_0_24_24_2_27_25-138-17.txt
              ****** 019375-90_88-223&491_503&573-503&575_225&570_225&493_503&498-0_0_15_24_17_32_33-104-30.txt
              ****** 020625-89_89-204&468_487&544-499&548_191&556_197&464_505&456-0_0_15_27_25_33_31-147-44.txt
        *** ccpd_base.yaml

# 约定：
#      - 表示目录
#      * 表示文件

```

### **训练**

```bash
cd Yolov5_Attention_Detection/
python train.py --data data/YOLO_CCPD/ccpd_base/ccpd_base.yaml --weights weights/yolo/yolov5s.pt  
```

### **测试**

```bash
cd Yolov5_Attention_Detection/
python test.py --data data/YOLO_CCPD/ccpd_base/ccpd_base.yaml --weights weights/yolo/yolov5s.pt
```

## **2. 识别模块**

**LPR-Attention-Recognition**

### **数据集**

- **存放目录结构及命名**

```bash 
LPR-Attention-Recognition/
    - data
      -- CCPD
        --- ccpd_base
          ---- train
            ***** 皖AVS180.0.jpg       # 表示车牌为 皖AVS180 的第1张车牌
            ***** 皖AVS180.1.jpg       # 表示车牌为 皖AVS180 的第2张车牌
            ***** 皖NW905J.0.jpg
            ***** 甘AA1086.0.jpg
          ---- test
            ***** 皖AM9V86.0.jpg
            ***** 川AX0Z19.0.jpg
            ***** 川AX0Z19.1.jpg
            ***** 川FNH603.0.jpg
      ** load_data.py
    - model
    - result
    - weight
    * train.py
    * test.py

# 约定：
#      - 表示目录
#      * 表示文件

```

### **训练**

正确放置数据集后，开始训练

```bash
cd LPR_Attention_Recognition/
python train.py
# 训练好的权重存放在 weight/ 目录
```

### **测试**

```bash
cd LPR_Attention_Recognition/
python test.py --pretrained_model ./weight/LPRAtt_iteration_2000.pth 
# 使用训练好的权重进行推理预测
```

## **3. 检测车牌并识别**

- 图片

```bash
python detect_recognize.py --weights ./weights/yolo/best.pt --source ./data/YOLO_CCPD/ccpd_base/images/val/
```

- 视频

```bash
python detect_recognize.py --weights ./weights/yolo/best.pt --source ./data/YOLO_CCPD/ccpd_base/images/val/1.mp4
```

## **4. 自定义数据集**

(1) 合并数据（不同分辨率合并）

 - 将img和xml文件复制到data/temp目录
 * 执行 Scripts/merge.sh 两次，先合并img，再合并xml

 **获得了所有的img和xml**

(2) 检查img和xml是否对应

  - 执行 Scripts/check.sh ，注意执行两次

 **获得了所有的img和xml，并且一一对应**

(3) 压缩图像，并放到指定文件夹

  创建目录结构如下
        - data/plate/exp1/
          - img
          - label
          - txt
          - xml

 - 执行 Scripts/prePics.py，按上述目录更改好目录参数
 - 将所有的xml也复制到 data/plate/exp1/xml下

 **获得了压缩后的图片和所有的xml，以及随机分割的数据集**

(4) 获得训练数据

  创建目录结构如下
        - data/exp1/
          - images
            - train
            - val
          - labels
            - train
            - val

  - 执行 Scripts/make_dataset.py，注意修改路径的地方有九个

 **获得了可用于训练的数据集**

(5) 创建 data/exp1/exp1.yaml

(6) 训练

(7) 一些可能用到的命令

```bash
# 复制时以时间命名
cp -r old copy_test/new_`date '+%Y%m%d_%H.%M.%S'`

# 批量重命名，在需要重命名的目录下执行
i=1; for x in *; do mv $x $i.jpg; let i=i+1; done

#随机移动指定数量的文件到指定目录
shuf -n 10 -e * | xargs -i mv {} path-to-new-folder
```
