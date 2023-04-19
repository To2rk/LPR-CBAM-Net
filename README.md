# 车牌检测与识别



## 检测模块

训练数据集：CCPD

- 训练

```bash
python train.py --data data/ccpd_base/ccpd_base.yaml --weights weights/weights/best.pt  

```

- 测试

```bash
python test.py --data data/ccpd_base/ccpd_base.yaml --weights weights/weights/best.pt
```

## 识别模块
**LPR-Attention-Recognition**

### 数据集

- **存放目录结构及命名**

```bash 
LPR-Attention-Recognition/
    - data
      -- CCPD
        --- ccpd_base
          ---- train
            ***** 川A6K0V3.0.jpg       # 表示车牌为 川A6K0V3 的第1张车牌
            ***** 川A6K0V3.1.jpg       # 表示车牌为 川A6K0V3 的第2张车牌
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

- **训练**

正确放置数据集后，开始训练

```bash
python train.py
# 训练好的权重存放在 weight/ 目录
```

- **测试**



