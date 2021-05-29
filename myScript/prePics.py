import os
import random
import cv2
from PIL import Image


# 将 plate/文件夹下的车牌图片resized为 124*204 
def resized_plate(PLATE_PATH, PLATE_RESIZED, WIDTH, HEIGHT):
    plate = os.listdir(PLATE_PATH)
    for index, item in enumerate(plate):
        image = Image.open(PLATE_PATH + plate[index])
        resized_img = image.resize((WIDTH, HEIGHT))
        resized_img.save(PLATE_RESIZED + plate[index]) 

# 压缩图片
class Compress_img:

    def __init__(self, img_path, resized_path):
        self.img_path = img_path
        self.resized_path = resized_path

    def compress_img_CV(self, compress_rate=0.5, show=False):
        
        all_img = os.listdir(self.img_path)
        for img_name in all_img:
            img = cv2.imread(self.img_path + img_name)
            width, heigh = img.shape[:2]
            # 双三次插值
            img_resize = cv2.resize(img, (int(heigh*compress_rate), int(width*compress_rate)),
                                    interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.resized_path + img_name.split('/')[-1], img_resize)
            print("%s 已压缩，" % (self.img_path + img_name), "压缩率：", compress_rate)
            if show:
                cv2.imshow(img_name, img_resize)
                cv2.waitKey(0)


def convert(size, box):
    '''
    将标注的xml文件标注转换为darknet形的坐标
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def divide_sets(xml_file_path,txt_save_path):
    trainval_percent = 0.1  
    train_percent = 0.9    

    all_xml = os.listdir(xml_file_path)
    num = len(all_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv) #从所有list中返回tv个数量的项目
    train = random.sample(trainval, tr)
    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
    f_trainval = open(txt_save_path + 'trainval.txt', 'w')
    f_test = open(txt_save_path + 'test.txt', 'w')
    f_train = open(txt_save_path + 'train.txt', 'w')
    f_val = open(txt_save_path + 'val.txt', 'w')
    for i in list:
        name = all_xml[i][:-4] + '\n'
        if i in trainval:
            f_trainval.write(name)
            if i in train:
                f_test.write(name)
            else:
                f_val.write(name)
        else:
            f_train.write(name)
    f_trainval.close()
    f_train.close()
    f_val.close()
    f_test.close()

if __name__ == "__main__":

    # resized_plate('data/images/test/', 'data/images/val/', 600, 800)

    # 使用opencv压缩图片
    compress = Compress_img('data/plate/all_images/', 'data/plate/exp1/img/')
    compress.compress_img_CV(compress_rate=0.25)
    
    # 随机切分数据集，训练集：验证集 = 9：1
    divide_sets('data/plate/all_xml/', 'data/plate/exp1/txt/')