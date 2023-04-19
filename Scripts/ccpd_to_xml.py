import os
import re
import cv2
root_path = '/home/cuckoo/Public/WDisk/Yolov5-ER/data/YOLO_CCPD/train/ccpd_challenge/'
file_name = os.listdir(root_path)
class_name = 'plate'
xml_dir =r'D:\CHEPAI\Annotations/'  # 保存xml的目录
a= 0
for i in file_name:
    if os.path.exists((xml_dir + i)):
        print('已经存在文件%s'%(xml_dir + i))
    else:os.mkdir((xml_dir + i))
    file_name1 = os.listdir(os.path.join(root_path,i))
    for ii in file_name1:
        if os.path.exists(xml_dir + i + '/'+ ii.split('.')[0]+'.xml'):
            continue
        #print(root_path+i+'/' + ii)
        img = cv2.imread((root_path+i+'/'+ii))
        print((root_path+i+'/'+ii))
        if img is None:
            print('文件%s读取失败'%ii)
            continue

        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        point = ii.split('.')[0].split('-')[3]
        #Xmin = point.split('_')[2].split('&')[0]
        num  = re.findall('\d+\d*',point)  #  正则表达式 从字符串中提取数值
        Xmin =min(num[0::2])  #　list[start:stop:step]
        Ymin = min(num[1::2])
        Xmax = max(num[0::2])
        Ymax = max(num[1::2])
        #print(ii.split('&'))
        fname = ii.split('&')[0] + '&amp;'+ii.split('&')[1]+ '&amp;'+ii.split('&')[2]+ '&amp;'+ii.split('&')[3]+ '&amp;'+ii.split('&')[4]+ '&amp;'+ii.split('&')[5]+ '&amp;'+ii.split('&')[1]+ '&amp;'+ii.split('&')[6]
        xml_str = "<annotation>\n\t\
                <folder>"+ i+ "</folder>\n\t\
                <filename>" + fname + "</filename>\n\t\
                " + "<path>" + root_path + i + '/'+fname+ "</path>\n\t\
                <source>\n\t\t\
                <database>Unknown</database>\n\t\
                </source>\n\t\
                <size>\n\t\t\
                <width>" + str(width) + "</width>\n\t\t\
                <height>" + str(height) + "</height>\n\t\t\
                <depth>" + str(depth) + "</depth>\n\t\
                </size>\n\t\
                <segmented>0</segmented>"
        obj_str = "\n\t\
                    <object>\n\t\t\
                    <name>" + class_name + "</name>\n\t\t\
                    <pose>Unspecified</pose>\n\t\t\
                    <truncated>0</truncated>\n\t\t\
                    <difficult>0</difficult>\n\t\t\
                    <bndbox>\n\t\t\t\
                    <xmin>" + str(Xmin) + "</xmin>\n\t\t\t\
                    <ymin>" + str(Ymin) + "</ymin>\n\t\t\t\
                    <xmax>" + str(Xmax) + "</xmax>\n\t\t\t\
                    <ymax>" + str(Ymax) + "</ymax>\n\t\t\
                    </bndbox>\n\t\
                    </object>"
        xml_str += obj_str
        xml_str +="\n</annotation>\n"
        with open(xml_dir + i + '/'+ ii.split('.')[0]+'.xml','w') as f:
            f.write(xml_str)
            a += 1
            print('成功读写文件夹%s第%d'%(i,a))
            f.close()
print('end')
