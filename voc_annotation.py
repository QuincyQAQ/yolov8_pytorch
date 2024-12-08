import os
import random
import xml.etree.ElementTree as ET
import os
import shutil
from PIL import Image
import numpy as np
from utils.utils import get_classes


# 读取文件并处理每一行
def process_line(line):
    parts = line.strip().split(' ')
    image_path = parts[0]  # 图片路径
    coordinates = parts[1:]  # 坐标信息
    
    # 获取图片的宽度和高度
    img = Image.open(image_path)
    width, height = img.size
    
    result = []
    for coordinate in coordinates:
        # 拆分标注框的坐标，并转换为整数
        coords = coordinate.split(',')
        x = int(coords[0])
        y = int(coords[1])
        w = int(coords[2]) - x  # 宽度 = xmax - xmin
        h = int(coords[3]) - y  # 高度 = ymax - ymin
        
        # 计算归一化坐标
        normalized_xmin = x / width
        normalized_ymin = y / height
        normalized_xmax = (x + w) / width
        normalized_ymax = (y + h) / height
        
        result.append(f"0 {normalized_xmin} {normalized_ymin} {normalized_xmax} {normalized_ymax}")
    
    return result

# 主处理函数
def process_file(input_file, output_dir):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        result = process_line(line)
        
        # 获取图片名称并创建输出文件路径
        image_path = line.strip().split(' ')[0]
        image_name = os.path.basename(image_path).replace('.jpg', '.txt')
        output_file = os.path.join(output_dir, image_name)
        
        # 将结果写入文件
        with open(output_file, 'w') as out_f:
            for item in result:
                out_f.write(item + '\n')


#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
#-------------------------------------------------------------------#
classes_path        = 'model_data/voc_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _      = get_classes(classes_path)

#-------------------------------------------------------#
#   统计目标数量
#-------------------------------------------------------#
input_file = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/2007_val.txt"
output_dir = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/validation_results/annotations"
photo_nums  = np.zeros(len(VOCdevkit_sets))
nums        = np.zeros(len(classes))
def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
        nums[classes.index(cls)] = nums[classes.index(cls)] + 1
        
if __name__ == "__main__":
    random.seed(0)
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")

    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
        
        def printTable(List1, List2):
            for i in range(len(List1[0])):
                print("|", end=' ')
                for j in range(len(List1)):
                    print(List1[j][i].rjust(int(List2[j])), end=' ')
                    print("|", end=' ')
                print()

        str_nums = [str(int(x)) for x in nums]
        tableData = [
            classes, str_nums
        ]
        colWidths = [0]*len(tableData)
        len1 = 0
        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        printTable(tableData, colWidths)

        if photo_nums[0] <= 500:
            print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

        if np.sum(nums) == 0:
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
            print("（重要的事情说三遍）。")

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 清空目标文件夹中的内容
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        # 读取输入文件并处理路径
        total_lines = 0
        missing_files = []

        with open(input_file, "r") as file:
            for line in file:
                total_lines += 1
                # 提取图片路径
                image_path = line.split()[0].strip()

                # 检查图片是否存在
                if os.path.exists(image_path):
                    # 获取图片的宽度和高度
                    img = Image.open(image_path)
                    width, height = img.size

                    coordinates = line.strip().split(' ')[1:]  # 获取坐标信息
                    result = []

                    for coordinate in coordinates:
                        # 拆分标注框的坐标，并转换为整数
                        coords = coordinate.split(',')
                        x = int(coords[0])
                        y = int(coords[1])
                        w = int(coords[2]) - x  # 宽度 = xmax - xmin
                        h = int(coords[3]) - y  # 高度 = ymax - ymin

                        # 计算归一化坐标
                        normalized_xmin = x / width
                        normalized_ymin = y / height
                        normalized_xmax = (x + w) / width
                        normalized_ymax = (y + h) / height

                        result.append(f"0 {normalized_xmin} {normalized_ymin} {normalized_xmax} {normalized_ymax}")

                    # 获取图片名称并创建输出文件路径
                    image_name_txt = os.path.basename(image_path).replace('.jpg', '.txt')
                    output_file = os.path.join(output_dir, image_name_txt)

                    # 将结果写入文件
                    with open(output_file, 'w') as out_f:
                        for item in result:
                            out_f.write(item + '\n')
                else:
                    missing_files.append(image_path)

        # 输出结果
        print(f"Total lines in file: {total_lines}")
        print(f"Missing files: {len(missing_files)}")
        if missing_files:
            print("The following files are missing:")
            for missing in missing_files:
                print(missing)


# 输入文件路径和输出目录
input_file = '2007_val.txt'  # 输入文件
output_dir = 'output_validation_results/annotations'  # 输出目录

# 创建目标文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 清空目标文件夹中的内容
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)

# 处理文件
process_file(input_file, output_dir)
