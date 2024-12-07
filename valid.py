import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from yolo import YOLO
import shutil
from utils.utils_calculate import iou,read_boxes,calculate_map_and_pa

if __name__ == "__main__":
    mode = "dir_predict"
    dir_origin_path = "val_img/"
    dir_save_path = "validation_results/detection-results/"
    model_path = "model_data/best.pth"
    input_file = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/2007_val.txt"
    source_folder = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/VOCdevkit/VOC2007/JPEGImages"
    destination_folder = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/val_img"
    detection_dir = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/validation_results/detection-results'
    annotations_dir = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/validation_results/annotations'

    
    # 创建目标文件夹（如果不存在）
    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 清空目标文件夹中的内容
    for file in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    total_lines = 0
    copied_count = 0
    missing_files = []

    # 读取输入文件并处理路径
    with open(input_file, "r") as file:
        for line in file:
            total_lines += 1
            # 提取图片路径
            image_path = line.split()[0].strip()

            # 检查图片是否存在
            if os.path.exists(image_path):
                # 获取图片文件名
                image_name = os.path.basename(image_path)
                # 复制图片到目标文件夹
                shutil.copy(image_path, os.path.join(destination_folder, image_name))
                copied_count += 1
            else:
                missing_files.append(image_path)

    # 输出结果
    print(f"Total lines in file: {total_lines}")
    print(f"Images copied: {copied_count}")
    print(f"Missing files: {len(missing_files)}")
    if missing_files:
        print("The following files are missing:")
        for missing in missing_files:
            print(missing)
    
    yolo = YOLO(model_path=model_path)

    if mode == "valid":
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)

        # 计算 mAP
        mAP, precision_per_class, recall_per_class, pa_accuracy = calculate_map_and_pa(detection_dir, annotations_dir)
        print(f'mAP: {mAP}')
        # print(f'precision_per_class: {precision_per_class}')
        # print(f'recall_per_class: {recall_per_class}')
        print(f'PA: {recall_per_class[0]}')  # 提取类别 0 的 recall
        print(f'P: {precision_per_class[0]}')  # 提取类别 0 的 precision
        print(f'R: {recall_per_class[0]}')  # 提取类别 0 的 precision
        print(f'识别算法误报率PF: {1 - precision_per_class[0]}')  # 提取类别 0 的 precision
        print(f'识别算法漏检率PM: {1 - recall_per_class[0]}')  # 提取类别 0 的 precision
        
        

    elif mode == "dir_predict": 
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                # 获取图片名（去除扩展名）
                image_id = os.path.splitext(img_name)[0]

                # 对图片进行预测
                r_image = yolo.detect_image(image, mode="save", image_id=image_id, dir_save_path=dir_save_path)  # 传入image_id

        print("All images processed and results saved!")
        
        
        
        
        # 计算 mAP
        mAP, precision_per_class, recall_per_class, pa_accuracy = calculate_map_and_pa(detection_dir, annotations_dir)
        print(f'mAP: {mAP}')
        # print(f'precision_per_class: {precision_per_class}')
        # print(f'recall_per_class: {recall_per_class}')
        print(f'PA: {recall_per_class[0]}')  # 提取类别 0 的 recall
        print(f'P: {precision_per_class[0]}')  # 提取类别 0 的 precision
        print(f'R: {recall_per_class[0]}')  # 提取类别 0 的 precision
        print(f'识别算法误报率PF: {1 - precision_per_class[0]}')  # 提取类别 0 的 precision
        print(f'识别算法漏检率PM: {1 - recall_per_class[0]}')  # 提取类别 0 的 precision
        
        
        

