import os
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
from yolo import YOLO
import shutil
import matplotlib.pyplot as plt
from utils.utils_calculate import iou, read_boxes, calculate_map_and_pa


# 保存每个图像
def save_plot(x, y, label, xlabel, ylabel, title, filename):
    # 确保 `plot` 文件夹存在
    os.makedirs('plot', exist_ok=True)
    
    # 构造完整的文件路径
    filepath = os.path.join('plot', filename)
    
    plt.figure(figsize=(10, 6))
    # 设置 markersize 参数来控制点的大小
    plt.plot(x, y, label=label, marker='o', linestyle='-', color='blue', markersize=3)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath)  # 保存到文件
    plt.close()  # 关闭当前图形，以免多张图叠加




if __name__ == "__main__":
    mode = "dir_predict"
    dir_origin_path = "val_img/"
    dir_save_path = "output_validation_results/detection-results/"
    model_path = "model_data/hunhe-ep300-loss1.847-val_loss2.640.pth"
    input_file = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/2007_val.txt"
    source_folder = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/VOCdevkit/VOC2007/JPEGImages"
    destination_folder = "/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/val_img"
    detection_dir = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/output_validation_results/detection-results'
    annotations_dir = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/output_validation_results/annotations'
    startvalue=0.1
    endvalue=1.0
    step=0.1
    
    # 目标文件夹清理
    os.makedirs(destination_folder, exist_ok=True)
    for file in os.listdir(destination_folder):
        file_path = os.path.join(destination_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # 复制图片到目标文件夹
    total_lines = 0
    copied_count = 0
    missing_files = []
    with open(input_file, "r") as file:
        for line in file:
            total_lines += 1
            image_path = line.split()[0].strip()
            if os.path.exists(image_path):
                image_name = os.path.basename(image_path)
                shutil.copy(image_path, os.path.join(destination_folder, image_name))
                copied_count += 1
            else:
                missing_files.append(image_path)
    print(f"Total lines in file: {total_lines}")
    print(f"Images copied: {copied_count}")
    print(f"Missing files: {len(missing_files)}")
    if missing_files:
        print("The following files are missing:")
        for missing in missing_files:
            print(missing)
    
    # 准备 CSV 文件
    csv_file = "plot/metrics_results.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Confidence", "mAP", "PA", "Precision", "Recall", "False Positive Rate (PF)", "Miss Detection Rate (PM)"])

    # 遍历不同的 confidence 值
    for confidence in np.arange(startvalue, endvalue, step):  # 调整步长和范围以适应需求
        yolo = YOLO(model_path=model_path, confidence=confidence)
        
        if mode == "dir_predict":
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            
            img_names = os.listdir(dir_origin_path)
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path = os.path.join(dir_origin_path, img_name)
                    image = Image.open(image_path)
                    image_id = os.path.splitext(img_name)[0]
                    r_image = yolo.detect_image(image, mode="save", image_id=image_id, dir_save_path=dir_save_path)
        
        # 计算 mAP 和其他指标
        mAP, precision_per_class, recall_per_class, pa_accuracy = calculate_map_and_pa(detection_dir, annotations_dir)
        precision_class_0 = precision_per_class[0]
        recall_class_0 = recall_per_class[0]
        pf = 1 - precision_class_0
        pm = 1 - recall_class_0

        print(f"Confidence: {confidence:.2f}, mAP: {mAP}, PA: {pa_accuracy}, Precision: {precision_class_0}, Recall: {recall_class_0}, PF: {pf}, PM: {pm}")
        
        # 写入到 CSV 文件
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([confidence, mAP, pa_accuracy, precision_class_0, recall_class_0, pf, pm])
    
    print(f"Results saved to {csv_file}.")

    confidences = []
    precision_values = []
    recall_values = []
    pa_values = []

    # 从CSV文件中提取数据
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            confidences.append(float(row[0]))  # Confidence
            precision_values.append(float(row[3]))  # Precision
            recall_values.append(float(row[4]))  # Recall
            pa_values.append(float(row[2]))  # PA
    
    # 保存 Precision vs Confidence 图
    save_plot(
        confidences, 
        precision_values, 
        label='Precision (P)', 
        xlabel='Confidence', 
        ylabel='Precision', 
        title='Precision vs Confidence', 
        filename='precision_vs_confidence.png'
    )

    # 保存 Recall vs Confidence 图
    save_plot(
        confidences, 
        recall_values, 
        label='Recall (R)', 
        xlabel='Confidence', 
        ylabel='Recall', 
        title='Recall vs Confidence', 
        filename='recall_vs_confidence.png'
    )

    # 保存 PA vs Confidence 图
    save_plot(
        confidences, 
        pa_values, 
        label='PA (Per Class Accuracy)', 
        xlabel='Confidence', 
        ylabel='PA', 
        title='PA vs Confidence', 
        filename='pa_vs_confidence.png'
    )

    print("All plots have been saved to the local directory.")


    
    
    
