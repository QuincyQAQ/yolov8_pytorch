import os
import numpy as np
from PIL import Image


# 计算 IOU
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

# 读取并解析 YOLO 格式的 txt 文件
def read_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])  # 类别是整数类型
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((cls, x_center, y_center, width, height))
    return boxes


def calculate_map_and_pa(detection_dir, annotations_dir, iou_threshold=0.5):
    detection_files = sorted(os.listdir(detection_dir))
    annotation_files = sorted(os.listdir(annotations_dir))

    all_detections = []
    all_annotations = []

    for detection_file, annotation_file in zip(detection_files, annotation_files):
        detection_path = os.path.join(detection_dir, detection_file)
        annotation_path = os.path.join(annotations_dir, annotation_file)

        # 仅处理文件，跳过目录
        if os.path.isdir(detection_path) or os.path.isdir(annotation_path):
            continue

        detections = read_boxes(detection_path)
        annotations = read_boxes(annotation_path)

        all_detections.append(detections)
        all_annotations.append(annotations)

    # 计算每个类别的AP
    true_positives = {}  # 动态字典，按实际标签存储
    false_positives = {}
    total_annotations = {}

    # PA相关数据统计
    tp_total = 0  # True Positive 总数
    fp_total = 0  # False Positive 总数
    fn_total = 0  # False Negative 总数
    tn_total = 0  # True Negative 总数

    # 遍历所有检测结果，计算 True Positives 和 False Positives
    for detections, annotations in zip(all_detections, all_annotations):
        for cls, x_center, y_center, width, height in annotations:
            if cls not in total_annotations:
                total_annotations[cls] = 0
                true_positives[cls] = []
                false_positives[cls] = []
            total_annotations[cls] += 1

        for cls, x_center, y_center, width, height in detections:
            if cls not in true_positives:
                true_positives[cls] = []
                false_positives[cls] = []

            matched = False
            for ann in annotations:
                ann_cls, ann_x, ann_y, ann_w, ann_h = ann
                if cls == ann_cls:
                    iou_score = iou([x_center, y_center, width, height], [ann_x, ann_y, ann_w, ann_h])
                    if iou_score >= iou_threshold:
                        matched = True
                        break

            if matched:
                true_positives[cls].append(1)
                false_positives[cls].append(0)
                tp_total += 1
            else:
                true_positives[cls].append(0)
                false_positives[cls].append(1)
                fp_total += 1

        # 未匹配的标注框为 False Negative
        for ann in annotations:
            ann_cls, ann_x, ann_y, ann_w, ann_h = ann
            matched = False
            for det in detections:
                det_cls, det_x, det_y, det_w, det_h = det
                if ann_cls == det_cls:
                    iou_score = iou([det_x, det_y, det_w, det_h], [ann_x, ann_y, ann_w, ann_h])
                    if iou_score >= iou_threshold:
                        matched = True
                        break
            if not matched:
                fn_total += 1

    # 计算 True Negative：总样本减去已统计的 TP, FP, FN
    total_samples = sum([len(dets) + len(anns) for dets, anns in zip(all_detections, all_annotations)])
    tn_total = total_samples - (tp_total + fp_total + fn_total)

    # PA 准确率计算
    pa_accuracy = (tp_total + tn_total) / (tp_total + fp_total + tn_total + fn_total) if total_samples > 0 else 0

    # 计算每个类别的Precision-Recall曲线
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for cls in total_annotations.keys():
        tp = np.array(true_positives[cls])
        fp = np.array(false_positives[cls])
        fn = total_annotations[cls] - tp.sum()

        precision = tp.sum() / (tp.sum() + fp.sum()) if (tp.sum() + fp.sum()) > 0 else 0
        recall = tp.sum() / (tp.sum() + fn) if (tp.sum() + fn) > 0 else 0
        ap_per_class[cls] = precision * recall
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall

    # 计算mAP
    mAP = np.mean(list(ap_per_class.values()))

    return mAP, precision_per_class, recall_per_class, pa_accuracy


            
            
def generate_heatmaps_from_folder(dir_origin_path, output_folder, yolo_model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(dir_origin_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(dir_origin_path, image_file)
        heatmap_save_path = os.path.join(output_folder, image_file.replace('.', '_heatmap.'))

        # 打开图片
        image = Image.open(image_path)
        # 将PIL图像转换为NumPy数组
        image_np = np.array(image)
        
        print(f"生成热力图: {image_file}")
        # 传入转换后的NumPy数组
        yolo_model.detect_heatmap(image_np, heatmap_save_path)
        print(f"热力图已保存: {heatmap_save_path}")
