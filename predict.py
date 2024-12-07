import time
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import datetime
import os
from functools import partial
import numpy as np
import torch
import shutil
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch
from utils.utils_calculate import iou, read_boxes, calculate_map, generate_heatmaps_from_folder

if __name__ == "__main__":

    mode = "dir_predict"
    crop = False
    count = False
    confidence = 0.1
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    test_interval = 100
    dir_origin_path = "prediction_results/original_image"
    output_folder = "prediction_results/"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    yolo = YOLO(dir_predict=output_folder, confidence=confidence, mode="no_save")

    if mode == "dir_predict":

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                r_image.save(os.path.join(output_folder, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        img_names = os.listdir(dir_origin_path)

        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)

                # 对图片进行预测
                r_image = yolo.detect_image(image)  # 获取处理过的图片

                # Step 4: Move original image to a new folder
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Move the original image to the new folder
                shutil.copy(image_path, os.path.join(output_folder, img_name))

        # 调用批量生成热力图函数
        generate_heatmaps_from_folder(dir_origin_path, output_folder, yolo)

        print("All images processed and results saved!")

