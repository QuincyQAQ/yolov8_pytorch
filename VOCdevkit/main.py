import shutil
import os

# 源文件夹路径
source_folder = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/VOCdevkit/VOC2007-fire/ImageSets'  # 替换为你的源文件夹路径

# 目标文件夹路径
destination_folder = '/root/autodl-tmp/haichong/YOLOv8-Magic-8.3.12/yolov8-pytorch-master/VOCdevkit/VOC2007/ImageSets'  # 替换为目标文件夹路径

# 检查目标文件夹是否存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 使用 shutil.copytree 复制整个 ImageSets 文件夹及其所有内容
shutil.copytree(source_folder, destination_folder)

print(f"已将 '{source_folder}' 中的所有文件复制到 '{destination_folder}'。")
