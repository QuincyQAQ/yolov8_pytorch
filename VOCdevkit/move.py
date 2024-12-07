import shutil
import os

# 定义源文件夹和目标文件夹
src_folder = 'VOC2007-dandu'
dst_folder = 'VOC2007'

# 检查目标文件夹是否存在，不存在则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 使用shutil复制整个文件夹内容
shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
