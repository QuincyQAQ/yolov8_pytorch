

import os
import xml.etree.ElementTree as ET

# 设置文件夹路径
folder_path =  "Annotations" 

# 遍历文件夹中的所有 XML 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xml'):
        file_path = os.path.join(folder_path, filename)
        
        # 解析 XML 文件
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 查找所有 <name> 标签并修改其中的 fire 为 Fire
        for obj in root.findall('object'):
            name = obj.find('name')
            if name is not None and name.text == 'fire':
                name.text = 'Fire'
        
        # 保存修改后的 XML 文件
        tree.write(file_path)

print("所有 XML 文件中的 fire 已被成功替换为 Fire")
