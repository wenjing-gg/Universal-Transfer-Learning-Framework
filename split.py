import os
import nrrd
import numpy as np
from PIL import Image
import random

def process_and_save_image_slices(image_file, label_file, output_dir, label, image_counter, image_format="png"):
    """
    处理单个样本的 image 和 label 文件，保存 label 切片非全黑对应的 image 切片为图片
    
    Args:
        image_file (str): image 文件路径
        label_file (str): label 文件路径
        output_dir (str): 保存图片的输出目录
        label (int): 该样本的标签 (0: 无转移, 1: 有转移)
        image_counter (list): 图片计数器列表，用于记录全局唯一图片编号
        image_format (str): 保存图片的格式，默认 "png"
    """
    # 读取 image 和 label 的 nrrd 文件
    image_data, _ = nrrd.read(image_file)
    label_data, _ = nrrd.read(label_file)

    # 检查 image 和 label 的切片数是否一致
    if image_data.shape[2] != label_data.shape[2]:
        print(f"跳过样本：{image_file} 和 {label_file} 切片数不一致")
        return
    
    # 创建输出目录
    label_output_dir = os.path.join(output_dir, str(label))  # 0: 无转移, 1: 有转移
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    # 遍历切片并保存非全黑的 label 对应的 image 切片
    depth = image_data.shape[2]
    for i in range(depth):
        label_slice = label_data[:, :, i]
        
        if np.any(label_slice != 0):  # 如果该 label 切片非全黑
            image_slice = image_data[:, :, i]
            
            # 归一化 image 切片到 0-255 范围
            image_slice_normalized = ((image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice)) * 255).astype(np.uint8)
            
            # 生成唯一的文件名：图片编号.png
            file_name = f"image_{image_counter[0]:06d}.{image_format}"  # 使用全局图片编号
            image_counter[0] += 1  # 递增全局图片编号
            
            # 保存 image 切片
            img = Image.fromarray(image_slice_normalized)
            img.save(os.path.join(label_output_dir, file_name))

def process_directory(image_dirs, label_dirs, output_base_dir, train_ratio=0.8, image_format="png"):
    """
    处理多个目录中的 image 和 label 文件，并按比例划分为训练集和测试集
    
    Args:
        image_dirs (list): 包含多个 image 目录路径的列表
        label_dirs (list): 包含多个 label 目录路径的列表
        output_base_dir (str): 输出图片的基本目录
        train_ratio (float): 训练集所占比例
        image_format (str): 保存图片的格式，默认 "png"
    """
    all_samples = []
    image_counter = [0]  # 初始化图片计数器
    
    # 遍历 image 和 label 目录，匹配相同序号的文件
    for image_dir, label_dir, label in zip(image_dirs, label_dirs, [0, 0, 1, 1]):  # 前两个是无转移（0），后两个是有转移（1）
        image_files = [f for f in os.listdir(image_dir) if f.endswith("_image.nrrd")]
        label_files = [f for f in os.listdir(label_dir) if f.endswith("_label.nrrd")]
        
        image_dict = {f.split("_image")[0]: os.path.join(image_dir, f) for f in image_files}
        label_dict = {f.split("_label")[0]: os.path.join(label_dir, f) for f in label_files}
        
        for seq in image_dict.keys():
            if seq in label_dict:
                all_samples.append((image_dict[seq], label_dict[seq], label))
            else:
                print(f"序号 {seq} 的 image 文件没有对应的 label 文件")
    
    # 打乱所有样本
    random.shuffle(all_samples)
    
    # 划分训练集和测试集
    train_count = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:train_count]
    test_samples = all_samples[train_count:]
    
    # 创建训练集和测试集目录
    train_output_dir = os.path.join(output_base_dir, "train")
    test_output_dir = os.path.join(output_base_dir, "test")
    
    for image_file, label_file, label in train_samples:
        process_and_save_image_slices(image_file, label_file, train_output_dir, label, image_counter, image_format)
    
    for image_file, label_file, label in test_samples:
        process_and_save_image_slices(image_file, label_file, test_output_dir, label, image_counter, image_format)

    print(f"处理完成，共处理了 {len(all_samples)} 个样本，训练集 {len(train_samples)} 个，测试集 {len(test_samples)} 个")
    print(f"总共保存了 {image_counter[0]} 张图片")

# 使用示例
image_dirs = [
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data0",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-0",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data1",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1"
]

label_dirs = [
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data0",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-0",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data1",
    "/home/yuwenjing/data/肾母细胞瘤CT数据/Data3-1"
]

output_base_dir = "/home/yuwenjing/data/train_test_split"
process_directory(image_dirs, label_dirs, output_base_dir, train_ratio=0.8)
