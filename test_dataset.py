# import os
# import nrrd
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np

# class MyNRRDDataSet(Dataset):
#     """自定义NRRD格式数据集，针对3D卷积适应 (D, H, W) 格式，并进行线性插值和分块"""

#     def __init__(self, root_dir: str, split: str, transform=None, target_shape=(512, 512, 1024), block_size=(256, 256, 256), threshold=0.8):
#         """
#         Args:
#             root_dir (str): 数据集的根目录
#             split (str): 数据集划分，'train' 或 'test'
#             transform (callable, optional): Optional transform to be applied on a sample.
#             target_shape (tuple, optional): 目标的形状 (D, H, W)，默认填充到 (512, 512, 1024).
#             block_size (tuple, optional): 分块大小 (D, H, W)，默认分块为 (256, 256, 256).
#             threshold (float, optional): 丢弃块的掩码中0的比例阈值，默认为 80%。
#         """
#         self.block_list = []  # 存储所有样本的分块、标签和掩码
#         self.transform = transform
#         self.target_shape = target_shape
#         self.block_size = block_size
#         self.threshold = threshold

#         # 加载无转移组 (NoMetastasis) 和 有转移组 (Metastasis) 的数据
#         self._load_images_from_folder(os.path.join(root_dir, split, 'NoMetastasis'), label=0)
#         self._load_images_from_folder(os.path.join(root_dir, split, 'Metastasis'), label=1)

#     def _load_images_from_folder(self, folder: str, label: int):
#         """加载指定文件夹中的所有 NRRD 文件，并分配类别标签"""
#         for filename in os.listdir(folder):
#             if filename.endswith(".nrrd"):  # 假设所有文件都为图像文件
#                 img_path = os.path.join(folder, filename)
#                 print(f"Processing data for: {img_path}")
#                 blocks = self._process_nrrd(img_path)  
#                 self.block_list.extend([(block.clone().detach(), label) for block in blocks]) 

#     def _process_nrrd(self, file_path):
#         """处理 NRRD 文件并返回分块"""
#         data, header = nrrd.read(file_path)

#         # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
#         img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

#         # 确保输入是 3D 数据 (D, H, W)
#         if img.ndim != 3:
#             raise ValueError(f"Image at {file_path} is not a 3D volume.")

#         # 生成掩码，1 表示有效数据，0 表示填充值 -3024 的部分
#         mask = (img != -3024).float()  # 掩码：有效数据为 1，填充值为 0

#         # 线性插值到目标形状 (512, 512, 512)，并且在插值后重新生成掩码
#         img = self.interpolate_to_shape(img, self.target_shape)
#         mask = self.interpolate_to_shape(mask, self.target_shape, fill_value=0)  # 掩码插值并使用 0 填充

#         # 分块，生成多个 (256, 256, 256) 的块
#         blocks = self.split_into_blocks(img, mask, self.block_size)
        
#         # 如果没有生成有效块，抛出一个异常
#         if len(blocks) == 0:
#             raise ValueError(f"No valid blocks generated for file: {file_path}")

#         return blocks  # 只返回 blocks

#     def __len__(self):
#         return len(self.block_list)

#     def __getitem__(self, idx):
#         block, label = self.block_list[idx]  

#         # 如果有 transform，应用 transform 到该块
#         if self.transform:
#             block = self.transform(block)

#         # 对图像块进行标准化
#         block = self.normalize(block)  # 在这里调用标准化函数

#         # 添加通道维度 (C)
#         block = block.unsqueeze(0)
#         return block, label  


#     def normalize(self, img):
#         """
#         将图像归一化到 [0, 1] 范围
#         Args:
#             img: 输入的图像张量 (D, H, W)
#         Returns:
#             归一化到 [0, 1] 的张量
#         """
#         min_val = img.min()
#         max_val = img.max()

#         # 避免除以零的情况
#         if max_val > min_val:
#             img = (img - min_val) / (max_val - min_val)
#         else:
#             img = torch.zeros_like(img)  # 如果 max == min，直接返回全零张量

#         return img


#     def interpolate_to_shape(self, img, target_shape, fill_value=-3024):
#         """
#         对输入的 3D 图像或掩码进行调整，仅对深度维度进行插值或填充，保持高度和宽度为512x512。
#         Args:
#             img: 输入图像张量或掩码 (D, H, W)
#             target_shape: 目标形状 (target_D, target_H, target_W)
#             fill_value: 填充值，默认为 -3024
#         Returns:
#             调整后的张量
#         """
#         d, h, w = img.shape
#         target_d, target_h, target_w = target_shape

#         # 假设 h = target_h 和 w = target_w 都是 512，不需要调整高度和宽度

#         # 处理深度维度：当切片数（d）小于或大于目标尺寸时进行插值或填充
#         if d < target_d:
#             # 如果切片数小于目标，填充到目标深度
#             pad_d = target_d - d
#             img = F.pad(img.unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 0, pad_d), "constant", fill_value)  # 填充前后片
#             img = img.squeeze(0).squeeze(0)
#         elif d > target_d:
#             # 如果切片数大于目标，插值到目标深度
#             img = img.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
#             img = F.interpolate(img, size=(target_d, h, w), mode='trilinear', align_corners=True)  # 仅调整深度维度
#             img = img.squeeze(0).squeeze(0)

#         return img


#     def split_into_blocks(self, img, mask, block_size):
#         """
#         将 3D 图像按块分割，如果掩码中超过80%为0，则丢弃该块。
#         Args:
#             img: 输入图像张量 (D, H, W)
#             mask: 对应的掩码张量 (D, H, W)
#             block_size: 分块的大小 (block_D, block_H, block_W)
#         Returns:
#             列表，其中每个元素是有效的分割块
#         """
#         blocks = []
#         d, h, w = img.shape
#         bd, bh, bw = block_size
#         for i in range(0, d, bd):
#             for j in range(0, h, bh):
#                 for k in range(0, w, bw):
#                     block = img[i:i+bd, j:j+bh, k:k+bw]
#                     mask_block = mask[i:i+bd, j:j+bh, k:k+bw]
                    
#                     # 检查掩码中有效部分的比例
#                     zero_ratio = (mask_block == 0).float().mean().item()
#                     if zero_ratio < self.threshold:  # 如果有效部分少于设定阈值80%，则丢弃
#                         blocks.append(block)

#         # 添加调试信息，检查是否有有效块生成
#         if len(blocks) == 0:
#             print(f"Warning: No valid blocks generated for image with shape {img.shape}")
        
#         return blocks


#     @staticmethod
#     def collate_fn(batch):
#         all_blocks, all_labels = zip(*batch)  # 只解包 block 和 label
#         all_blocks = torch.stack(all_blocks, dim=0)  # 堆叠所有分块图像
#         all_labels = torch.as_tensor(all_labels)     # 转换标签为张量
#         return all_blocks, all_labels  # 不再返回掩码



# # 使用示例
# if __name__ == "__main__":
#     root_dir = r'/home/yuwenjing/data/tmp_data'

#     # 创建数据集实例，假设目标形状为 (512, 512, 512)，分块大小为 (256, 256, 256)
#     dataset = MyNRRDDataSet(root_dir, split='test')

#     # 检查数据集总数（块数）
#     print(f"数据集样本总数（块数）: {len(dataset)}")

#     # 遍历整个数据集，并打印每个块的形状和标签
#     for i in range(len(dataset)):
#         block,label = dataset[i]
#         print(f"第 {i + 1} 个块的形状: {block.shape}")
#         print(f"第 {i + 1} 个块的标签: {label}")
#     firstblock,firstlabel = dataset[0]
#     print(firstblock[:100])
import os
import nrrd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class MyNRRDDataSet(Dataset):
    """自定义NRRD格式数据集，直接加载和处理为 (128, 128, 128) 形状的图像"""

    def __init__(self, root_dir: str, split: str, transform=None, target_shape=(128, 128, 128)):
        """
        Args:
            root_dir (str): 数据集的根目录
            split (str): 数据集划分，'train' 或 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
            target_shape (tuple, optional): 目标形状 (D, H, W)，默认调整到 (128, 128, 128).
        """
        self.data_list = []  # 存储所有样本的图像数据和标签
        self.transform = transform
        self.target_shape = target_shape

        # 加载 0 和 1 文件夹中的数据
        self._load_images_from_folder(os.path.join(root_dir, split, '0'), label=0)  # NoMetastasis
        self._load_images_from_folder(os.path.join(root_dir, split, '1'), label=1)  # Metastasis

    def _load_images_from_folder(self, folder: str, label: int):
        """加载指定文件夹中的所有 NRRD 文件，并分配类别标签"""
        for filename in os.listdir(folder):
            if filename.endswith(".nrrd"):  # 假设所有文件都为 NRRD 图像文件
                img_path = os.path.join(folder, filename)
                print(f"Processing data for: {img_path}")
                img = self._process_nrrd(img_path)  
                self.data_list.append((img, label))  # 直接存储图像及其标签

    def _process_nrrd(self, file_path):
        """处理 NRRD 文件并返回调整后的图像"""
        data, header = nrrd.read(file_path)

        # 转换为 PyTorch 张量，并确保数据格式为 (D, H, W)
        print(data.shape)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # 将 (H, W, D) 转换为 (D, H, W)

        # 确保输入是 3D 数据 (D, H, W)
        if img.ndim != 3:
            raise ValueError(f"Image at {file_path} is not a 3D volume.")

        # 线性插值到目标形状 (128, 128, 128)
        img = self.interpolate_to_shape(img, self.target_shape)
        
        return img  # 直接返回插值后的完整图像

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, label = self.data_list[idx]

        # 如果有 transform，应用 transform 到图像
        if self.transform:
            img = self.transform(img)

        # 对图像进行标准化
        img = self.normalize(img)

        # 添加通道维度 (C)
        img = img.unsqueeze(0)
        return img, label  

    def normalize(self, img):
        """
        将图像归一化到 [0, 1] 范围
        Args:
            img: 输入的图像张量 (D, H, W)
        Returns:
            归一化到 [0, 1] 的张量
        """
        min_val = img.min()
        max_val = img.max()

        # 避免除以零的情况
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = torch.zeros_like(img)  # 如果 max == min，直接返回全零张量

        return img

    def interpolate_to_shape(self, img, target_shape):
        """
        对输入的 3D 图像进行调整到指定形状
        Args:
            img: 输入图像张量 (D, H, W)
            target_shape: 目标形状 (target_D, target_H, target_W)
        Returns:
            调整后的张量
        """
        img = img.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
        img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=True)  # 线性插值到目标形状
        img = img.squeeze(0).squeeze(0)  # 移除批次和通道维度
        return img

    @staticmethod
    def collate_fn(batch):
        all_imgs, all_labels = zip(*batch)  # 只解包图像和标签
        all_imgs = torch.stack(all_imgs, dim=0)  # 堆叠所有图像
        all_labels = torch.as_tensor(all_labels)  # 转换标签为张量
        return all_imgs, all_labels  # 返回图像和标签

# 使用示例
if __name__ == "__main__":
    root_dir = r'/home/yuwenjing/data/tmp_data'

    # 创建数据集实例，直接调整到 (128, 128, 128)
    dataset = MyNRRDDataSet(root_dir, split='test')

    # 检查数据集总数
    print(f"数据集样本总数: {len(dataset)}")

    # 遍历整个数据集，并打印每个图像的形状和标签
    for i in range(len(dataset)):
        img, label = dataset[i]
        print(f"第 {i + 1} 个图像的形状: {img.shape}")
        print(f"第 {i + 1} 个图像的标签: {label}")
