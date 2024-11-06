import os
import argparse
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import Compose, RandFlip, RandRotate90, ToTensor
from tqdm import tqdm

from test_dataset import MyNRRDDataSet  # 使用您自定义的数据集类
from sup import SwinUNETRClassifier as create_model  # 确保模型为3D版本
from utils import train_one_epoch, evaluate
from torch.cuda.amp import GradScaler


def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 初始化混合精度梯度缩放器
    scaler = torch.amp.GradScaler()

    # 创建保存模型的目录
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # 初始化 TensorBoard 写入器
    tb_writer = SummaryWriter()

    # 定义数据增强和预处理
    data_transform = Compose([
        RandFlip(spatial_axis=[1], prob=0.5),  # 随机X轴翻转
        RandFlip(spatial_axis=[0], prob=0.5),  # 随机Y轴翻转
        RandRotate90(prob=0.5, max_k=3, spatial_axes=(0, 1)),  # 随机90度旋转
        ToTensor()  # 将数据转换为Tensor
    ])

    # 实例化训练数据集
    train_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='train',
        transform=data_transform,
        target_shape=(224,224,224)  # 修改后的目标形状
    )

    # 实例化验证数据集
    val_dataset = MyNRRDDataSet(
        root_dir=args.data_path,
        split='test',
        transform=ToTensor(),  # 验证集通常不需要数据增强
        target_shape=(224,224,224)  # 修改后的目标形状
    )

    # 定义数据加载器
    batch_size = args.batch_size
    nw = 8
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )

    # 初始化模型
    model = create_model(num_classes=args.num_classes).to(device)
    print("Model's state_dict keys:")
    for key in model.state_dict().keys():
        print(key)

    # 载入预训练权重（如果提供）
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' does not exist."
        checkpoint = torch.load(args.weights, map_location=device)
        
        # 提取权重字典
        if 'net' in checkpoint:
            weights_dict = checkpoint['net']
        else:
            weights_dict = checkpoint

        # 移除 `module.` 前缀
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}

        # 移除 `backbone.` 前缀（如果存在）
        weights_dict = {k.replace('backbone.', ''): v for k, v in weights_dict.items()}

        # 删除解码器相关的键
        decoder_keys = [k for k in weights_dict.keys() if k.startswith('decoder')]
        for k in decoder_keys:
            del weights_dict[k]

        # 处理输入通道数不匹配的问题
        conv1_key = 'swinViT.patch_embed.proj.weight'
        if conv1_key in weights_dict and weights_dict[conv1_key].shape[1] != model.swinViT.patch_embed.proj.weight.shape[1]:
            if weights_dict[conv1_key].shape[1] == 3 and model.swinViT.patch_embed.proj.weight.shape[1] == 1:
                # 将预训练权重在通道维度上取平均
                weights_dict[conv1_key] = weights_dict[conv1_key].mean(dim=1, keepdim=True)

        # 加载预训练权重
        load_info = model.load_state_dict(weights_dict, strict=False)

        # 打印加载信息
        print("Successfully loaded pre-trained weights.")
        print(f"Missing keys: {load_info.missing_keys}")
        print(f"Unexpected keys: {load_info.unexpected_keys}")

        # 计算加载的参数比例
        loaded_params = len(model.state_dict()) - len(load_info.missing_keys)
        total_params = len(model.state_dict())
        load_percentage = (loaded_params / total_params) * 100
        print(f"Percentage of loaded weights: {load_percentage:.2f}%")




    # 冻结部分层（如果指定）
    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "fc" not in name:  # 只训练最后的全连接层
                param.requires_grad = False
            else:
                print(f"training {name}")

    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 定义学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # 训练参数
    num_epochs = args.epochs
    best_val_loss = float('inf')
    early_stopping_patience = 30
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler
        )

        # 验证
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch,
            num_classes=args.num_classes
        )

        # 记录到 TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        
        # 调整学习率（现在 val_loss 已经定义）
        scheduler.step(val_loss)

        # 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join("./weights", "best_model.pth"))
            print(f"Best model saved at epoch {epoch} with Val Loss={val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        # 早停检查
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break

        # 可选：打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # 保存每个epoch的模型
        torch.save(model.state_dict(), os.path.join("./weights", f"model-{epoch}.pth"))

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join("./weights", "final_model.pth"))
    print("Training complete. Final model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Vision Transformer for 3D Classification')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate')
    
    # 数据集所在根目录
    parser.add_argument('--data_path', type=str, default="/home/yuwenjing/data/sm_label", help='Path to the dataset')
    
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='/home/yuwenjing/DeepLearning/MN/supervised_suprem_swinunetr_2100.pth', help='Initial weights path')
    
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=False, help='Freeze layers except head and pre_logits')
    
    # 设备选择
    parser.add_argument('--device', type=str, default='cuda:0', help='Device ID (e.g., cuda:0 or cpu)')
    
    opt = parser.parse_args()
    
    main(opt)
