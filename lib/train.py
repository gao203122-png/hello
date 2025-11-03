# ============================================
# 8. 快速训练脚本
# ============================================

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.tracker import RGBDTextTracker
# from lib.dataset import TrackingDataset

# def quick_train():
#     """简化的训练流程"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = RGBDTextTracker().to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

#     # 1. 指定正确路径 & split
#     train_loader = DataLoader(
#         TrackingDataset(data_root='/data/depth/aic25', split='train'),
#         batch_size=16,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )

#     os.makedirs('outputs/exp1/ckpt', exist_ok=True)

#     for epoch in range(50):
#         model.train()
#         epoch_loss = 0.
#         for batch in train_loader:
#             # 2. 统一 to(device)
#             tpl_rgb   = batch['template_rgb'].to(device)
#             tpl_dep   = batch['template_depth'].to(device)
#             srh_rgb   = batch['search_rgb'].to(device)
#             srh_dep   = batch['search_depth'].to(device)
#             bbox_gt   = batch['bbox'].to(device)

#             pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)

#             loss = F.l1_loss(pred_bbox, bbox_gt)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         avg_loss = epoch_loss / len(train_loader)
#         print(f"Epoch {epoch:02d}  Avg-Loss: {avg_loss:.6f}")

#         # 3. 每 10 轮保存权重（供后面推理用）
#         if epoch % 5 == 0 or epoch == 49:
#             ckpt_path = f'outputs/exp1/ckpt2/epoch{epoch}.pth'
#             torch.save(model.state_dict(), ckpt_path)
#             print(f"  →  saved {ckpt_path}")

#     # 训练完把「最佳」链接到 best.pth，方便推理脚本直接加载
#     torch.save(model.state_dict(), 'best.pth')
#     print('All done! best.pth ready for inference.')

# if __name__ == "__main__":
#     quick_train()
# lib/train.py
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.tracker import RGBDTextTracker
# from lib.dataset import TrackingDataset

# def giou_loss(pred_bbox, gt_bbox):
#     """GIoU损失(更适合跟踪)"""
#     # 转换为xyxy格式
#     pred_x1, pred_y1 = pred_bbox[:, 0], pred_bbox[:, 1]
#     pred_x2, pred_y2 = pred_x1 + pred_bbox[:, 2], pred_y1 + pred_bbox[:, 3]
    
#     gt_x1, gt_y1 = gt_bbox[:, 0], gt_bbox[:, 1]
#     gt_x2, gt_y2 = gt_x1 + gt_bbox[:, 2], gt_y1 + gt_bbox[:, 3]
    
#     # 交集
#     inter_x1 = torch.max(pred_x1, gt_x1)
#     inter_y1 = torch.max(pred_y1, gt_y1)
#     inter_x2 = torch.min(pred_x2, gt_x2)
#     inter_y2 = torch.min(pred_y2, gt_y2)
    
#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
#     # 并集
#     pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
#     gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
#     union_area = pred_area + gt_area - inter_area
    
#     iou = inter_area / (union_area + 1e-7)
    
#     # 最小外接矩形
#     enclose_x1 = torch.min(pred_x1, gt_x1)
#     enclose_y1 = torch.min(pred_y1, gt_y1)
#     enclose_x2 = torch.max(pred_x2, gt_x2)
#     enclose_y2 = torch.max(pred_y2, gt_y2)
    
#     enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
#     # GIoU
#     giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
#     return 1 - giou.mean()

# def quick_train():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # # ✅ 打印初始显存
#     # print(f"[INFO] 初始显存: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    
#     model = RGBDTextTracker().to(device)

#     #  # ✅ 打印模型显存
#     # print(f"[INFO] 模型加载后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
#     train_loader = DataLoader(
#         TrackingDataset(data_root='/data/depth/aic25', split='train', k=40),
#         batch_size=16,  # 减小batch适应Transformer
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True
#     )
    
#     os.makedirs('outputs/exp_jvg/ckpt', exist_ok=True)
#     best_loss = float('inf')
    
#     for epoch in range(50):
#         model.train()
#         epoch_loss = 0.
        
#         for batch_idx, batch in enumerate(train_loader):
#             tpl_rgb = batch['template_rgb'].to(device)
#             tpl_dep = batch['template_depth'].to(device)
#             srh_rgb = batch['search_rgb'].to(device)
#             srh_dep = batch['search_depth'].to(device)
#             bbox_gt = batch['bbox'].to(device)
            
#             pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)
            
#             # === 组合损失 ===
#             l1_loss = F.l1_loss(pred_bbox, bbox_gt)
#             giou_loss_val = giou_loss(pred_bbox, bbox_gt)
#             loss = l1_loss + 3.0 * giou_loss_val  # GIoU权重更大
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
            
#             epoch_loss += loss.item()
            
#             # if batch_idx % 50 == 0:
#             #     print(f"  Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
#         scheduler.step()
#         avg_loss = epoch_loss / len(train_loader)
#         print(f"Epoch {epoch:02d}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
        
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), 'best.pth')
#             print(f"  → Best model saved (loss: {best_loss:.4f})")
        
#         if epoch % 5 == 0 or epoch == 49:
#             torch.save(model.state_dict(), f'outputs/exp_jvg/ckpt/epoch{epoch}.pth')
    
#     print('Training done!')

# if __name__ == "__main__":
#     quick_train()

# lib/train.py
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.tracker import RGBDTextTracker
# from lib.dataset import TrackingDataset

# def giou_loss(pred_bbox, gt_bbox):
#     """GIoU损失"""
#     # ✅ 确保输入格式：[x,y,w,h]
#     pred_x1 = pred_bbox[:, 0]
#     pred_y1 = pred_bbox[:, 1]
#     pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2]
#     pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3]
    
#     gt_x1 = gt_bbox[:, 0]
#     gt_y1 = gt_bbox[:, 1]
#     gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2]
#     gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3]
    
#     # 交集
#     inter_x1 = torch.max(pred_x1, gt_x1)
#     inter_y1 = torch.max(pred_y1, gt_y1)
#     inter_x2 = torch.min(pred_x2, gt_x2)
#     inter_y2 = torch.min(pred_y2, gt_y2)
    
#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
#     pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
#     gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
#     union_area = pred_area + gt_area - inter_area + 1e-7
    
#     iou = inter_area / union_area
    
#     # 最小外接矩形
#     enclose_x1 = torch.min(pred_x1, gt_x1)
#     enclose_y1 = torch.min(pred_y1, gt_y1)
#     enclose_x2 = torch.max(pred_x2, gt_x2)
#     enclose_y2 = torch.max(pred_y2, gt_y2)
    
#     enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
    
#     giou = iou - (enclose_area - union_area) / enclose_area
    
#     return 1 - giou.mean()

# def quick_train():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = RGBDTextTracker().to(device)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
#     # ✅ 关键：确保dataset返回256x256尺度的bbox
#     train_loader = DataLoader(
#         TrackingDataset(data_root='/data/depth/aic25', split='train', k=40),
#         batch_size=8,  # ✅ 提高到24
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True
#     )
    
#     os.makedirs('outputs/exp_final/ckpt', exist_ok=True)
#     best_loss = float('inf')
    
#     for epoch in range(50):
#         model.train()
#         epoch_loss = 0.
        
#         for batch_idx, batch in enumerate(train_loader):
#             tpl_rgb = batch['template_rgb'].to(device)
#             tpl_dep = batch['template_depth'].to(device)
#             srh_rgb = batch['search_rgb'].to(device)
#             srh_dep = batch['search_depth'].to(device)
#             bbox_gt = batch['bbox'].to(device)
            
#             # ✅ 关键：确保GT bbox也是256x256尺度
#             # 如果GT是原图尺度，需要缩放
#             # 假设原图是1920x1080，需要缩放到256x256
#             # bbox_gt_scaled = bbox_gt * (256.0 / 原图尺寸)
            
#             pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)
            
#             # === 组合损失 ===
#             l1_loss = F.l1_loss(pred_bbox, bbox_gt)
#             giou_loss_val = giou_loss(pred_bbox, bbox_gt)
            
#             # ✅ 添加尺度约束
#             w_penalty = torch.mean(torch.abs(pred_bbox[:, 2] - bbox_gt[:, 2]))
#             h_penalty = torch.mean(torch.abs(pred_bbox[:, 3] - bbox_gt[:, 3]))
            
#             loss = l1_loss + 2.0 * giou_loss_val + 0.5 * (w_penalty + h_penalty)
            
#             optimizer.zero_grad()
#             # loss.backward()
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             # optimizer.step()
            
#             epoch_loss += loss.item()
            
#             if batch_idx % 100 == 0:
#                 print(f"  Batch {batch_idx} Loss: {loss.item():.4f} L1: {l1_loss.item():.2f} GIoU: {giou_loss_val.item():.2f}")
        
#         scheduler.step()
#         avg_loss = epoch_loss / len(train_loader)
#         print(f"Epoch {epoch:02d}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
        
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), 'best.pth')
#             print(f"  → Best model saved (loss: {best_loss:.4f})")
        
#         if epoch % 5 == 0 or epoch == 49:
#             torch.save(model.state_dict(), f'outputs/exp_final/ckpt/epoch{epoch}.pth')
    
#     print('Training done!')

# if __name__ == "__main__":
#     quick_train()

# lib/train_final.py - 简化稳定的训练脚本
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.tracker import RGBDTextTracker
from lib.dataset import TrackingDataset

def giou_loss(pred_bbox, gt_bbox):
    """GIoU损失 - 256尺度"""
    pred_x1 = pred_bbox[:, 0]
    pred_y1 = pred_bbox[:, 1]
    pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2]
    pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3]
    
    gt_x1 = gt_bbox[:, 0]
    gt_y1 = gt_bbox[:, 1]
    gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2]
    gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3]
    
    # 交集
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area + 1e-7
    
    iou = inter_area / union_area
    
    # 最小外接矩形
    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_y1 = torch.min(pred_y1, gt_y1)
    enclose_x2 = torch.max(pred_x2, gt_x2)
    enclose_y2 = torch.max(pred_y2, gt_y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
    
    giou = iou - (enclose_area - union_area) / enclose_area
    
    return 1 - giou.mean()

def quick_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDTextTracker().to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Dataset（现在GT已经是256尺度）
    train_loader = DataLoader(
        TrackingDataset(data_root='/data/depth/aic25', split='train', k=30),
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    os.makedirs('outputs/exp_final/ckpt', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0.
        epoch_giou = 0.
        epoch_l1 = 0.
        
        for batch_idx, batch in enumerate(train_loader):
            tpl_rgb = batch['template_rgb'].to(device)
            tpl_dep = batch['template_depth'].to(device)
            srh_rgb = batch['search_rgb'].to(device)
            srh_dep = batch['search_depth'].to(device)
            bbox_gt = batch['bbox'].to(device)  # 已经是256尺度
            
            # 前向传播
            pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)
            
            # ===== 多任务损失 =====
            # 1. GIoU损失（主要）
            giou_loss_val = giou_loss(pred_bbox, bbox_gt)
            
            # 2. L1损失（辅助，帮助快速收敛）
            l1_loss = F.smooth_l1_loss(pred_bbox, bbox_gt)
            
            # 3. 中心点损失（提升定位精度）
            pred_center = pred_bbox[:, :2] + pred_bbox[:, 2:] / 2
            gt_center = bbox_gt[:, :2] + bbox_gt[:, 2:] / 2
            center_loss = F.mse_loss(pred_center, gt_center)
            
            # 组合损失
            loss = 2.0 * giou_loss_val + 1.0 * l1_loss + 0.5 * center_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_giou += giou_loss_val.item()
            epoch_l1 += l1_loss.item()
            
            if batch_idx % 1000 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} "
                      f"GIoU: {giou_loss_val.item():.4f} "
                      f"L1: {l1_loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        avg_giou = epoch_giou / len(train_loader)
        avg_l1 = epoch_l1 / len(train_loader)
        
        # print(f"\n{'='*60}")
        # print(f"Epoch {epoch:02d}  "
        #       f"Loss: {avg_loss:.4f}  "
        #       f"GIoU: {avg_giou:.4f}  "
        #       f"L1: {avg_l1:.4f}  "
        #       f"LR: {scheduler.get_last_lr()[0]:.6f}")
        # print(f"{'='*60}\n")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f"  ✅ Best model saved (loss: {best_loss:.4f})")
        
        # 定期checkpoint
        if epoch % 5 == 0 or epoch == 49:
            torch.save(model.state_dict(), f'outputs/exp_final/ckpt/epoch{epoch}.pth')
    
    print('\n🎉 Training done!')

if __name__ == "__main__":
    quick_train()

# ↑能跑通但单卡
# lib/train_final.py
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from models.tracker import RGBDTextTracker
# from lib.dataset import TrackingDataset


# def giou_loss(pred_bbox, gt_bbox):
#     pred_x1 = pred_bbox[:, 0]
#     pred_y1 = pred_bbox[:, 1]
#     pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2]
#     pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3]
#     gt_x1 = gt_bbox[:, 0]
#     gt_y1 = gt_bbox[:, 1]
#     gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2]
#     gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3]
#     inter_x1 = torch.max(pred_x1, gt_x1)
#     inter_y1 = torch.max(pred_y1, gt_y1)
#     inter_x2 = torch.min(pred_x2, gt_x2)
#     inter_y2 = torch.min(pred_y2, gt_y2)
#     inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
#     pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
#     gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
#     union_area = pred_area + gt_area - inter_area + 1e-7
#     iou = inter_area / union_area
#     enclose_x1 = torch.min(pred_x1, gt_x1)
#     enclose_y1 = torch.min(pred_y1, gt_y1)
#     enclose_x2 = torch.max(pred_x2, gt_x2)
#     enclose_y2 = torch.max(pred_y2, gt_y2)
#     enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7
#     giou = iou - (enclose_area - union_area) / enclose_area
#     return 1 - giou.mean()


# def quick_train():
#     # ✅ 多卡安全初始化（稳定 cudnn）
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # ✅ 模型加载与多卡封装
#     model = RGBDTextTracker().to(device)
#     if torch.cuda.device_count() > 1:
#         print(f"🔧 Using {torch.cuda.device_count()} GPUs for training")
#         model = torch.nn.DataParallel(model)  # ✅ 仅添加这一行
#     else:
#         print("⚙️ Using single GPU")

#     # 优化器与调度器
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

#     # 数据集加载
#     train_loader = DataLoader(
#         TrackingDataset(data_root='/data/depth/aic25', split='train', k=30),
#         batch_size=32,              # ✅ 建议改成能被GPU数整除的batch size
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True
#     )

#     os.makedirs('outputs/exp_final/ckpt', exist_ok=True)
#     best_loss = float('inf')

#     for epoch in range(50):
#         model.train()
#         epoch_loss = epoch_giou = epoch_l1 = 0.0

#         for batch_idx, batch in enumerate(train_loader):
#             tpl_rgb = batch['template_rgb'].to(device)
#             tpl_dep = batch['template_depth'].to(device)
#             srh_rgb = batch['search_rgb'].to(device)
#             srh_dep = batch['search_depth'].to(device)
#             bbox_gt = batch['bbox'].to(device)

#             # 前向传播
#             pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)

#             # 多任务损失
#             giou_loss_val = giou_loss(pred_bbox, bbox_gt)
#             l1_loss = F.smooth_l1_loss(pred_bbox, bbox_gt)
#             pred_center = pred_bbox[:, :2] + pred_bbox[:, 2:] / 2
#             gt_center = bbox_gt[:, :2] + bbox_gt[:, 2:] / 2
#             center_loss = F.mse_loss(pred_center, gt_center)
#             loss = 2.0 * giou_loss_val + 1.0 * l1_loss + 0.5 * center_loss

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_giou += giou_loss_val.item()
#             epoch_l1 += l1_loss.item()

#             if batch_idx % 1000 == 0:
#                 print(f"  Batch {batch_idx}/{len(train_loader)} "
#                       f"Loss: {loss.item():.4f} "
#                       f"GIoU: {giou_loss_val.item():.4f} "
#                       f"L1: {l1_loss.item():.4f}")

#         scheduler.step()
#         avg_loss = epoch_loss / len(train_loader)
#         avg_giou = epoch_giou / len(train_loader)
#         avg_l1 = epoch_l1 / len(train_loader)

#         # 保存最佳模型
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save(model.state_dict(), 'best.pth')
#             print(f"  ✅ Best model saved (loss: {best_loss:.4f})")

#         if epoch % 5 == 0 or epoch == 49:
#             torch.save(model.state_dict(), f'outputs/exp_final/ckpt/epoch{epoch}.pth')

#     print('\n🎉 Training done!')


# if __name__ == "__main__":
#     quick_train()
