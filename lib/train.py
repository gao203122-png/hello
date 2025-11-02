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
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.tracker import RGBDTextTracker
from lib.dataset import TrackingDataset

def giou_loss(pred_bbox, gt_bbox):
    """GIoU损失(更适合跟踪)"""
    # 转换为xyxy格式
    pred_x1, pred_y1 = pred_bbox[:, 0], pred_bbox[:, 1]
    pred_x2, pred_y2 = pred_x1 + pred_bbox[:, 2], pred_y1 + pred_bbox[:, 3]
    
    gt_x1, gt_y1 = gt_bbox[:, 0], gt_bbox[:, 1]
    gt_x2, gt_y2 = gt_x1 + gt_bbox[:, 2], gt_y1 + gt_bbox[:, 3]
    
    # 交集
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 并集
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    
    # 最小外接矩形
    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_y1 = torch.min(pred_y1, gt_y1)
    enclose_x2 = torch.max(pred_x2, gt_x2)
    enclose_y2 = torch.max(pred_y2, gt_y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    return 1 - giou.mean()

def quick_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDTextTracker().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    train_loader = DataLoader(
        TrackingDataset(data_root='/data/depth/aic25', split='train', k=40),
        batch_size=4,  # 减小batch适应Transformer
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    os.makedirs('outputs/exp_jvg/ckpt', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0.
        
        for batch_idx, batch in enumerate(train_loader):
            tpl_rgb = batch['template_rgb'].to(device)
            tpl_dep = batch['template_depth'].to(device)
            srh_rgb = batch['search_rgb'].to(device)
            srh_dep = batch['search_depth'].to(device)
            bbox_gt = batch['bbox'].to(device)
            
            pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)
            
            # === 组合损失 ===
            l1_loss = F.l1_loss(pred_bbox, bbox_gt)
            giou_loss_val = giou_loss(pred_bbox, bbox_gt)
            loss = l1_loss + 3.0 * giou_loss_val  # GIoU权重更大
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:02d}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f"  → Best model saved (loss: {best_loss:.4f})")
        
        if epoch % 5 == 0 or epoch == 49:
            torch.save(model.state_dict(), f'outputs/exp_jvg/ckpt/epoch{epoch}.pth')
    
    print('Training done!')

if __name__ == "__main__":
    quick_train()