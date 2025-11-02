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


#

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.tracker import RGBDTextTracker
from lib.dataset import TrackingDataset


def iou_loss(pred_bbox, gt_bbox):
    """IoU损失（更适合跟踪任务）"""
    x1_pred, y1_pred = pred_bbox[:, 0], pred_bbox[:, 1]
    x2_pred, y2_pred = pred_bbox[:, 0] + pred_bbox[:, 2], pred_bbox[:, 1] + pred_bbox[:, 3]
    
    x1_gt, y1_gt = gt_bbox[:, 0], gt_bbox[:, 1]
    x2_gt, y2_gt = gt_bbox[:, 0] + gt_bbox[:, 2], gt_bbox[:, 1] + gt_bbox[:, 3]
    
    # 计算交集
    x1_inter = torch.max(x1_pred, x1_gt)
    y1_inter = torch.max(y1_pred, y1_gt)
    x2_inter = torch.min(x2_pred, x2_gt)
    y2_inter = torch.min(y2_pred, y2_gt)
    
    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
    
    # 计算并集
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = pred_area + gt_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return 1 - iou.mean()

def quick_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDTextTracker().to(device)
    
    # ===== 关键改进：使用AdamW + 余弦退火 =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    train_loader = DataLoader(
        TrackingDataset(data_root='/data/depth/aic25', split='train', k=30),  # 增加采样帧数
        batch_size=8,  # 减小batch保证显存
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    os.makedirs('outputs/exp1/ckpt', exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0.
        
        for batch in train_loader:
            tpl_rgb = batch['template_rgb'].to(device)
            tpl_dep = batch['template_depth'].to(device)
            srh_rgb = batch['search_rgb'].to(device)
            srh_dep = batch['search_depth'].to(device)
            bbox_gt = batch['bbox'].to(device)
            
            pred_bbox, response_map = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)
            
            # ===== 关键改进：组合损失 =====
            l1_loss = F.l1_loss(pred_bbox, bbox_gt)
            iou_loss_val = iou_loss(pred_bbox, bbox_gt)
            loss = l1_loss + 2.0 * iou_loss_val  # IoU权重更大
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:02d}  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f"  → Best model saved (loss: {best_loss:.4f})")
        
        if epoch % 5 == 0 or epoch == 49:
            torch.save(model.state_dict(), f'outputs/exp1/ckpt/epoch{epoch}.pth')
    
    print('Training done!')

if __name__ == "__main__":
    quick_train()