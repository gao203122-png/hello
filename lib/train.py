# ============================================
# 8. 快速训练脚本
# ============================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.tracker import RGBDTextTracker
from lib.dataset import TrackingDataset

def quick_train():
    """简化的训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDTextTracker().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 1. 指定正确路径 & split
    train_loader = DataLoader(
        TrackingDataset(data_root='/data/depth/aic25', split='train'),
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    os.makedirs('outputs/exp1/ckpt', exist_ok=True)

    for epoch in range(50):
        model.train()
        epoch_loss = 0.
        for batch in train_loader:
            # 2. 统一 to(device)
            tpl_rgb   = batch['template_rgb'].to(device)
            tpl_dep   = batch['template_depth'].to(device)
            srh_rgb   = batch['search_rgb'].to(device)
            srh_dep   = batch['search_depth'].to(device)
            bbox_gt   = batch['bbox'].to(device)

            pred_bbox, _ = model(tpl_rgb, tpl_dep, batch['text'], srh_rgb, srh_dep)

            loss = F.l1_loss(pred_bbox, bbox_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:02d}  Avg-Loss: {avg_loss:.6f}")

        # 3. 每 10 轮保存权重（供后面推理用）
        if epoch % 10 == 0 or epoch == 49:
            ckpt_path = f'outputs/exp1/ckpt/epoch{epoch}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  →  saved {ckpt_path}")

    # 训练完把「最佳」链接到 best.pth，方便推理脚本直接加载
    torch.save(model.state_dict(), 'best.pth')
    print('All done! best.pth ready for inference.')

if __name__ == "__main__":
    quick_train()
