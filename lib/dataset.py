# lib/dataset.py
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T

to_rgb_tensor = T.Compose([T.ToTensor(), T.Resize((256, 256))])
to_depth_tensor = T.Compose([
    T.Lambda(lambda x: torch.from_numpy(
        np.array(x, dtype=np.float32) / np.array(x).max()
    ).unsqueeze(0)),   # ← 扩维成 1×H×W
    T.Resize((256, 256))
])

class TrackingDataset(Dataset):
    def __init__(self, data_root='/data/depth/aic25', split='train'):
        self.data_root = os.path.join(data_root, split)
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        sequences = []
        # 先拿到所有 nlp.txt，再反推序列根目录
        for text_path in sorted(glob.glob(f"{self.data_root}/**/nlp.txt", recursive=True)):
            seq_dir     = os.path.dirname(text_path)
            rgb_paths   = sorted(glob.glob(f"{seq_dir}/color/*.jpg"))
            depth_paths = sorted(glob.glob(f"{seq_dir}/depth/*.png"))
            gt_path     = f"{seq_dir}/groundtruth_rect.txt"   # ← 只改这里
            # ↓↓↓ 调试：看哪一步为空
            print(f"DIR {seq_dir}  color={len(rgb_paths)}  depth={len(depth_paths)}  gt={os.path.exists(gt_path)}")
            if rgb_paths and depth_paths and os.path.exists(text_path) and os.path.exists(gt_path):
                sequences.append({
                    'rgb': rgb_paths,
                    'depth': depth_paths,
                    'text': text_path,
                    'gt': gt_path,
                    'name': os.path.basename(seq_dir.rstrip('/'))
                })
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 文本
        with open(seq['text'], 'r') as f:
            text = f.read().strip()

        # GT 框
        with open(seq['gt'], 'r') as f:
            bboxes = [list(map(float, line.strip().split(','))) for line in f]

        # 模板：第一帧
        template_rgb   = Image.open(seq['rgb'][0]).convert('RGB')
        template_depth = Image.open(seq['depth'][0])

        # 搜索：随机帧
        search_idx = np.random.randint(1, len(seq['rgb']))
        search_rgb   = Image.open(seq['rgb'][search_idx]).convert('RGB')
        search_depth = Image.open(seq['depth'][search_idx])

        # ↓↓↓ 分开转 tensor ↓↓↓
        template_rgb   = to_rgb_tensor(template_rgb)
        template_depth = to_depth_tensor(template_depth)
        search_rgb     = to_rgb_tensor(search_rgb)
        search_depth   = to_depth_tensor(search_depth)
        
        # 返回张量
        return {
            'template_rgb':   template_rgb,      # 已经是 Tensor
            'template_depth': template_depth,    # 已经是 Tensor
            'search_rgb':     search_rgb,        # 已经是 Tensor
            'search_depth':   search_depth,      # 已经是 Tensor
            'text': text,
            'bbox': torch.tensor(bboxes[search_idx], dtype=torch.float32)
        }


# ===== 供 test.py 调用的工具函数 =====
import cv2

def preprocess_image(path, tgt_size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (tgt_size, tgt_size)).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)

def preprocess_depth(path, tgt_size=256):
    dep = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    dep = cv2.resize(dep, (tgt_size, tgt_size))
    dep = dep / dep.max()
    return torch.from_numpy(dep).unsqueeze(0)

def crop_template(img_tensor, bbox, pad=16, out_size=127):
    x, y, w, h = map(int, bbox)
    x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
    x = max(0, x);
    y = max(0, y)
    img = img_tensor.permute(1, 2, 0).numpy()
    crop = img[y:y + h, x:x + w]
    crop = cv2.resize(crop, (out_size, out_size))
    return torch.from_numpy(crop).permute(2, 0, 1)

def crop_search_region(img_tensor, prev_bbox, scale=2, out_size=255):
    cx = prev_bbox[0] + prev_bbox[2] / 2
    cy = prev_bbox[1] + prev_bbox[3] / 2
    w = prev_bbox[2] * scale;
    h = prev_bbox[3] * scale
    x = int(cx - w / 2);
    y = int(cy - h / 2)
    x = max(0, x);
    y = max(0, y)
    img = img_tensor.permute(1, 2, 0).numpy()
    crop = img[y:y + int(h), x:x + int(w)]
    crop = cv2.resize(crop, (out_size, out_size))
    return torch.from_numpy(crop).permute(2, 0, 1)

def transform_bbox(pred_tensor, prev_bbox, scale=2):
    dx, dy, dw, dh = pred_tensor.cpu().numpy().squeeze()
    cx = prev_bbox[0] + prev_bbox[2] / 2
    cy = prev_bbox[1] + prev_bbox[3] / 2
    w = prev_bbox[2] * scale;
    h = prev_bbox[3] * scale
    x = (dx - w / 2) + cx;
    y = (dy - h / 2) + cy
    w = dw;
    h = dh
    return np.array([x, y, w, h])