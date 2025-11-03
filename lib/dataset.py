# # lib/dataset.py
# import os
# import glob
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from torchvision import transforms as T

# train_transform = T.Compose([
#     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     T.RandomGrayscale(p=0.1),
#     T.Resize((256, 256)),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# to_rgb_tensor = T.Compose([T.ToTensor(), T.Resize((256, 256))])
# to_depth_tensor = T.Compose([
#     T.Lambda(lambda x: torch.from_numpy(
#         np.array(x, dtype=np.float32) / np.array(x).max()
#     ).unsqueeze(0)),   # ← 扩维成 1×H×W
#     T.Resize((256, 256))
# ])

# class TrackingDataset(Dataset):
#     def __init__(self, data_root='/data/depth/aic25', split='train', k=20):
#         self.data_root = os.path.join(data_root, split)
#         self.k = k                                    # 每序列抽 k 帧
#         self.sequences = self._load_sequences()
#         # 预生成 (seq_idx, frame_idx) 列表
#         self.samples = []
#         for seq_i, seq in enumerate(self.sequences):
#             total = len(seq['rgb'])
#             # 均匀抽 k 帧（含第 0 帧做模板）
#             indices = np.linspace(0, total-1, self.k, dtype=int)
#             for frm_i in indices:
#                 self.samples.append((seq_i, frm_i))

#     def _load_sequences(self):
#         sequences = []
#         for txt in sorted(glob.glob(f"{self.data_root}/**/nlp.txt", recursive=True)):
#             seq_dir = os.path.dirname(txt)
#             rgb   = sorted(glob.glob(f"{seq_dir}/color/*.jpg"))
#             depth = sorted(glob.glob(f"{seq_dir}/depth/*.png"))
#             gt    = f"{seq_dir}/groundtruth_rect.txt"
#             if rgb and depth and os.path.exists(gt):
#                 sequences.append({'rgb': rgb, 'depth': depth, 'text': txt, 'gt': gt})
#         return sequences

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         seq_idx, frm_idx = self.samples[idx]
#         seq = self.sequences[seq_idx]

#         # 模板：第 0 帧（训练增强）
#         tpl_img = Image.open(seq['rgb'][0]).convert('RGB')
#         tpl_rgb = train_transform(tpl_img)
#         tpl_dep = to_depth_tensor(Image.open(seq['depth'][0]))

#         # 搜索图：当前抽帧（训练增强）
#         sr_img = Image.open(seq['rgb'][frm_idx]).convert('RGB')
#         sr_rgb = train_transform(sr_img)
#         sr_dep = to_depth_tensor(Image.open(seq['depth'][frm_idx]))
        
#         # GT 框
#         with open(seq['gt'], 'r') as f:
#             bboxes = [list(map(float, line.strip().split(','))) for line in f]
#         bbox = torch.tensor(bboxes[frm_idx], dtype=torch.float32)

#         # 文本
#         with open(seq['text'], 'r') as f:
#             text = f.read().strip()

#         return {
#             'template_rgb': tpl_rgb,
#             'template_depth': tpl_dep,
#             'search_rgb': sr_rgb,
#             'search_depth': sr_dep,
#             'text': text,
#             'bbox': bbox
#         }

# # ===== 供 test.py 调用的工具函数 =====
# import cv2

# def preprocess_image(path, tgt_size=256):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (tgt_size, tgt_size)).astype(np.float32) / 255.0
#     return torch.from_numpy(img).permute(2, 0, 1)

# def preprocess_depth(path, tgt_size=256):
#     dep = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
#     dep = cv2.resize(dep, (tgt_size, tgt_size))
#     dep = dep / dep.max()
#     return torch.from_numpy(dep).unsqueeze(0)

# def crop_template(img_tensor, bbox, pad=16, out_size=127):
#     x, y, w, h = map(int, bbox)
#     x, y, w, h = x - pad, y - pad, w + 2 * pad, h + 2 * pad
#     x = max(0, x);
#     y = max(0, y)
#     img = img_tensor.permute(1, 2, 0).numpy()
#     crop = img[y:y + h, x:x + w]
#     crop = cv2.resize(crop, (out_size, out_size))
#     return torch.from_numpy(crop).permute(2, 0, 1)

# def crop_search_region(img_tensor, prev_bbox, scale=2, out_size=255):
#     cx = prev_bbox[0] + prev_bbox[2] / 2
#     cy = prev_bbox[1] + prev_bbox[3] / 2
#     w = prev_bbox[2] * scale;
#     h = prev_bbox[3] * scale
#     x = int(cx - w / 2);
#     y = int(cy - h / 2)
#     x = max(0, x);
#     y = max(0, y)
#     img = img_tensor.permute(1, 2, 0).numpy()
#     crop = img[y:y + int(h), x:x + int(w)]
#     crop = cv2.resize(crop, (out_size, out_size))
#     return torch.from_numpy(crop).permute(2, 0, 1)

# def transform_bbox(pred_tensor, prev_bbox, scale=2):
#     dx, dy, dw, dh = pred_tensor.cpu().numpy().squeeze()
#     cx = prev_bbox[0] + prev_bbox[2] / 2
#     cy = prev_bbox[1] + prev_bbox[3] / 2
#     w = prev_bbox[2] * scale;
#     h = prev_bbox[3] * scale
#     x = (dx - w / 2) + cx;
#     y = (dy - h / 2) + cy
#     w = dw;
#     h = dh
#     return np.array([x, y, w, h])

# lib/dataset_fixed.py - 修复bbox尺度问题
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T

train_transform = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomGrayscale(p=0.1),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_rgb_tensor = T.Compose([T.ToTensor(), T.Resize((256, 256))])
to_depth_tensor = T.Compose([
    T.Lambda(lambda x: torch.from_numpy(
        np.array(x, dtype=np.float32) / (np.array(x).max() + 1e-6)
    ).unsqueeze(0)),
    T.Resize((256, 256))
])

class TrackingDataset(Dataset):
    def __init__(self, data_root='/data/depth/aic25', split='train', k=20):
        self.data_root = os.path.join(data_root, split)
        self.k = k
        self.sequences = self._load_sequences()
        self.samples = []
        for seq_i, seq in enumerate(self.sequences):
            total = len(seq['rgb'])
            indices = np.linspace(0, total-1, self.k, dtype=int)
            for frm_i in indices:
                self.samples.append((seq_i, frm_i))

    def _load_sequences(self):
        sequences = []
        for txt in sorted(glob.glob(f"{self.data_root}/**/nlp.txt", recursive=True)):
            seq_dir = os.path.dirname(txt)
            rgb   = sorted(glob.glob(f"{seq_dir}/color/*.jpg"))
            depth = sorted(glob.glob(f"{seq_dir}/depth/*.png"))
            gt    = f"{seq_dir}/groundtruth_rect.txt"
            if rgb and depth and os.path.exists(gt):
                sequences.append({'rgb': rgb, 'depth': depth, 'text': txt, 'gt': gt})
        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, frm_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        # ===== 关键修复：获取原图尺寸 =====
        tpl_img = Image.open(seq['rgb'][0]).convert('RGB')
        orig_w, orig_h = tpl_img.size  # 原始尺寸，比如(1920, 1080)
        
        # 处理图像（resize到256x256）
        tpl_rgb = train_transform(tpl_img)
        # 修正：先 Resize 再 to tensor 的深度流程
        tpl_dep = T.Resize((256, 256))(Image.open(seq['depth'][0]))
        tpl_dep = torch.from_numpy(np.array(tpl_dep, dtype=np.float32) / (np.array(tpl_dep).max() + 1e-6)).unsqueeze(0)

        sr_img = Image.open(seq['rgb'][frm_idx]).convert('RGB')
        sr_rgb = train_transform(sr_img)
        sr_dep_img = T.Resize((256, 256))(Image.open(seq['depth'][frm_idx]))
        sr_dep = torch.from_numpy(np.array(sr_dep_img, dtype=np.float32) / (np.array(sr_dep_img).max() + 1e-6)).unsqueeze(0)

        # ===== 关键：GT框同步缩放到256尺度并归一化到[0,1] =====
        with open(seq['gt'], 'r') as f:
            bboxes = [list(map(float, line.strip().split(','))) for line in f]
        bbox_orig = bboxes[min(frm_idx, len(bboxes)-1)]  # 在原图尺度

        # 缩放到256x256
        scale_x = 256.0 / orig_w
        scale_y = 256.0 / orig_h
        bbox_scaled = [
            bbox_orig[0] * scale_x,
            bbox_orig[1] * scale_y,
            bbox_orig[2] * scale_x,
            bbox_orig[3] * scale_y
        ]

        # 归一化到 [0,1] —— 与模型 sigmoid 输出对齐
        bbox_norm = [bbox_scaled[0] / 256.0,
                    bbox_scaled[1] / 256.0,
                    bbox_scaled[2] / 256.0,
                    bbox_scaled[3] / 256.0]
        bbox = torch.tensor(bbox_norm, dtype=torch.float32)

        with open(seq['text'], 'r') as f:
            text = f.read().strip()
            
        return {
            'template_rgb': tpl_rgb,
            'template_depth': tpl_dep,
            'search_rgb': sr_rgb,
            'search_depth': sr_dep,
            'text': text,
            'bbox': bbox,  # 现在是归一化到 [0,1]
            'orig_size': (orig_w, orig_h)
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
    dep = dep / (dep.max() + 1e-6)
    return torch.from_numpy(dep).unsqueeze(0)