# # ============================================
# # 6. test.py - 测试推理
# # ============================================
# import os
# import glob
# import torch
# import cv2
# import numpy as np
# from models.tracker import RGBDTextTracker
# from models.text_encoder import TextEncoder
# from lib.dataset import preprocess_image, preprocess_depth, crop_template, crop_search_region, transform_bbox

# def test_sequence(model, sequence_path):
#     """测试单个序列"""
#     # 读取第一帧和文本
#     rgb_list = sorted(glob.glob(f"{sequence_path}/rgb/*.jpg"))
#     depth_list = sorted(glob.glob(f"{sequence_path}/depth/*.png"))
    
#     with open(f"{sequence_path}/text.txt", 'r') as f:
#         text = f.read().strip()
    
#     # 读取第一帧的bbox
#     with open(f"{sequence_path}/groundtruth.txt", 'r') as f:
#         init_bbox = list(map(float, f.readline().strip().split(',')))
    
#     results = []
#     model.eval()
    
#     for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
#         rgb = preprocess_image(rgb_path)
#         depth = preprocess_depth(depth_path)
        
#         if i == 0:
#             # 初始化模板
#             template_rgb = crop_template(rgb, init_bbox)
#             template_depth = crop_template(depth, init_bbox)
        
#         # 提取搜索区域
#         search_rgb = crop_search_region(rgb, prev_bbox)
#         search_depth = crop_search_region(depth, prev_bbox)
        
#         # 预测
#         with torch.no_grad():
#             pred_bbox, _ = model(
#                 template_rgb, template_depth, [text],
#                 search_rgb, search_depth
#             )
        
#         # 坐标转换回原图
#         bbox = transform_bbox(pred_bbox, prev_bbox)
#         results.append(bbox)
#         prev_bbox = bbox
    
#     return results


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs('results', exist_ok=True)   

#     model = RGBDTextTracker().to(device)
#     model.load_state_dict(torch.load('best.pth', map_location=device))
#     seqs = [d for d in os.listdir('data/test') if os.path.isdir(os.path.join('data/test', d))]
#     for s in seqs:
#         res = test_sequence(model, f'data/test/{s}')
#         np.savetxt(f'results/{s}.txt', res, fmt='%.2f %.2f %.2f %.2f')
#     print('All results saved to results/')



import os
import glob
import torch
import numpy as np
from models.tracker import RGBDTextTracker
# 确保这些函数被正确导入
from lib.dataset import preprocess_image, preprocess_depth, crop_template, crop_search_region, transform_bbox

def test_sequence(model, sequence_path):
   """测试单个序列（已修正，适配比赛要求）"""
   
   # 1. 读取文件列表
   rgb_list = sorted(glob.glob(f"{sequence_path}/color/*.jpg"))
   depth_list = sorted(glob.glob(f"{sequence_path}/depth/*.png"))
   
   # 2. 读取文本
   with open(f"{sequence_path}/nlp.txt", 'r') as f:
       text = f.read().strip()

   # 3. 【修正】读取第一帧的 BBox
   # 比赛要求会提供第一帧BBox。这里假设它在 groundtruth.txt 中
   # (与你的训练集 和旧代码 保持一致)
   # 如果测试集的文件名不同（例如 init.txt），请修改这里
   gt_path = f"{sequence_path}/groundtruth.txt"
   if not os.path.exists(gt_path):
       # 尝试备用名称
       gt_path = os.path.join(sequence_path, "init.txt") # 举例
       if not os.path.exists(gt_path):
           print(f"警告：在 {sequence_path} 中未找到 groundtruth.txt 或 init.txt。")
           # 尝试从 groundtruth_rect.txt 读取 (如果测试集和训练集格式一样)
           gt_path = f"{sequence_path}/groundtruth_rect.txt"
           if not os.path.exists(gt_path):
                raise FileNotFoundError(f"找不到 {sequence_path} 的初始BBox文件")

   with open(gt_path, 'r') as f:
       # 读取第一行作为初始BBox
       init_bbox = list(map(float, f.readline().strip().split(',')))
   
   model.eval()
   
   # 4. 【修正】结果列表必须包含第一帧的BBox
   results = [init_bbox]
   
   # 5. 【修正】将 prev_bbox 初始化为第一帧的BBox
   prev_bbox = init_bbox

   # 6. 【修正】模板图像（只在第一次循环时处理）
   template_rgb = None
   template_depth = None

   # 7. 【修正】循环从 0 开始，但第 0 帧只用于初始化模板
   for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
       
       rgb = preprocess_image(rgb_path) #
       depth = preprocess_depth(depth_path) #

       if i == 0:
           # 【修正】使用第一帧的BBox裁剪模板
           template_rgb = crop_template(rgb, init_bbox)
           template_depth = crop_template(depth, init_bbox)
       else:
           # 从第二帧开始，进行跟踪
           
           # 【修正】使用上一帧的 prev_bbox 裁剪搜索区域
           search_rgb = crop_search_region(rgb, prev_bbox)
           search_depth = crop_search_region(depth, prev_bbox)

           with torch.no_grad():
               pred_bbox, _ = model(
                   template_rgb, template_depth, [text],
                   search_rgb, search_depth
               ) [cite: 9]

           # [cite_start]【修正】坐标转换
           bbox = transform_bbox(pred_bbox, prev_bbox)
           prev_bbox = bbox
           results.append(bbox) #

   return results

# ---------------------------------------------------
# 你的 if __name__ == '__main__': 部分不需要大改
# ---------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)

    # 加载模型
    model = RGBDTextTracker().to(device)
    model.load_state_dict(torch.load('best.pth', map_location=device))

    # 【修正】指定到你的 aic25 根目录
    test_root = '/data/depth/aic25'
    
    # 【修正】修改此行以过滤文件夹
    # 我们只选择那些是目录、且文件夹名称是纯数字（例如 '001', '050'）的
    seqs = [
        d for d in os.listdir(test_root) 
        if os.path.isdir(os.path.join(test_root, d)) and d.isdigit()
    ]
    
    # 确保它们按顺序执行（'001', '002'...）
    seqs.sort() 

    print(f"在 {test_root} 中找到了 {len(seqs)} 个测试序列。")
    if 'train' in seqs or 'val' in seqs:
        print("警告：'train' 或 'val' 文件夹被错误地包含，请检查代码。")
    
    for s in seqs:
        seq_path = os.path.join(test_root, s)
        print(f'Processing {s} ...')
        
        # (确保你已经使用了我在上一步回复中提供的、修正后的 test_sequence 函数)
        res = test_sequence(model, seq_path)
        
        # 结果保存
        np.savetxt(f'results/{s}.txt', res, fmt='%.2f %.2f %.2f %.2f')

    print('All results saved to results/')