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
from lib.dataset import preprocess_image, preprocess_depth, crop_template, crop_search_region, transform_bbox

def test_sequence(model, sequence_path):
    """测试单个序列（适配官方结构）"""
    # 读取帧列表
    rgb_list = sorted(glob.glob(f"{sequence_path}/color/*.jpg"))
    depth_list = sorted(glob.glob(f"{sequence_path}/depth/*.png"))
    
    # 文本文件
    with open(f"{sequence_path}/nlp.txt", 'r') as f:
        text = f.read().strip()
    
    model.eval()
    results = []

    prev_bbox = [0, 0, 0, 0]  # 初始化假bbox，模型可预测相对偏移
    
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
        rgb = preprocess_image(rgb_path)
        depth = preprocess_depth(depth_path)

        if i == 0:
            # 初始化模板
            template_rgb = rgb
            template_depth = depth
        else:
            search_rgb = crop_search_region(rgb, prev_bbox)
            search_depth = crop_search_region(depth, prev_bbox)

            with torch.no_grad():
                pred_bbox, _ = model(template_rgb, template_depth, [text], search_rgb, search_depth)

            bbox = transform_bbox(pred_bbox, prev_bbox)
            prev_bbox = bbox
            results.append(bbox)

    return results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)

    # 加载模型
    model = RGBDTextTracker().to(device)
    model.load_state_dict(torch.load('best.pth', map_location=device))

    # 改成官方初赛测试集路径
    test_root = '/data/depth/初赛数据集-多源异构数据协同的视频目标跟踪挑战赛/2-初赛测试集/TestSet'
    seqs = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]

    for s in seqs:
        seq_path = os.path.join(test_root, s)
        print(f'Processing {s} ...')
        res = test_sequence(model, seq_path)
        np.savetxt(f'results/{s}.txt', res, fmt='%.2f %.2f %.2f %.2f')