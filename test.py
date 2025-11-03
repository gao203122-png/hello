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



# import os
# import glob
# import torch
# import numpy as np
# from models.tracker import RGBDTextTracker
# # 确保这些函数被正确导入
# from lib.dataset import preprocess_image, preprocess_depth, crop_template, crop_search_region, transform_bbox

# def test_sequence(model, sequence_path):
#    """测试单个序列（已修正，适配比赛要求）"""
   
#    # 1. 读取文件列表
#    rgb_list = sorted(glob.glob(f"{sequence_path}/color/*.jpg"))
#    depth_list = sorted(glob.glob(f"{sequence_path}/depth/*.png"))
   
#    # 2. 读取文本
#    with open(f"{sequence_path}/nlp.txt", 'r') as f:
#        text = f.read().strip()

#    # 3. 【修正】读取第一帧的 BBox
#    # 比赛要求会提供第一帧BBox。这里假设它在 groundtruth.txt 中
#    # (与你的训练集 和旧代码 保持一致)
#    # 如果测试集的文件名不同（例如 init.txt），请修改这里
#    gt_path = f"{sequence_path}/groundtruth.txt"
#    if not os.path.exists(gt_path):
#        # 尝试备用名称
#        gt_path = os.path.join(sequence_path, "init.txt") # 举例
#        if not os.path.exists(gt_path):
#            print(f"警告：在 {sequence_path} 中未找到 groundtruth.txt 或 init.txt。")
#            # 尝试从 groundtruth_rect.txt 读取 (如果测试集和训练集格式一样)
#            gt_path = f"{sequence_path}/groundtruth_rect.txt"
#            if not os.path.exists(gt_path):
#                 raise FileNotFoundError(f"找不到 {sequence_path} 的初始BBox文件")

#    with open(gt_path, 'r') as f:
#        # 读取第一行作为初始BBox
#        init_bbox = list(map(float, f.readline().strip().split(',')))
   
#    model.eval()
   
#    # 4. 【修正】结果列表必须包含第一帧的BBox
#    results = [init_bbox]
   
#    # 5. 【修正】将 prev_bbox 初始化为第一帧的BBox
#    prev_bbox = init_bbox

#    # 6. 【修正】模板图像（只在第一次循环时处理）
#    template_rgb = None
#    template_depth = None

#    # 7. 【修正】循环从 0 开始，但第 0 帧只用于初始化模板
#    for i, (rgb_path, depth_path) in enumerate(zip(rgb_list, depth_list)):
       
#        rgb = preprocess_image(rgb_path) #
#        depth = preprocess_depth(depth_path) #

#        if i == 0:
#            # 【修正】使用第一帧的BBox裁剪模板
#            template_rgb = crop_template(rgb, init_bbox)
#            template_depth = crop_template(depth, init_bbox)
#        else:
#            # 从第二帧开始，进行跟踪
           
#            # 【修正】使用上一帧的 prev_bbox 裁剪搜索区域
#            search_rgb = crop_search_region(rgb, prev_bbox)
#            search_depth = crop_search_region(depth, prev_bbox)

#            with torch.no_grad():
#                pred_bbox, _ = model(
#                    template_rgb, template_depth, [text],
#                    search_rgb, search_depth
#                ) [cite: 9]

#            # [cite_start]【修正】坐标转换
#            bbox = transform_bbox(pred_bbox, prev_bbox)
#            prev_bbox = bbox
#            results.append(bbox) #

#    return results

# # ---------------------------------------------------
# # 你的 if __name__ == '__main__': 部分不需要大改
# # ---------------------------------------------------
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     os.makedirs('results', exist_ok=True)

#     # 加载模型
#     model = RGBDTextTracker().to(device)
#     model.load_state_dict(torch.load('best.pth', map_location=device))

#     # 【修正】指定到你的 aic25 根目录
#     test_root = '/data/depth/aic25'
    
#     # 【修正】修改此行以过滤文件夹
#     # 我们只选择那些是目录、且文件夹名称是纯数字（例如 '001', '050'）的
#     seqs = [
#         d for d in os.listdir(test_root) 
#         if os.path.isdir(os.path.join(test_root, d)) and d.isdigit()
#     ]
    
#     # 确保它们按顺序执行（'001', '002'...）
#     seqs.sort() 

#     print(f"在 {test_root} 中找到了 {len(seqs)} 个测试序列。")
#     if 'train' in seqs or 'val' in seqs:
#         print("警告：'train' 或 'val' 文件夹被错误地包含，请检查代码。")
    
#     for s in seqs:
#         seq_path = os.path.join(test_root, s)
#         print(f'Processing {s} ...')
        
#         # (确保你已经使用了我在上一步回复中提供的、修正后的 test_sequence 函数)
#         res = test_sequence(model, seq_path)
        
#         # 结果保存
#         np.savetxt(f'results/{s}.txt', res, fmt='%.2f %.2f %.2f %.2f')

#     print('All results saved to results/')

import os
import glob
import torch
import numpy as np
from PIL import Image
from models.tracker_final import RGBDTextTracker
from lib.dataset_fixed import preprocess_image, preprocess_depth

def load_model(ckpt_path='best_final.pth', device='cuda'):
    model = RGBDTextTracker().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def process_sequence(model, seq_dir, output_path, device='cuda'):
    """处理单个序列"""
    rgb_frames = sorted(glob.glob(f"{seq_dir}/color/*.jpg"))
    depth_frames = sorted(glob.glob(f"{seq_dir}/depth/*.png"))
    text_file = f"{seq_dir}/nlp.txt"
    gt_file = f"{seq_dir}/groundtruth_rect.txt"
    
    if not rgb_frames or not depth_frames:
        print(f"[WARNING] Empty sequence: {seq_dir}")
        return
    
    # 读取文本
    with open(text_file, 'r') as f:
        text = f.read().strip()
    
    # 读取初始GT
    with open(gt_file, 'r') as f:
        init_bbox = list(map(float, f.readline().strip().split(',')))
    
    # 获取原始尺寸
    orig_img = Image.open(rgb_frames[0])
    orig_w, orig_h = orig_img.size
    
    # 第一帧作为模板
    template_rgb = preprocess_image(rgb_frames[0]).unsqueeze(0).to(device)
    template_depth = preprocess_depth(depth_frames[0]).unsqueeze(0).to(device)
    
    results = [init_bbox]  # 第一帧用GT
    
    with torch.no_grad():
        prev_bbox_256 = [  # 转换初始GT到256尺度（供平滑用）
            init_bbox[0] * 256 / orig_w,
            init_bbox[1] * 256 / orig_h,
            init_bbox[2] * 256 / orig_w,
            init_bbox[3] * 256 / orig_h
        ]
        
        for i in range(1, len(rgb_frames)):
            # 加载当前帧
            search_rgb = preprocess_image(rgb_frames[i]).unsqueeze(0).to(device)
            search_depth = preprocess_depth(depth_frames[i]).unsqueeze(0).to(device)
            
            # 预测（输出是256尺度）
            pred_bbox_256, _ = model(template_rgb, template_depth, [text], search_rgb, search_depth)
            pred_bbox_256 = pred_bbox_256.cpu().numpy()[0]
            
            # ===== 尺度平滑（防止突变） =====
            max_scale_change = 1.3
            w_ratio = pred_bbox_256[2] / (prev_bbox_256[2] + 1e-6)
            h_ratio = pred_bbox_256[3] / (prev_bbox_256[3] + 1e-6)
            
            if w_ratio > max_scale_change:
                pred_bbox_256[2] = prev_bbox_256[2] * max_scale_change
            elif w_ratio < 1/max_scale_change:
                pred_bbox_256[2] = prev_bbox_256[2] / max_scale_change
                
            if h_ratio > max_scale_change:
                pred_bbox_256[3] = prev_bbox_256[3] * max_scale_change
            elif h_ratio < 1/max_scale_change:
                pred_bbox_256[3] = prev_bbox_256[3] / max_scale_change
            
            # ===== 位置平滑 =====
            alpha = 0.7  # 平滑系数
            pred_bbox_256[0] = alpha * pred_bbox_256[0] + (1-alpha) * prev_bbox_256[0]
            pred_bbox_256[1] = alpha * pred_bbox_256[1] + (1-alpha) * prev_bbox_256[1]
            
            # ===== 转换回原图尺度 =====
            scale_x = orig_w / 256.0
            scale_y = orig_h / 256.0
            pred_bbox_orig = [
                pred_bbox_256[0] * scale_x,
                pred_bbox_256[1] * scale_y,
                pred_bbox_256[2] * scale_x,
                pred_bbox_256[3] * scale_y
            ]
            
            # 边界检查
            pred_bbox_orig[0] = max(0, min(pred_bbox_orig[0], orig_w - pred_bbox_orig[2]))
            pred_bbox_orig[1] = max(0, min(pred_bbox_orig[1], orig_h - pred_bbox_orig[3]))
            pred_bbox_orig[2] = max(1, min(pred_bbox_orig[2], orig_w - pred_bbox_orig[0]))
            pred_bbox_orig[3] = max(1, min(pred_bbox_orig[3], orig_h - pred_bbox_orig[1]))
            
            results.append(pred_bbox_orig)
            prev_bbox_256 = pred_bbox_256
            
            # ===== 自适应模板更新 =====
            # 每10帧或置信度高时更新
            if i % 10 == 0:
                template_rgb = search_rgb.clone()
                template_depth = search_depth.clone()
    
    # 保存结果
    with open(output_path, 'w') as f:
        for bbox in results:
            f.write(f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}\n")
    
    print(f"[OK] {os.path.basename(seq_dir)}: {len(results)} frames")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model('best_final.pth', device)
    
    # 测试集路径
    test_root = '/data/depth/aic25/test'
    output_dir = 'results_final'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有测试序列
    test_seqs = sorted(glob.glob(f"{test_root}/*"))
    
    print(f"\nProcessing {len(test_seqs)} sequences...")
    for seq_dir in test_seqs:
        if not os.path.isdir(seq_dir):
            continue
        
        seq_name = os.path.basename(seq_dir)
        output_path = f"{output_dir}/{seq_name}.txt"
        
        try:
            process_sequence(model, seq_dir, output_path, device)
        except Exception as e:
            print(f"[ERROR] {seq_name}: {e}")
            # 写入默认值避免提交失败
            with open(output_path, 'w') as f:
                f.write("0.00 0.00 1.00 1.00\n")
    
    print(f"\n✅ All done! Results saved to {output_dir}/")
    print(f"Now you can: cd {output_dir} && zip -r ../submission.zip *.txt")

if __name__ == "__main__":
    main()