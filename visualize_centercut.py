#!/usr/bin/env python3
"""
可视化中心切割拼接算法的每个步骤
"""

import cv2
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_centercut_steps(img_top, img_bottom, overlap_ratio=0.2, shift_y=None, blend_width=20):
    """
    可视化中心切割拼接的每个步骤
    
    Args:
        img_top: 上图
        img_bottom: 下图
        overlap_ratio: 重叠比例
        shift_y: 垂直偏移量
        blend_width: 渐变融合宽度
    """
    # 对齐宽度
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bottom.shape[:2]
    
    if w1 != w2:
        target_w = min(w1, w2)
        img_top = cv2.resize(img_top, (target_w, int(h1 * target_w / w1)))
        img_bottom = cv2.resize(img_bottom, (target_w, int(h2 * target_w / w2)))
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
    
    # 步骤1：计算重叠区域
    overlap_h = int(min(h1, h2) * overlap_ratio)
    
    # 如果指定了偏移量，调整重叠
    if shift_y is not None:
        adjusted_overlap = int(overlap_h + shift_y)
        overlap_h = max(int(min(h1, h2) * 0.1), min(adjusted_overlap, int(min(h1, h2) * 0.5)))
    
    total_h = h1 + h2 - overlap_h
    
    # 步骤2：计算关键位置
    overlap_start_top = h1 - overlap_h  # 重叠区域在上图中的起始位置
    cut_position = h1 - overlap_h // 2  # 中心切割位置（在输出图像中的位置）
    
    # 步骤3：先创建拼接结果
    result = np.zeros((total_h, w1, 3), dtype=np.uint8)
    
    # 放置上图（完整放置）
    result[:h1, :w1] = img_top
    
    # 放置下图（从切割位置开始）
    bottom_end = min(cut_position + h2, total_h)
    result[cut_position:bottom_end, :w1] = img_bottom[:bottom_end - cut_position, :]
    
    # 渐变融合
    blend_start = max(overlap_start_top, cut_position - blend_width)
    blend_end = min(cut_position + overlap_h // 2, cut_position + blend_width)
    
    if blend_start < blend_end:
        for y in range(blend_start, blend_end):
            # 计算权重：切割线上方上图权重高，下方下图权重高
            y_in_blend = y - (cut_position - blend_width)
            if y_in_blend < blend_width:
                # 切割线上方：上图权重从1逐渐降低
                alpha = y_in_blend / (2 * blend_width)
            else:
                # 切割线下方：下图权重逐渐增加
                alpha = 1.0 - (2 * blend_width - y_in_blend) / (2 * blend_width)
            
            y_top = y
            y_bottom = y - cut_position
            
            if 0 <= y_top < h1 and 0 <= y_bottom < h2:
                result[y, :] = (
                    img_top[y_top, :] * (1 - alpha) +
                    img_bottom[y_bottom, :] * alpha
                ).astype(np.uint8)
    
    # 步骤4：创建可视化图像（基于计算结果确定尺寸）
    info_lines_count = 14
    info_height = info_lines_count * 18 + 20
    
    canvas_h = max(h1, h2) + total_h + 100 + info_height
    canvas_w = w1 * 2 + 250
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # 绘制原图
    y_offset = 20
    x_offset = 20
    
    # Top image
    canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = img_top
    cv2.putText(canvas, 'Top Image', (x_offset, y_offset-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(canvas, f'Size: {w1}x{h1}', (x_offset, y_offset+h1+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Bottom image
    canvas[y_offset:y_offset+h2, x_offset+w1+50:x_offset+w1+50+w1] = img_bottom
    cv2.putText(canvas, 'Bottom Image', (x_offset+w1+50, y_offset-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f'Size: {w1}x{h2}', (x_offset+w1+50, y_offset+h2+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Stitched result
    result_y = y_offset + max(h1, h2) + 60
    canvas[result_y:result_y+total_h, x_offset:x_offset+w1] = result
    cv2.putText(canvas, 'Stitched Result', (x_offset, result_y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(canvas, f'Size: {w1}x{total_h}', (x_offset, result_y+total_h+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Draw info panel
    info_y = result_y + total_h + 40
    info_lines = [
        f"Step 1: Calculate Overlap",
        f"  - Ratio: {overlap_ratio*100:.0f}%",
        f"  - Height: {overlap_h} px",
        "",
        f"Step 2: Center Cut Position",
        f"  - Overlap Start (in Top): Y={overlap_start_top}",
        f"  - Center Cut Line: Y={cut_position}",
        f"  - Overlap End: Y={cut_position + overlap_h // 2}",
        "",
        f"Step 3: Blend",
        f"  - Blend Width: {blend_width} px",
        f"  - Blend Area: Y={blend_start} to Y={blend_end}",
        "",
        f"Step 4: Output Size",
        f"  - Width: {w1} px",
        f"  - Height: {total_h} px"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (x_offset, info_y + i * 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    
    # Draw key position lines on result
    # Overlap start line (blue)
    cv2.line(canvas, (x_offset, result_y + overlap_start_top), 
             (x_offset + w1, result_y + overlap_start_top), (255, 0, 0), 1)
    cv2.putText(canvas, 'Overlap Start', (x_offset + w1 + 5, result_y + overlap_start_top - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    
    # Center cut line (red)
    cv2.line(canvas, (x_offset, result_y + cut_position), 
             (x_offset + w1, result_y + cut_position), (0, 0, 255), 2)
    cv2.putText(canvas, 'Center Cut', (x_offset + w1 + 5, result_y + cut_position - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # Overlap end line (blue)
    cv2.line(canvas, (x_offset, result_y + cut_position + overlap_h // 2), 
             (x_offset + w1, result_y + cut_position + overlap_h // 2), (255, 0, 0), 1)
    cv2.putText(canvas, 'Overlap End', (x_offset + w1 + 5, result_y + cut_position + overlap_h // 2 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    
    # Blend area (green box)
    cv2.rectangle(canvas, 
                  (x_offset, result_y + blend_start), 
                  (x_offset + w1, result_y + blend_end), 
                  (0, 255, 0), 1)
    cv2.putText(canvas, 'Blend Area', (x_offset + w1 + 5, result_y + blend_start + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize center cut stitching steps")
    parser.add_argument("--top", "-t", type=str, required=True, help="Top image path")
    parser.add_argument("--bottom", "-b", type=str, required=True, help="Bottom image path")
    parser.add_argument("--output", "-o", type=str, default="centercut_visualization.png", help="Output image path")
    parser.add_argument("--overlap", "-r", type=float, default=0.2, help="Overlap ratio (0.1-0.5)")
    parser.add_argument("--shift", "-s", type=float, default=None, help="Vertical shift")
    parser.add_argument("--blend-width", type=int, default=20, help="Blend width")
    
    args = parser.parse_args()
    
    # Load images
    img_top = cv2.imread(args.top)
    img_bottom = cv2.imread(args.bottom)
    
    if img_top is None or img_bottom is None:
        logger.error("Failed to read images")
        return
    
    logger.info(f"Input images: top={img_top.shape}, bottom={img_bottom.shape}")
    
    # Generate visualization
    canvas = visualize_centercut_steps(
        img_top, img_bottom,
        overlap_ratio=args.overlap,
        shift_y=args.shift,
        blend_width=args.blend_width
    )
    
    # Save result
    cv2.imwrite(args.output, canvas)
    logger.info(f"Visualization saved to: {args.output}")
    logger.info(f"Output size: {canvas.shape[1]} x {canvas.shape[0]}")


if __name__ == "__main__":
    main()
