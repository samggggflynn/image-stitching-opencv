#!/usr/bin/env python3
"""
竖向图像拼接最终版
- 优先使用特征匹配进行精确拼接
- 如果匹配效果不佳，自动回退到简单垂直拼接
- 支持手动指定重叠区域
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_vertical_stitch(img_top, img_bottom, overlap_ratio=0.2):
    """
    简单垂直拼接（当特征匹配失败时使用）
    
    Args:
        img_top: 上面的图像
        img_bottom: 下面的图像
        overlap_ratio: 预期重叠比例
    
    Returns:
        拼接后的图像
    """
    h1, w1 = img_top.shape[:2]
    h2, w2 = img_bottom.shape[:2]
    
    # 对齐宽度
    if w1 != w2:
        logger.warning(f"宽度不一致，调整: {w1} -> {w2}")
        img_top = cv2.resize(img_top, (w2, int(h1 * w2 / w1)))
        h1, w1 = img_top.shape[:2]
    
    # 计算重叠区域
    overlap_h = int(min(h1, h2) * overlap_ratio)
    
    # 创建输出图像
    output_h = h1 + h2 - overlap_h
    result = np.zeros((output_h, w1, 3), dtype=np.uint8)
    
    # 放置上图
    result[:h1, :w1] = img_top
    
    # 放置下图（考虑重叠融合）
    result[h1 - overlap_h:h1 - overlap_h + h2, :w1] = img_bottom
    
    # 融合重叠区域
    if overlap_h > 0:
        start_y = h1 - overlap_h
        for y in range(overlap_h):
            alpha = y / overlap_h
            result[start_y + y, :w1] = (
                img_top[h1 - overlap_h + y, :w1] * (1 - alpha) +
                img_bottom[y, :w1] * alpha
            ).astype(np.uint8)
    
    logger.info(f"简单垂直拼接完成: {w1} x {output_h}")
    return result


def feature_based_stitch(img1, img2):
    """
    基于特征匹配的拼接
    
    Returns:
        (success, result)
    """
    try:
        # SIFT 特征检测
        sift = cv2.SIFT_create(nfeatures=5000)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        logger.info(f"SIFT 特征点: {len(kp1)} vs {len(kp2)}")
        
        if len(kp1) < 20 or len(kp2) < 20:
            logger.warning("特征点太少，使用简单拼接")
            return False, None
        
        # FLANN 匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        logger.info(f"良好匹配点: {len(good_matches)}")
        
        if len(good_matches) < 10:
            logger.warning("匹配点不足，使用简单拼接")
            return False, None
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if H is None:
            logger.warning("无法计算单应性矩阵，使用简单拼接")
            return False, None
        
        inliers = mask.sum()
        logger.info(f"有效匹配点(RANSAC): {inliers}")
        
        if inliers < 8:
            logger.warning("内点太少，使用简单拼接")
            return False, None
        
        # 拼接图像
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后边界
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        
        min_x, min_y = np.int32(transformed.min(axis=0).ravel())
        max_x, max_y = np.int32(transformed.max(axis=0).ravel())
        
        # 检查变换是否合理（避免异常大的输出）
        output_w = max(max_x, w2) - min(min_x, 0)
        output_h = max(max_y, h2) - min(min_y, 0)
        
        if output_w > w1 * 3 or output_h > h1 * 3:
            logger.warning(f"输出尺寸异常({output_w}x{output_h})，使用简单拼接")
            return False, None
        
        # 创建平移矩阵
        translation = np.array([[1, 0, -min(min_x, 0)], 
                                [0, 1, -min(min_y, 0)], 
                                [0, 0, 1]], dtype=np.float64)
        H_adjusted = translation @ H
        
        # 变换并拼接
        warped = cv2.warpPerspective(img1, H_adjusted, (output_w, output_h))
        result = warped.copy()
        
        # 放置第二张图
        y_offset = -min(min_y, 0)
        x_offset = -min(min_x, 0)
        result[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2
        
        # 融合重叠区域
        overlap_mask = (warped > 0) & (result > 0)
        if overlap_mask.any():
            y_coords = np.indices(result.shape[:2])[0]
            min_y = np.where(overlap_mask)[0].min()
            max_y = np.where(overlap_mask)[0].max()
            
            weight = np.zeros(result.shape[:2], dtype=np.float32)
            weight[min_y:max_y+1, :] = np.linspace(0.0, 1.0, max_y - min_y + 1).reshape(-1, 1)
            
            for c in range(3):
                result[..., c] = (warped[..., c] * (1 - weight) + 
                                result[..., c] * weight).astype(np.uint8)
        
        logger.info(f"特征匹配拼接完成: {output_w} x {output_h}")
        return True, result
    
    except Exception as e:
        logger.error(f"特征匹配拼接失败: {e}")
        return False, None


def stitch_vertical(img_top, img_bottom, mode='auto'):
    """
    竖向图像拼接主函数
    
    Args:
        img_top: 上面的图像
        img_bottom: 下面的图像
        mode: 'auto'自动选择, 'feature'强制特征匹配, 'simple'强制简单拼接
    
    Returns:
        拼接后的图像
    """
    if mode == 'simple':
        return simple_vertical_stitch(img_top, img_bottom)
    
    if mode == 'feature':
        success, result = feature_based_stitch(img_top, img_bottom)
        return result if success else simple_vertical_stitch(img_top, img_bottom)
    
    # auto 模式：先尝试特征匹配，失败则回退
    success, result = feature_based_stitch(img_top, img_bottom)
    if success:
        return result
    
    logger.info("回退到简单垂直拼接")
    return simple_vertical_stitch(img_top, img_bottom)


def main():
    import sys
    
    if len(sys.argv) < 4:
        print("用法: python stitch_vertical_final.py <上图> <下图> <输出路径> [mode]")
        print("mode: auto(默认), feature(强制特征匹配), simple(强制简单拼接)")
        sys.exit(1)
    
    img_top_path = Path(sys.argv[1])
    img_bottom_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else 'auto'
    
    # 加载图像
    img_top = cv2.imread(str(img_top_path))
    img_bottom = cv2.imread(str(img_bottom_path))
    
    if img_top is None or img_bottom is None:
        logger.error("无法读取图像")
        sys.exit(1)
    
    logger.info(f"输入图像: {img_top_path.name} ({img_top.shape}) + {img_bottom_path.name} ({img_bottom.shape})")
    
    # 执行拼接
    result = stitch_vertical(img_top, img_bottom, mode)
    
    # 保存结果
    cv2.imwrite(str(output_path), result)
    logger.info(f"拼接完成！结果已保存到: {output_path}")
    logger.info(f"输出尺寸: {result.shape[1]} x {result.shape[0]}")


if __name__ == "__main__":
    main()
