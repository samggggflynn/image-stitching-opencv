#!/usr/bin/env python3
"""
专门用于竖向图像拼接的脚本
针对重叠区域小、特征点少的情况优化
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stitch_vertical_images(img1_path: Path, img2_path: Path, output_path: Path):
    """
    使用 ORB 特征 + 单应性矩阵手动拼接竖向图像
    
    相比 OpenCV Stitcher，这个方法：
    1. 使用 ORB 代替 SIFT（对低纹理图像更鲁棒）
    2. 降低 RANSAC 阈值
    3. 不做复杂的曝光补偿和波浪校正
    """
    # 加载图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        logger.error("无法读取图像")
        return False
    
    logger.info(f"图像 1: {img1.shape}")
    logger.info(f"图像 2: {img2.shape}")
    
    # 转灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # ORB 特征检测（比 SIFT 更宽松）
    orb = cv2.ORB_create(
        nfeatures=10000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31
    )
    
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    logger.info(f"特征点：{len(kp1)} vs {len(kp2)}")
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test（更宽松）
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
    
    logger.info(f"良好匹配：{len(good_matches)}")
    
    if len(good_matches) < 4:
        logger.error("匹配点不足 4 个，无法计算单应性矩阵")
        return False
    
    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 计算单应性矩阵（降低 RANSAC 阈值）
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        logger.error("无法计算单应性矩阵")
        return False
    
    inliers = mask.sum()
    logger.info(f"有效匹配点：{inliers}")
    
    # 根据图像尺寸计算输出画布大小
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 计算变换后的角点
    corners = np.array([
        [0, 0],
        [w1, 0],
        [w1, h1],
        [0, h1]
    ], dtype=np.float32)
    
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
    
    # 计算边界
    min_x = min(transformed_corners.min(), 0)
    max_x = max(transformed_corners.max(), w2)
    min_y = min(transformed_corners.min(), 0)
    max_y = max(transformed_corners.max(), h2)
    
    # 创建平移矩阵
    translation = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    
    H_adjusted = translation @ H
    
    # 计算输出尺寸
    output_w = int(max_x - min_x)
    output_h = int(max_y - min_y)
    
    logger.info(f"输出尺寸：{output_w} x {output_h}")
    
    # 拼接
    result = cv2.warpPerspective(img1, H_adjusted, (output_w, output_h))
    result[0:h2, 0:w2] = cv2.addWeighted(
        result[0:h2, 0:w2], 0.5,
        img2, 0.5, 0
    )
    
    # 保存
    cv2.imwrite(str(output_path), result)
    logger.info(f"拼接完成：{output_path}")
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 4:
        print("用法：python stitch_vertical.py <图像 1> <图像 2> <输出路径>")
        sys.exit(1)
    
    img1 = Path(sys.argv[1])
    img2 = Path(sys.argv[2])
    output = Path(sys.argv[3])
    
    success = stitch_vertical_images(img1, img2, output)
    sys.exit(0 if success else 1)
