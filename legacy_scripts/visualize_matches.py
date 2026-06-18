#!/usr/bin/env python3
"""可视化特征点匹配结果"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_matches(img1_path: Path, img2_path: Path, output_path: Path = None):
    """
    可视化两张图像之间的特征点匹配
    
    Args:
        img1_path: 第一张图像路径
        img2_path: 第二张图像路径
        output_path: 输出图像路径（可选）
    """
    # 加载图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        logger.error("无法读取图像")
        return
    
    logger.info(f"图像 1: {img1.shape}")
    logger.info(f"图像 2: {img2.shape}")
    
    # 转灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # ORB 特征检测
    orb = cv2.ORB_create(nfeatures=10000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    logger.info(f"特征点数量: 图像1={len(kp1)}, 图像2={len(kp2)}")
    
    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
    
    logger.info(f"良好匹配点: {len(good_matches)}")
    
    # 可视化匹配结果
    # 随机选择最多 50 个匹配点显示（避免连线过多）
    display_matches = min(50, len(good_matches))
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2,
        good_matches[:display_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),       # 匹配线为绿色
        singlePointColor=(255, 0, 0)   # 特征点为红色
    )
    
    # 计算单应性矩阵用于验证
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if H is not None:
            inliers = mask.sum()
            logger.info(f"有效匹配点(RANSAC): {inliers}")
            
            # 在图像上标记内点和外点
            for i, match in enumerate(good_matches[:display_matches]):
                if mask[i]:
                    # 内点用绿色
                    color = (0, 255, 0)
                else:
                    # 外点用红色
                    color = (0, 0, 255)
                
                # 在关键点上画圆
                pt1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
                pt2 = (int(img1.shape[1] + kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1]))
                
                cv2.circle(matched_img, pt1, 5, color, -1)
                cv2.circle(matched_img, pt2, 5, color, -1)
    
    # 添加图例
    h, w = matched_img.shape[:2]
    legend = np.zeros((60, w, 3), dtype=np.uint8)
    cv2.putText(legend, f"总匹配点: {len(good_matches)} | 显示: {display_matches}", 
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(legend, "绿色: 内点(RANSAC) | 红色: 外点", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 拼接图例
    result = np.vstack([legend, matched_img])
    
    # 保存结果
    if output_path:
        cv2.imwrite(str(output_path), result)
        logger.info(f"匹配可视化已保存到: {output_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python visualize_matches.py <图像1> <图像2> [输出路径]")
        sys.exit(1)
    
    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    
    visualize_matches(img1_path, img2_path, output_path)
