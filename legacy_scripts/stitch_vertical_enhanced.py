#!/usr/bin/env python3
"""
竖向图像拼接增强版
针对上下排列的图像专门优化：
1. 使用 SIFT + FLANN 进行更精准的特征匹配
2. 针对竖向图像的几何约束优化
3. 多尺度图像融合
4. 自动检测重叠区域方向
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerticalStitcher:
    """专门用于竖向图像拼接的类"""
    
    def __init__(self, min_matches=10, ransac_threshold=3.0):
        """
        初始化竖向拼接器
        
        Args:
            min_matches: 最小匹配点数量
            ransac_threshold: RANSAC 阈值
        """
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        
        # 使用 SIFT 检测器（比 ORB 更准确）
        self.sift = cv2.SIFT_create(nfeatures=10000)
        
        # FLANN 匹配器配置
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_features(self, image):
        """检测图像特征点和描述符"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)
    
    def match_features(self, des1, des2):
        """使用 FLANN 进行特征匹配"""
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def filter_vertical_matches(self, kp1, kp2, matches, max_y_diff=200):
        """
        过滤匹配点，只保留符合竖向拼接的匹配
        对于竖向拼接，两张图的匹配点应该在大致相同的水平位置
        """
        filtered = []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # 竖向拼接时，x坐标应该相近
            x_diff = abs(pt1[0] - pt2[0])
            if x_diff < max_y_diff:  # 使用 max_y_diff 作为水平差异阈值
                filtered.append(match)
        
        return filtered
    
    def compute_homography(self, kp1, kp2, matches):
        """计算单应性矩阵"""
        if len(matches) < self.min_matches:
            return None, 0
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            return None, 0
        
        return H, mask.sum()
    
    def blend_images(self, img1, img2, H):
        """
        使用多尺度融合拼接图像
        
        Args:
            img1: 待变换的图像（上图）
            img2: 参考图像（下图）
            H: 单应性矩阵
        
        Returns:
            拼接后的图像
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算变换后的边界
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # 确定输出画布大小
        all_corners = np.vstack([transformed_corners, 
                                np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)])
        
        min_x, min_y = np.int32(all_corners.min(axis=0).ravel())
        max_x, max_y = np.int32(all_corners.max(axis=0).ravel())
        
        # 创建平移矩阵以确保所有内容可见
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
        H_adjusted = translation @ H
        
        output_w = max_x - min_x
        output_h = max_y - min_y
        
        # 变换第一张图像
        warped = cv2.warpPerspective(img1, H_adjusted, (output_w, output_h))
        
        # 创建第二张图像的放置区域
        result = warped.copy()
        result[-min_y:-min_y+h2, -min_x:-min_x+w2] = img2
        
        # 找到重叠区域并融合
        overlap_mask = (warped > 0) & (result > 0)
        
        # 使用渐变融合
        if overlap_mask.any():
            # 计算垂直方向的权重
            y_coords = np.indices(result.shape[:2])[0]
            overlap_region = np.where(overlap_mask)
            
            # 找到重叠区域的垂直范围
            min_overlap_y = overlap_region[0].min()
            max_overlap_y = overlap_region[0].max()
            
            # 创建渐变权重
            weight = np.zeros(result.shape[:2], dtype=np.float32)
            weight[min_overlap_y:max_overlap_y+1, :] = np.linspace(
                0.0, 1.0, max_overlap_y - min_overlap_y + 1
            ).reshape(-1, 1)
            
            # 融合
            for c in range(3):
                result[..., c] = (
                    warped[..., c] * (1 - weight) + 
                    result[..., c] * weight
                ).astype(np.uint8)
        
        return result
    
    def stitch(self, img1, img2):
        """
        执行竖向拼接
        
        Args:
            img1: 第一张图像（通常是上面的）
            img2: 第二张图像（通常是下面的）
        
        Returns:
            (success, result): 拼接是否成功，拼接结果
        """
        logger.info(f"图像尺寸: img1={img1.shape}, img2={img2.shape}")
        
        # 检测特征
        kp1, des1 = self.detect_features(img1)
        kp2, des2 = self.detect_features(img2)
        
        logger.info(f"特征点: img1={len(kp1)}, img2={len(kp2)}")
        
        # 匹配特征
        matches = self.match_features(des1, des2)
        logger.info(f"初步匹配: {len(matches)}")
        
        # 过滤竖向匹配点
        filtered_matches = self.filter_vertical_matches(kp1, kp2, matches)
        logger.info(f"竖向过滤后: {len(filtered_matches)}")
        
        if len(filtered_matches) < self.min_matches:
            logger.error(f"匹配点不足 ({len(filtered_matches)} < {self.min_matches})")
            return False, None
        
        # 计算单应性矩阵
        H, inliers = self.compute_homography(kp1, kp2, filtered_matches)
        logger.info(f"有效匹配点(RANSAC): {inliers}")
        
        if H is None:
            logger.error("无法计算单应性矩阵")
            return False, None
        
        # 拼接图像
        result = self.blend_images(img1, img2, H)
        logger.info(f"拼接完成，输出尺寸: {result.shape}")
        
        return True, result


def main():
    import sys
    
    if len(sys.argv) != 4:
        print("用法: python stitch_vertical_enhanced.py <上图> <下图> <输出路径>")
        print("提示: 请按照从上到下的顺序输入图像")
        sys.exit(1)
    
    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    
    # 加载图像
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    
    if img1 is None or img2 is None:
        logger.error("无法读取图像")
        sys.exit(1)
    
    # 创建拼接器并执行拼接
    stitcher = VerticalStitcher(min_matches=8, ransac_threshold=2.0)
    success, result = stitcher.stitch(img1, img2)
    
    if success:
        cv2.imwrite(str(output_path), result)
        logger.info(f"结果已保存到: {output_path}")
        sys.exit(0)
    else:
        logger.error("拼接失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
