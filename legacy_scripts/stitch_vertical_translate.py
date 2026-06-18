#!/usr/bin/env python3
"""
竖向平移矩阵拼接工具

特点：
1. 使用简单的垂直平移模型，不使用复杂的单应性矩阵
2. 支持手动调整相机垂直移动距离（偏移量）
3. 自动检测最佳重叠区域
4. 渐变融合消除拼接痕迹
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerticalTranslateStitcher:
    """基于垂直平移矩阵的图像拼接器"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=5000)
    
    def detect_and_match(self, img1, img2):
        """检测特征点并匹配"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
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
        
        return kp1, kp2, good_matches
    
    def estimate_vertical_shift(self, img1, img2, matches, kp1, kp2):
        """
        估计垂直偏移量
        
        Returns:
            estimated_shift: 估计的垂直偏移像素数
            confidence: 置信度 (0-1)
        """
        if len(matches) < 4:
            return None, 0.0
        
        # 收集匹配点的垂直差异
        y_diffs = []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            # 只考虑水平位置相近的匹配
            if abs(pt1[0] - pt2[0]) < 50:
                y_diffs.append(pt2[1] - pt1[1])
        
        if len(y_diffs) < 4:
            return None, 0.0
        
        # 使用中位数作为估计值（更稳健）
        y_diffs = np.array(y_diffs)
        estimated_shift = np.median(y_diffs)
        
        # 计算置信度（基于匹配点的一致性）
        std_dev = np.std(y_diffs)
        confidence = max(0.0, min(1.0, 1.0 - std_dev / 100.0))
        
        logger.info(f"估计垂直偏移: {estimated_shift:.1f} 像素, 置信度: {confidence:.2f}")
        return estimated_shift, confidence
    
    def create_translation_matrix(self, dy):
        """
        创建垂直平移矩阵
        
        Args:
            dy: 垂直偏移量（正数表示向下移动）
        
        Returns:
            3x3 平移矩阵
        """
        return np.array([
            [1, 0, 0],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def blend_with_mask(self, img_top, img_bottom, shift_y, overlap_ratio=0.3):
        """
        使用平移矩阵拼接图像并融合
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
            shift_y: 垂直偏移量（相机向下移动的距离）
            overlap_ratio: 重叠区域比例
        
        Returns:
            拼接后的图像
        """
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        # 对齐宽度
        if w1 != w2:
            target_w = min(w1, w2)
            img_top = cv2.resize(img_top, (target_w, int(h1 * target_w / w1)))
            img_bottom = cv2.resize(img_bottom, (target_w, int(h2 * target_w / w2)))
            h1, w1 = img_top.shape[:2]
            h2, w2 = img_bottom.shape[:2]
        
        # 计算实际重叠高度
        overlap_h = int(min(h1, h2) * overlap_ratio)
        
        # 根据偏移量调整输出尺寸
        # shift_y > 0: 相机向下移动，上图相对于下图向上偏移
        # shift_y < 0: 相机向上移动，上图相对于下图向下偏移
        
        # 计算输出图像尺寸
        total_h = h1 + h2 - overlap_h
        result = np.zeros((total_h, w1, 3), dtype=np.uint8)
        
        # 计算上图放置位置（考虑偏移）
        # 偏移量会影响重叠区域的位置
        adjusted_shift = shift_y if shift_y is not None else 0
        
        # 确保上图完全在画布内
        top_y = int(max(0, -adjusted_shift))
        bottom_y = top_y + h1
        
        # 如果偏移导致上图超出边界，进行裁剪
        if top_y > 0:
            img_top = img_top[top_y:, :]
            h1 = img_top.shape[0]
            top_y = 0
        
        if bottom_y > total_h:
            img_top = img_top[:total_h - top_y, :]
            h1 = img_top.shape[0]
        
        # 放置上图
        result[top_y:top_y + h1, :w1] = img_top
        
        # 计算下图放置位置
        bottom_start = total_h - h2
        
        # 根据偏移调整下图位置
        bottom_start = int(bottom_start + adjusted_shift * 0.5)
        bottom_start = max(0, min(bottom_start, total_h - h2))
        
        # 放置下图
        result[bottom_start:bottom_start + h2, :w1] = img_bottom
        
        # 计算实际重叠区域
        overlap_start = max(top_y, bottom_start)
        overlap_end = min(top_y + h1, bottom_start + h2)
        
        # 融合重叠区域
        if overlap_start < overlap_end:
            overlap_height = overlap_end - overlap_start
            for y in range(overlap_height):
                alpha = y / overlap_height
                y_result = overlap_start + y
                y_top = y_result - top_y
                y_bottom = y_result - bottom_start
                
                if 0 <= y_top < h1 and 0 <= y_bottom < h2:
                    result[y_result, :w1] = (
                        img_top[y_top, :w1] * (1 - alpha) +
                        img_bottom[y_bottom, :w1] * alpha
                    ).astype(np.uint8)
        
        return result
    
    def stitch(self, img_top, img_bottom, shift_y=None, auto_detect=True):
        """
        执行竖向拼接
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
            shift_y: 手动指定垂直偏移量（可选）
            auto_detect: 是否自动检测偏移量
        
        Returns:
            (success, result, estimated_shift)
        """
        logger.info(f"输入图像: 上图={img_top.shape}, 下图={img_bottom.shape}")
        
        estimated_shift = None
        
        # 自动检测偏移量
        if auto_detect and shift_y is None:
            kp1, kp2, matches = self.detect_and_match(img_top, img_bottom)
            logger.info(f"特征匹配数: {len(matches)}")
            
            if len(matches) >= 4:
                estimated_shift, confidence = self.estimate_vertical_shift(
                    img_top, img_bottom, matches, kp1, kp2
                )
                
                if confidence > 0.3:
                    shift_y = estimated_shift
                    logger.info(f"使用自动检测的偏移量: {shift_y:.1f}")
                else:
                    logger.info(f"置信度不足({confidence:.2f})，使用默认偏移")
        
        # 如果没有检测到偏移量，使用默认值
        if shift_y is None:
            # 默认重叠约 20%
            shift_y = -(min(img_top.shape[0], img_bottom.shape[0]) * 0.2)
            logger.info(f"使用默认偏移量: {shift_y:.1f}")
        
        # 执行拼接
        result = self.blend_with_mask(img_top, img_bottom, shift_y)
        
        logger.info(f"拼接完成: {result.shape[1]} x {result.shape[0]}")
        return True, result, estimated_shift


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="竖向平移矩阵拼接工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--top", "-t",
        type=Path,
        required=True,
        help="上面的图像路径"
    )
    
    parser.add_argument(
        "--bottom", "-b",
        type=Path,
        required=True,
        help="下面的图像路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="输出图像路径"
    )
    
    parser.add_argument(
        "--shift", "-s",
        type=float,
        default=None,
        help="垂直偏移量（像素），正数表示相机向下移动，负数表示向上移动。不指定则自动检测"
    )
    
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="禁用自动检测偏移量"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="重叠区域比例 (0.1-0.5)"
    )
    
    args = parser.parse_args()
    
    # 加载图像
    img_top = cv2.imread(str(args.top))
    img_bottom = cv2.imread(str(args.bottom))
    
    if img_top is None or img_bottom is None:
        logger.error("无法读取图像")
        return
    
    # 创建拼接器
    stitcher = VerticalTranslateStitcher()
    
    # 执行拼接
    success, result, estimated_shift = stitcher.stitch(
        img_top, img_bottom,
        shift_y=args.shift,
        auto_detect=not args.no_auto
    )
    
    if success:
        cv2.imwrite(str(args.output), result)
        logger.info(f"结果已保存到: {args.output}")
        logger.info(f"输出尺寸: {result.shape[1]} x {result.shape[0]}")
        if estimated_shift is not None:
            logger.info(f"自动检测的偏移量: {estimated_shift:.1f} 像素")
    else:
        logger.error("拼接失败")


if __name__ == "__main__":
    main()
