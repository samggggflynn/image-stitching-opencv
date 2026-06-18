"""竖向图像拼接模块

提供多种竖向拼接算法：
1. FeatureBasedStitcher - 基于特征匹配的精确拼接
2. TranslationStitcher - 基于平移矩阵的简单拼接
3. SimpleStitcher - 简单垂直拼接（固定重叠）
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseVerticalStitcher:
    """竖向拼接器基类"""
    
    def __init__(self):
        pass
    
    def stitch(self, img_top, img_bottom):
        """
        执行竖向拼接
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
        
        Returns:
            (success, result)
        """
        raise NotImplementedError("子类必须实现 stitch 方法")
    
    @staticmethod
    def align_width(img1, img2):
        """对齐两张图像的宽度"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if w1 == w2:
            return img1, img2
        
        target_w = min(w1, w2)
        img1 = cv2.resize(img1, (target_w, int(h1 * target_w / w1)))
        img2 = cv2.resize(img2, (target_w, int(h2 * target_w / w2)))
        
        return img1, img2
    
    @staticmethod
    def blend_overlap(img_top, img_bottom, result, top_y, bottom_start):
        """
        渐变融合重叠区域
        
        Args:
            img_top: 上图（已裁剪）
            img_bottom: 下图
            result: 输出图像
            top_y: 上图起始位置
            bottom_start: 下图起始位置
        
        Returns:
            融合后的图像
        """
        h1 = img_top.shape[0]
        h2 = img_bottom.shape[0]
        
        overlap_start = max(top_y, bottom_start)
        overlap_end = min(top_y + h1, bottom_start + h2)
        
        if overlap_start < overlap_end:
            overlap_height = overlap_end - overlap_start
            for y in range(overlap_height):
                alpha = y / overlap_height
                y_result = overlap_start + y
                y_top = y_result - top_y
                y_bottom = y_result - bottom_start
                
                if 0 <= y_top < h1 and 0 <= y_bottom < h2:
                    result[y_result, :] = (
                        img_top[y_top, :] * (1 - alpha) +
                        img_bottom[y_bottom, :] * alpha
                    ).astype(np.uint8)
        
        return result


class SimpleStitcher(BaseVerticalStitcher):
    """简单垂直拼接器（固定重叠比例）"""
    
    def __init__(self, overlap_ratio=0.2):
        """
        Args:
            overlap_ratio: 重叠区域比例 (0.1-0.5)
        """
        self.overlap_ratio = overlap_ratio
    
    def stitch(self, img_top, img_bottom):
        """执行简单垂直拼接"""
        img_top, img_bottom = self.align_width(img_top, img_bottom)
        
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        # 计算重叠区域
        overlap_h = int(min(h1, h2) * self.overlap_ratio)
        total_h = h1 + h2 - overlap_h
        
        # 创建输出图像
        result = np.zeros((total_h, w1, 3), dtype=np.uint8)
        
        # 放置上图
        result[:h1, :w1] = img_top
        
        # 放置下图
        bottom_start = h1 - overlap_h
        result[bottom_start:bottom_start + h2, :w1] = img_bottom
        
        # 融合重叠区域
        self.blend_overlap(img_top, img_bottom, result, 0, bottom_start)
        
        logger.info(f"简单垂直拼接完成: {w1} x {total_h}")
        return True, result


class CenterCutStitcher(BaseVerticalStitcher):
    """中心切割拼接器（重叠区域中间分割 + 渐变融合）"""
    
    def __init__(self, overlap_ratio=0.2, blend_width=20):
        """
        Args:
            overlap_ratio: 重叠区域比例 (0.1-0.5)
            blend_width: 渐变融合宽度（像素）
        """
        self.overlap_ratio = overlap_ratio
        self.blend_width = blend_width
    
    def stitch(self, img_top, img_bottom, shift_y=None):
        """
        执行中心切割拼接：
        - 支持手动指定垂直偏移量
        - 重叠区域的上半部分使用上图的像素
        - 重叠区域的下半部分使用下图的像素
        - 在切割线附近进行渐变融合
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
            shift_y: 手动指定垂直偏移量（像素），负数表示上图向下偏移
        
        Returns:
            (success, result)
        """
        img_top, img_bottom = self.align_width(img_top, img_bottom)
        
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        # 计算重叠区域
        overlap_h = int(min(h1, h2) * self.overlap_ratio)
        
        # 如果指定了偏移量，使用偏移量计算重叠
        if shift_y is not None:
            # shift_y 为负数表示上图向下偏移（即重叠减少）
            # 转换为重叠高度的调整
            adjusted_overlap = int(overlap_h + shift_y)
            overlap_h = max(int(min(h1, h2) * 0.1), min(adjusted_overlap, int(min(h1, h2) * 0.5)))
        
        total_h = h1 + h2 - overlap_h
        
        # 创建输出图像
        result = np.zeros((total_h, w1, 3), dtype=np.uint8)
        
        # 计算关键位置
        # 重叠区域在上图中的起始位置
        overlap_start_top = h1 - overlap_h
        # 重叠区域在下图中的起始位置
        overlap_start_bottom = 0
        # 中心切割位置（在输出图像中的位置）
        cut_position = h1 - overlap_h // 2
        
        # 放置上图（只放置到切割位置）
        result[:cut_position, :w1] = img_top[:cut_position, :]
        
        # 放置下图（从切割位置开始）
        result[cut_position:cut_position + h2, :w1] = img_bottom
        
        # 在切割线附近进行渐变融合
        # 融合区域：切割线上下各 blend_width 像素
        blend_start = max(overlap_start_top, cut_position - self.blend_width)
        blend_end = min(cut_position + overlap_h // 2, cut_position + self.blend_width)
        
        if blend_start < blend_end:
            for y in range(blend_start, blend_end):
                # 计算权重：切割线上方上图权重高，下方下图权重高
                y_in_blend = y - (cut_position - self.blend_width)
                if y_in_blend < self.blend_width:
                    # 切割线上方：上图权重从1逐渐降低
                    alpha = y_in_blend / (2 * self.blend_width)
                else:
                    # 切割线下方：下图权重逐渐增加
                    alpha = 1.0 - (2 * self.blend_width - y_in_blend) / (2 * self.blend_width)
                
                y_top = y
                y_bottom = y - cut_position
                
                if 0 <= y_top < h1 and 0 <= y_bottom < h2:
                    result[y, :] = (
                        img_top[y_top, :] * (1 - alpha) +
                        img_bottom[y_bottom, :] * alpha
                    ).astype(np.uint8)
        
        logger.info(f"中心切割拼接完成: {w1} x {total_h} (重叠高度: {overlap_h}, 切割位置: {cut_position})")
        return True, result


class FeatureBasedStitcher(BaseVerticalStitcher):
    """基于特征匹配的竖向拼接器"""
    
    def __init__(self, min_matches=10, ransac_threshold=3.0, vertical_filter=True):
        """
        Args:
            min_matches: 最小匹配点数量
            ransac_threshold: RANSAC 阈值
            vertical_filter: 是否过滤非竖向匹配点
        """
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.vertical_filter = vertical_filter
        
        # SIFT 检测器
        self.sift = cv2.SIFT_create(nfeatures=10000)
        
        # FLANN 匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def _detect_features(self, image):
        """检测特征点和描述符"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.sift.detectAndCompute(gray, None)
    
    def _match_features(self, des1, des2):
        """特征匹配"""
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def _filter_vertical_matches(self, kp1, kp2, matches, max_x_diff=50):
        """过滤非竖向匹配点（x坐标应相近）"""
        if not self.vertical_filter:
            return matches
        
        filtered = []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            if abs(pt1[0] - pt2[0]) < max_x_diff:
                filtered.append(match)
        
        return filtered
    
    def _compute_homography(self, kp1, kp2, matches):
        """计算单应性矩阵"""
        if len(matches) < self.min_matches:
            return None, 0
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            return None, 0
        
        return H, mask.sum()
    
    def stitch(self, img_top, img_bottom):
        """执行基于特征匹配的竖向拼接"""
        img_top, img_bottom = self.align_width(img_top, img_bottom)
        
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        # 检测特征
        kp1, des1 = self._detect_features(img_top)
        kp2, des2 = self._detect_features(img_bottom)
        
        logger.info(f"特征点: top={len(kp1)}, bottom={len(kp2)}")
        
        if len(kp1) < 20 or len(kp2) < 20:
            logger.warning("特征点太少")
            return False, None
        
        # 匹配特征
        matches = self._match_features(des1, des2)
        logger.info(f"初步匹配: {len(matches)}")
        
        if len(matches) < self.min_matches:
            logger.warning("匹配点不足")
            return False, None
        
        # 过滤竖向匹配点
        if self.vertical_filter:
            matches = self._filter_vertical_matches(kp1, kp2, matches)
            logger.info(f"竖向过滤后: {len(matches)}")
        
        if len(matches) < self.min_matches:
            logger.warning("竖向匹配点不足")
            return False, None
        
        # 计算单应性矩阵
        H, inliers = self._compute_homography(kp1, kp2, matches)
        logger.info(f"有效匹配点(RANSAC): {inliers}")
        
        if H is None or inliers < 8:
            logger.warning("无法计算可靠的单应性矩阵")
            return False, None
        
        # 计算变换后边界
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        
        min_x, min_y = np.int32(transformed.min(axis=0).ravel())
        max_x, max_y = np.int32(transformed.max(axis=0).ravel())
        
        # 检查输出尺寸是否合理
        output_w = max(max_x, w2) - min(min_x, 0)
        output_h = max(max_y, h2) - min(min_y, 0)
        
        if output_w > w1 * 3 or output_h > h1 * 3:
            logger.warning(f"输出尺寸异常({output_w}x{output_h})")
            return False, None
        
        # 创建平移矩阵
        translation = np.array([[1, 0, -min(min_x, 0)], 
                                [0, 1, -min(min_y, 0)], 
                                [0, 0, 1]], dtype=np.float64)
        H_adjusted = translation @ H
        
        # 变换并拼接
        warped = cv2.warpPerspective(img_top, H_adjusted, (output_w, output_h))
        result = warped.copy()
        
        # 放置下图
        y_offset = -min(min_y, 0)
        x_offset = -min(min_x, 0)
        result[y_offset:y_offset+h2, x_offset:x_offset+w2] = img_bottom
        
        # 融合重叠区域
        overlap_mask = (warped > 0) & (result > 0)
        if overlap_mask.any():
            min_y = np.where(overlap_mask)[0].min()
            max_y = np.where(overlap_mask)[0].max()
            
            weight = np.zeros(result.shape[:2], dtype=np.float32)
            weight[min_y:max_y+1, :] = np.linspace(0.0, 1.0, max_y - min_y + 1).reshape(-1, 1)
            
            for c in range(3):
                result[..., c] = (warped[..., c] * (1 - weight) + 
                                result[..., c] * weight).astype(np.uint8)
        
        logger.info(f"特征匹配拼接完成: {output_w} x {output_h}")
        return True, result


class TranslationStitcher(BaseVerticalStitcher):
    """基于平移矩阵的竖向拼接器"""
    
    def __init__(self):
        # SIFT 检测器用于自动检测偏移
        self.sift = cv2.SIFT_create(nfeatures=5000)
        
        # FLANN 匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def _detect_and_match(self, img1, img2):
        """检测特征点并匹配"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        return kp1, kp2, good_matches
    
    def _estimate_shift(self, kp1, kp2, matches):
        """估计垂直偏移量"""
        if len(matches) < 4:
            return None, 0.0
        
        y_diffs = []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            if abs(pt1[0] - pt2[0]) < 50:
                y_diffs.append(pt2[1] - pt1[1])
        
        if len(y_diffs) < 4:
            return None, 0.0
        
        y_diffs = np.array(y_diffs)
        estimated_shift = np.median(y_diffs)
        std_dev = np.std(y_diffs)
        confidence = max(0.0, min(1.0, 1.0 - std_dev / 100.0))
        
        return estimated_shift, confidence
    
    def stitch(self, img_top, img_bottom, shift_y=None, auto_detect=True):
        """
        执行基于平移矩阵的竖向拼接
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
            shift_y: 手动指定垂直偏移量（可选）
            auto_detect: 是否自动检测偏移量
        
        Returns:
            (success, result, estimated_shift)
        """
        img_top, img_bottom = self.align_width(img_top, img_bottom)
        
        h1, w1 = img_top.shape[:2]
        h2, w2 = img_bottom.shape[:2]
        
        estimated_shift = None
        
        # 自动检测偏移量
        if auto_detect and shift_y is None:
            kp1, kp2, matches = self._detect_and_match(img_top, img_bottom)
            logger.info(f"特征匹配数: {len(matches)}")
            
            if len(matches) >= 4:
                estimated_shift, confidence = self._estimate_shift(kp1, kp2, matches)
                logger.info(f"估计垂直偏移: {estimated_shift:.1f} 像素, 置信度: {confidence:.2f}")
                
                if confidence > 0.3:
                    shift_y = estimated_shift
        
        # 使用默认偏移量
        if shift_y is None:
            shift_y = -(min(h1, h2) * 0.2)
            logger.info(f"使用默认偏移量: {shift_y:.1f}")
        
        # 计算输出尺寸
        overlap_h = int(min(h1, h2) * 0.3)
        total_h = h1 + h2 - overlap_h
        result = np.zeros((total_h, w1, 3), dtype=np.uint8)
        
        # 放置上图（考虑偏移）
        adjusted_shift = shift_y
        top_y = int(max(0, -adjusted_shift))
        
        # 裁剪上图（如果偏移导致超出边界）
        cropped_top = img_top[top_y:, :] if top_y > 0 else img_top
        cropped_h1 = cropped_top.shape[0]
        
        # 确保上图不超出画布
        if cropped_h1 > total_h:
            cropped_top = cropped_top[:total_h, :]
            cropped_h1 = total_h
        
        result[:cropped_h1, :w1] = cropped_top
        
        # 放置下图
        bottom_start = total_h - h2
        bottom_start = int(bottom_start + adjusted_shift * 0.5)
        bottom_start = max(0, min(bottom_start, total_h - h2))
        result[bottom_start:bottom_start + h2, :w1] = img_bottom
        
        # 融合重叠区域
        self.blend_overlap(cropped_top, img_bottom, result, 0, bottom_start)
        
        logger.info(f"平移拼接完成: {w1} x {total_h}")
        return True, result, estimated_shift


class AutoVerticalStitcher(BaseVerticalStitcher):
    """自动选择最佳算法的竖向拼接器"""
    
    def __init__(self):
        self.feature_stitcher = FeatureBasedStitcher()
        self.translation_stitcher = TranslationStitcher()
        self.simple_stitcher = SimpleStitcher()
        self.center_cut_stitcher = CenterCutStitcher()
    
    def stitch(self, img_top, img_bottom, mode='auto', **kwargs):
        """
        执行竖向拼接
        
        Args:
            img_top: 上面的图像
            img_bottom: 下面的图像
            mode: 'auto'自动选择, 'feature'特征匹配, 'translation'平移矩阵, 'simple'简单拼接, 'centercut'中心切割
            **kwargs: 额外参数（如 shift_y）
        
        Returns:
            (success, result)
        """
        if mode == 'feature':
            success, result = self.feature_stitcher.stitch(img_top, img_bottom)
            if not success:
                logger.info("特征匹配失败，回退到简单拼接")
                return True, self.simple_stitcher.stitch(img_top, img_bottom)[1]
            return success, result
        
        if mode == 'translation':
            shift_y = kwargs.get('shift_y')
            auto_detect = kwargs.get('auto_detect', True)
            success, result, _ = self.translation_stitcher.stitch(img_top, img_bottom, shift_y=shift_y, auto_detect=auto_detect)
            return success, result
        
        if mode == 'simple':
            return self.simple_stitcher.stitch(img_top, img_bottom)
        
        if mode == 'centercut':
            shift_y = kwargs.get('shift_y')
            return self.center_cut_stitcher.stitch(img_top, img_bottom, shift_y=shift_y)
        
        # auto 模式：依次尝试各种算法
        logger.info("自动选择拼接算法...")
        
        # 先尝试特征匹配
        success, result = self.feature_stitcher.stitch(img_top, img_bottom)
        if success:
            logger.info("使用特征匹配拼接")
            return True, result
        
        # 再尝试平移矩阵
        success, result, _ = self.translation_stitcher.stitch(img_top, img_bottom)
        if success:
            logger.info("使用平移矩阵拼接")
            return True, result
        
        # 最后使用简单拼接
        logger.info("使用简单垂直拼接")
        return self.simple_stitcher.stitch(img_top, img_bottom)


# 便捷函数
def stitch_vertical(img_top, img_bottom, mode='auto', **kwargs):
    """
    竖向拼接便捷函数
    
    Args:
        img_top: 上面的图像
        img_bottom: 下面的图像
        mode: 'auto', 'feature', 'translation', 'simple'
        **kwargs: 额外参数
    
    Returns:
        拼接后的图像
    """
    stitcher = AutoVerticalStitcher()
    success, result = stitcher.stitch(img_top, img_bottom, mode)
    
    if not success:
        raise RuntimeError("拼接失败")
    
    return result
