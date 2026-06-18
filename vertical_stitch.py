#!/usr/bin/env python3
"""
统一竖向图像拼接命令行工具

整合了以下原脚本的功能：
- stitch_vertical.py: ORB + 单应性矩阵
- stitch_vertical_enhanced.py: SIFT + FLANN + 竖向过滤
- stitch_vertical_final.py: 自动选择模式
- stitch_vertical_translate.py: 平移矩阵 + 手动偏移

Usage:
    python vertical_stitch.py -t top.png -b bottom.png -o result.png
"""

import argparse
import cv2
from pathlib import Path
import logging

# 导入模块化的拼接器
from modules import AutoVerticalStitcher, TranslationStitcher, CenterCutStitcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="竖向图像拼接工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
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
    
    # 算法选择
    parser.add_argument(
        "--mode", "-m",
        choices=['auto', 'feature', 'translation', 'simple', 'centercut'],
        default='auto',
        help="拼接算法模式: auto(自动), feature(特征匹配), translation(平移矩阵), simple(简单拼接), centercut(中心切割)"
    )
    
    # 平移矩阵模式的额外参数
    parser.add_argument(
        "--shift", "-s",
        type=float,
        default=None,
        help="垂直偏移量（像素），仅在 translation 模式下生效"
    )
    
    parser.add_argument(
        "--no-auto-shift",
        action="store_true",
        help="禁用自动检测偏移量"
    )
    
    args = parser.parse_args()
    
    # 加载图像
    img_top = cv2.imread(str(args.top))
    img_bottom = cv2.imread(str(args.bottom))
    
    if img_top is None or img_bottom is None:
        logger.error("无法读取图像")
        return
    
    logger.info(f"输入图像: {args.top.name} ({img_top.shape}) + {args.bottom.name} ({img_bottom.shape})")
    
    # 根据模式选择拼接器
    if args.mode == 'translation':
        # 使用平移矩阵拼接器（支持手动偏移）
        stitcher = TranslationStitcher()
        success, result, estimated_shift = stitcher.stitch(
            img_top, img_bottom,
            shift_y=args.shift,
            auto_detect=not args.no_auto_shift
        )
        
        if success:
            cv2.imwrite(str(args.output), result)
            logger.info(f"结果已保存到: {args.output}")
            logger.info(f"输出尺寸: {result.shape[1]} x {result.shape[0]}")
            if estimated_shift is not None:
                logger.info(f"自动检测的偏移量: {estimated_shift:.1f} 像素")
        else:
            logger.error("拼接失败")
    
    elif args.mode == 'centercut':
        # 使用中心切割拼接器（支持手动偏移）
        stitcher = CenterCutStitcher()
        success, result = stitcher.stitch(img_top, img_bottom, shift_y=args.shift)
        
        if success:
            cv2.imwrite(str(args.output), result)
            logger.info(f"结果已保存到: {args.output}")
            logger.info(f"输出尺寸: {result.shape[1]} x {result.shape[0]}")
            if args.shift is not None:
                logger.info(f"手动指定的偏移量: {args.shift} 像素")
        else:
            logger.error("拼接失败")
    
    else:
        # 使用自动拼接器
        stitcher = AutoVerticalStitcher()
        success, result = stitcher.stitch(img_top, img_bottom, mode=args.mode)
        
        if success:
            cv2.imwrite(str(args.output), result)
            logger.info(f"结果已保存到: {args.output}")
            logger.info(f"输出尺寸: {result.shape[1]} x {result.shape[0]}")
        else:
            logger.error("拼接失败")


if __name__ == "__main__":
    main()
