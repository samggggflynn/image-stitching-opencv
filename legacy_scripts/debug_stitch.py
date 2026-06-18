#!/usr/bin/env python3
"""调试脚本：分析特征点匹配情况"""
import cv2
import numpy as np
from pathlib import Path

img_dir = Path('/home/zjt/dev/On_Git_Projects/image-stitching-opencv/images/featured_sample1/tobestiched')

# 加载图像
img1 = cv2.imread(str(img_dir / 'up_back_rgb_Color.png'))
img2 = cv2.imread(str(img_dir / 'down_back_rgb_Color.png'))

print(f"图像 1 尺寸：{img1.shape}")
print(f"图像 2 尺寸：{img2.shape}")

# 使用 SIFT 检测特征点
sift = cv2.SIFT_create()

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"\n图像 1 特征点：{len(kp1)}")
print(f"图像 2 特征点：{len(kp2)}")

# 特征点匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"良好匹配点数量：{len(good_matches)}")

# 可视化匹配结果
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite('debug_matches.png', img_matches)
print(f"\n匹配可视化已保存到：debug_matches.png")

# 尝试不同模式
print("\n=== 测试不同拼接模式 ===")
for mode_name, mode in [('PANORAMA', cv2.Stitcher_PANORAMA)]:
    stitcher = cv2.Stitcher.create(mode)
    status, result = stitcher.stitch([img1, img2])
    print(f"{mode_name} 模式：状态码 {status}")
    if status == 0:
        print(f"  ✓ 成功！输出尺寸：{result.shape}")
        cv2.imwrite(f'result_{mode_name.lower()}.png', result)
    else:
        status_msgs = {
            1: "需要更多图像",
            2: "同质性检查失败",
            3: "相机参数调整失败",
            4: "波形校正失败"
        }
        print(f"  ✗ 失败：{status_msgs.get(status, f'未知错误 {status}')}")

# 尝试手动拼接（更宽松的参数）
print("\n=== 尝试手动拼接（降低阈值）===")
orb = cv2.ORB_create(nfeatures=5000)
kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
matches_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)

good_orb = []
for m, n in matches_orb:
    if m.distance < 0.8 * n.distance:
        good_orb.append(m)

print(f"ORB 特征点匹配数：{len(good_orb)}")

if len(good_orb) >= 4:
    src_pts = np.float32([kp1_orb[m.queryIdx].pt for m in good_orb]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2_orb[m.trainIdx].pt for m in good_orb]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"单应性矩阵计算：{'成功' if H is not None else '失败'}")
    print(f"有效匹配点：{mask.sum()}")
    
    if H is not None:
        result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2
        cv2.imwrite('result_manual.png', result)
        print("手动拼接结果已保存到：result_manual.png")
