#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import time

"""
完整代码学习 - 带窗口显示版（用于演示视频录制）
"""

# 中文显示配置（不再使用 Agg 后端，恢复 GUI 显示）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 不使用 plt.switch_backend('Agg')，让 matplotlib 自动选择 GUI 后端


# =========================
# 天空分割相关函数
# =========================
def iterative_threshold(gray, eps=1e-3):
    gray_f = gray.astype(np.float64)
    T = np.mean(gray_f)
    while True:
        fg = gray_f[gray_f > T]
        bg = gray_f[gray_f <= T]
        if len(fg) == 0 or len(bg) == 0:
            break
        new_T = (np.mean(fg) + np.mean(bg)) / 2.0
        if abs(new_T - T) < eps:
            break
        T = new_T
    return T


def get_gray_image(img):
    img_u8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    return gray, img_u8


def canny_edges(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(blur)
    low = int(max(0, 0.66 * med))
    high = int(min(255, 1.33 * med))
    edges = cv2.Canny(blur, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def build_sky_prior(gray):
    T = iterative_threshold(gray)
    prior = (gray > T).astype(np.uint8)
    return prior, T


def column_sky_boundary(gray, edges, prior,
                        max_scan_ratio=0.78, window_half=5,
                        edge_thresh=0.08, var_thresh=140.0, gray_drop_thresh=16.0):
    h, w = gray.shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    max_y = int(h * max_scan_ratio)

    for x in range(w):
        stop_y = -1
        prev_mean = None
        for y in range(0, max_y):
            if prior[y, x] == 0:
                stop_y = y - 1
                break
            y1 = max(0, y - window_half)
            y2 = min(h, y + window_half + 1)
            x1 = max(0, x - window_half)
            x2 = min(w, x + window_half + 1)

            patch_gray = gray[y1:y2, x1:x2]
            patch_edges = edges[y1:y2, x1:x2]

            if patch_gray.size == 0:
                continue
            edge_ratio = np.mean(patch_edges > 0)
            gray_var = np.var(patch_gray)
            mean_gray = np.mean(patch_gray)

            if edge_ratio > edge_thresh or gray_var > var_thresh:
                stop_y = y - 1
                break
            if prev_mean is not None and abs(mean_gray - prev_mean) > gray_drop_thresh:
                stop_y = y - 1
                break
            prev_mean = mean_gray

        if stop_y < 0:
            continue
        final_mask[:stop_y + 1, x] = 1
    return final_mask


def smooth_mask(mask):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    return mask


def adaptive_sky_segmentation(img):
    gray, img_u8 = get_gray_image(img)
    edges = canny_edges(gray)
    prior, T = build_sky_prior(gray)
    final_mask = column_sky_boundary(gray, edges, prior)
    final_mask = smooth_mask(final_mask)
    return final_mask.astype(bool), gray, edges, prior.astype(bool), T


def visualize_result(img, gray, edges, prior, final_mask):
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()
    overlay[final_mask] = [255, 0, 0]
    blended = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

    # 灰度图
    plt.figure(figsize=(8, 6))
    plt.imshow(gray, cmap='gray')
    plt.title("灰度图")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("灰度图.png", bbox_inches='tight', dpi=200)
    plt.show()  # 显示窗口

    # Canny边缘图
    plt.figure(figsize=(8, 6))
    plt.imshow(edges, cmap='gray')
    plt.title("Canny 边缘")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("Canny边缘图.png", bbox_inches='tight', dpi=200)
    plt.show()

    # 天空先验图
    plt.figure(figsize=(8, 6))
    plt.imshow(prior, cmap='gray')
    plt.title("天空先验图")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("天空先验图.png", bbox_inches='tight', dpi=200)
    plt.show()

    # 最终天空mask
    plt.figure(figsize=(8, 6))
    plt.imshow(final_mask, cmap='gray')
    plt.title("最终天空mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("最终天空mask.png", bbox_inches='tight', dpi=200)
    plt.show()

    # 天空区域标记图
    plt.figure(figsize=(8, 6))
    plt.imshow(blended)
    plt.title("天空区域标记")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("天空区域标记.png", bbox_inches='tight', dpi=200)
    plt.show()


# =========================
# 暗通道 + 大气光（天空区域中位数）
# =========================
def cal_Dark_Channel(image, patch_size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def cal_Light_A(dark_channel, img, sky_mask):
    dark_sky = dark_channel.copy()
    dark_sky[~sky_mask] = 0

    sky_y, sky_x = np.where(sky_mask)
    if len(sky_y) == 0:
        print("警告：未检测到天空区域，使用全图计算大气光")
        sky_y, sky_x = np.where(np.ones_like(dark_channel, dtype=bool))

    dark_vals = dark_sky[sky_y, sky_x]
    num = max(10, int(len(dark_vals) * 0.001))
    indices = np.argsort(dark_vals)[-num:]

    top_y = sky_y[indices]
    top_x = sky_x[indices]
    A = np.median(img[top_y, top_x, :], axis=0)

    light_points = list(zip(top_y, top_x))
    print("大气光 A (BGR顺序):", A)
    return A, light_points


def visualize_atmospheric_light(img, sky_mask, light_points):
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.imshow(sky_mask, cmap='Blues', alpha=0.3)
    y, x = zip(*light_points)
    plt.scatter(x, y, c='red', s=20, marker='o', label='大气光取点')
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig("大气光取点可视化.png", bbox_inches='tight', dpi=200)
    plt.show()


# =========================
# 透射率修正 + 引导滤波 + 去雾
# =========================
def region_correction_factor(img_gray, trans, alpha=0.6, beta=0.4):
    brightness = img_gray.copy()
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    k = alpha * brightness - beta * grad_mag
    k = np.clip(k, 0, 1)
    trans_corrected = trans * (1 + 0.5 * k)
    return np.clip(trans_corrected, 0, 1)



def cal_trans(A, img, w=0.95):
    dark = cal_Dark_Channel(img / (A + 1e-8))
    t = np.maximum(1 - w * dark, 0)
    return t


def Guided_filtering(t, img_gray, width, sigma=0.0001):
    img_gray = img_gray.astype(np.float64)
    t = t.astype(np.float64)
    mean_I = cv2.boxFilter(img_gray, -1, (width, width), normalize=True)
    mean_t = cv2.boxFilter(t, -1, (width, width), normalize=True)
    corr_I = cv2.boxFilter(img_gray * img_gray, -1, (width, width), normalize=True)
    corr_IT = cv2.boxFilter(img_gray * t, -1, (width, width), normalize=True)
    var_I = corr_I - mean_I * mean_I
    cov_IT = corr_IT - mean_I * mean_t
    a = cov_IT / (var_I + sigma)
    b = mean_t - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (width, width), normalize=True)
    mean_b = cv2.boxFilter(b, -1, (width, width), normalize=True)
    return mean_a * img_gray + mean_b


def harz_Rec(A, img, t, t0=0.1):
    img_o = np.zeros_like(img)
    t = np.maximum(t, t0)[:, :, np.newaxis]
    img_o = (img - A) / t + A
    return img_o


def calculate_metrics(hazy, dehazed, clean):
    hazy = hazy.astype(np.float64)
    dehazed = dehazed.astype(np.float64)
    clean = clean.astype(np.float64)

    if hazy.shape != clean.shape:
        print("警告：有雾图像与参考图像尺寸不一致，将参考图像缩放到有雾图像尺寸")
        clean = cv2.resize(clean, (hazy.shape[1], hazy.shape[0]))
    if dehazed.shape != clean.shape:
        print("警告：去雾图像与参考图像尺寸不一致，将参考图像缩放到去雾图像尺寸")
        clean = cv2.resize(clean, (dehazed.shape[1], dehazed.shape[0]))

    psnr_hazy = psnr(clean, hazy, data_range=1.0)
    psnr_dehazed = psnr(clean, dehazed, data_range=1.0)

    ssim_hazy = ssim(clean, hazy, data_range=1.0, multichannel=True)
    ssim_dehazed = ssim(clean, dehazed, data_range=1.0, multichannel=True)

    print(f"有雾图像 vs 参考图像： PSNR = {psnr_hazy:.2f} dB, SSIM = {ssim_hazy:.4f}")
    print(f"去雾图像 vs 参考图像： PSNR = {psnr_dehazed:.2f} dB, SSIM = {ssim_dehazed:.4f}")

    return psnr_hazy, ssim_hazy, psnr_dehazed, ssim_dehazed


# =========================
# 主函数（显示窗口版）
# =========================
if __name__ == '__main__':
    start_time = time.time()

    # 图像路径
    hazy_path = 'input/hazy/hazy3.0.png'
    gt_path = 'input/clear/GT3.0.png'

    img = cv2.imread(hazy_path) / 255.0
    img_GT = cv2.imread(gt_path) / 255.0

    if img is None or img_GT is None:
        print("错误：请检查图像路径是否正确！")
    else:
        print("===== 开始演示 =====")

        # 天空分割
        final_mask, gray, edges, prior, T = adaptive_sky_segmentation(img)
        cv2.imwrite("sky_mask.png", (final_mask.astype(np.uint8) * 255))
        visualize_result(img, gray, edges, prior, final_mask)
        print("✅ 天空分割完成，请关闭当前图像窗口继续...")


        img_gray = gray / 255.0

        # 暗通道图
        im_dark = cal_Dark_Channel(img)
        plt.figure(figsize=(8, 6))
        plt.imshow(im_dark, 'gray')
        plt.title('暗通道')
        plt.axis('off')
        plt.savefig('暗通道.png', bbox_inches='tight', dpi=200)
        plt.show()
        print("暗通道图已显示，关闭窗口继续")


        # 大气光+可视化
        A, light_points = cal_Light_A(im_dark, img, final_mask)
        visualize_atmospheric_light(img, final_mask, light_points)
        print("大气光取点图已显示，关闭窗口继续")


        # 初始透射率
        trans = cal_trans(A, img)
        plt.figure(figsize=(8, 6))
        plt.imshow(trans, 'gray')
        plt.title('初始透射率')
        plt.axis('off')
        plt.savefig('初始透射率.png', bbox_inches='tight', dpi=200)
        plt.show()


        # 区域修正透射率
        trans_region = region_correction_factor(img_gray, trans)
        plt.figure(figsize=(8, 6))
        plt.imshow(trans_region, 'gray')
        plt.title('区域修正后透射率')
        plt.axis('off')
        plt.savefig('区域修正后透射率.png', bbox_inches='tight', dpi=200)
        plt.show()


        # 引导滤波透射率
        trans_refined = Guided_filtering(trans_region, img_gray, 41)
        trans_refined = np.clip(trans_refined, 0, 1)
        plt.figure(figsize=(8, 6))
        plt.imshow(trans_refined, 'gray')
        plt.title('引导滤波改进透射率')
        plt.axis('off')
        plt.savefig('引导滤波改进透射率.png', bbox_inches='tight', dpi=200)
        plt.show()


        # 去雾结果
        result = harz_Rec(A, img, trans_refined)
        result = np.clip(result, 0, 1)
        result_bgr = (result * 255).astype(np.uint8)
        cv2.imwrite("out.png", result_bgr)
        print(f"✅ 去雾结果已保存，尺寸：{result_bgr.shape[1]}x{result_bgr.shape[0]}")

        # 显示去雾结果
        plt.figure(figsize=(8, 6))
        plt.imshow(result[:, :, ::-1])  # BGR转RGB显示
        plt.title('无雾气图像')
        plt.axis('off')
        plt.savefig('无雾气图像.png', bbox_inches='tight', dpi=200)
        plt.show()


        # 计算指标
        calculate_metrics(img, result, img_GT)

        # 额外显示原图与参考图对比
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img[:, :, ::-1])
        plt.title("原图（有雾）")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(result[:, :, ::-1])
        plt.title("去雾结果")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_GT[:, :, ::-1])
        plt.title("参考清晰图")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("对比图.png", bbox_inches='tight', dpi=200)
        plt.show()


    # 精准计时输出
    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 60)
    print(f"⏱️ 程序总运行时间：{total_time:.2f} 秒 ({total_time * 1000:.0f} 毫秒)")
    print("🎉 全部处理完成！")
    print("-" * 60)