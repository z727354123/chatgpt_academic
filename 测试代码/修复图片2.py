import cv2
import numpy as np

if __name__ == '__main__':
    # 读取二维码图像
    img = cv2.imread('/Users/judy/Downloads/123_png.png')

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用高斯滤波，进行平滑处理
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 应用自适应二值化
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 7)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 获取轮廓区域的大小
        area = cv2.contourArea(cnt)
        # 若轮廓区域太小，则忽略处理
        if area < 10:
            continue
        # 创建仿射变换的坐标点
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        # 画出二维码区域
        cv2.drawContours(img, [box], 0, (0, 0, 0), 3)

    # 显示处理后的图像
    cv2.imshow("Result", img)
    cv2.waitKey(0)