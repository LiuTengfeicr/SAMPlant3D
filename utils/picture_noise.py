import cv2
import numpy as np


def remove_pic_noise(image_path, method='median', ksize=3):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'median':
        denoised_image = cv2.medianBlur(gray_image, ksize)
        denoised_image = cv2.medianBlur(denoised_image, ksize+2)
        denoised_image = cv2.medianBlur(denoised_image, ksize+4)
        denoised_image = cv2.medianBlur(denoised_image, ksize+6)
    elif method == 'gaussian':
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    elif method == 'bilateral':
        denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

    cv2.imwrite(image_path, denoised_image)
    return denoised_image, image_path


def remove_pic_noise1(image_path, method='median'):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'median':
        denoised_image = cv2.medianBlur(gray_image, 5)
        denoised_image = cv2.medianBlur(denoised_image, 5)
        denoised_image = cv2.medianBlur(denoised_image, 5)


    elif method == 'gaussian':
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    elif method == 'bilateral':
        denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

    cv2.imwrite(image_path, denoised_image)
    return denoised_image, image_path

