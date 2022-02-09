import cv2
import os


original_hr_img_dir = r"/home/yuyang/data/bluetooth/valid_image"
hr_img_patch_save_dir = r"/home/yuyang/data/sr_data/valid_img"
hr_img_patch_h = 300  # should bigger than LR_size[0] * r
hr_img_patch_w = 300  # should bigger than LR_size[1] * r


def crop_one_hr_img(cv2_hr_img):
    original_hr_img_h, original_hr_img_w = cv2_hr_img.shape[:2]
    h_patch_count = int(original_hr_img_h // hr_img_patch_h)
    w_patch_count = int(original_hr_img_w // hr_img_patch_w)
    for r in range(h_patch_count):
        for c in range(w_patch_count):
            img_patch = cv2_hr_img[r * hr_img_patch_h:(r + 1) * hr_img_patch_h, c * hr_img_patch_w:(c + 1) * hr_img_patch_w, :]
            img_name = "%d.png" % (len(os.listdir(hr_img_patch_save_dir)),)
            cv2.imwrite(os.path.join(hr_img_patch_save_dir, img_name), img_patch)


def crop_all_hr_img():
    hr_img_pths = [os.path.join(original_hr_img_dir, i) for i in os.listdir(original_hr_img_dir)]
    for hr_img_pth in hr_img_pths:
        cv2_hr_img = cv2.imread(hr_img_pth)
        crop_one_hr_img(cv2_hr_img)


if __name__ == "__main__":
    crop_all_hr_img()