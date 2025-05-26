import cv2
import os
import random
import itertools

# 全局记录每个图像已接受的裁剪区域（以绝对坐标表示）
global_accepted_crops = {}

def iou_rect(rect1, rect2):
    """
    计算两个矩形区域的 IoU，其中 rect 格式为 (x1, y1, x2, y2)
    """
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def read_yolo_labels(label_file, img_w, img_h):
    """
    读取 YOLO 标签文件，将归一化坐标转换为绝对坐标，
    返回列表格式：[ (cls, x1, y1, x2, y2), ... ]
    """
    bboxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, w, h = parts
                try:
                    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                except:
                    continue
                xc_abs = xc * img_w
                yc_abs = yc * img_h
                w_abs = w * img_w
                h_abs = h * img_h
                x1 = xc_abs - w_abs / 2
                y1 = yc_abs - h_abs / 2
                x2 = xc_abs + w_abs / 2
                y2 = yc_abs + h_abs / 2
                bboxes.append((cls, x1, y1, x2, y2))
    return bboxes

def adjust_bbox_to_crop(bbox, crop_x, crop_y, crop_size):
    """
    将 bbox 从原图坐标转换为裁剪区域坐标，并转换为 YOLO 格式（归一化坐标）。
    bbox 格式：(cls, x1, y1, x2, y2)
    """
    cls, x1, y1, x2, y2 = bbox
    new_x1 = x1 - crop_x
    new_y1 = y1 - crop_y
    new_x2 = x2 - crop_x
    new_y2 = y2 - crop_y
    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1
    new_xc = new_x1 + new_w / 2
    new_yc = new_y1 + new_h / 2
    return (cls, new_xc / crop_size, new_yc / crop_size, new_w / crop_size, new_h / crop_size)

def generate_crop_for_target(img_w, img_h, target_box, crop_size):
    """
    随机生成一个裁剪区域 (crop_x, crop_y)，要求该区域完全包含 target_box。
    target_box: (cls, x1, y1, x2, y2)（绝对坐标）
    返回 (crop_x, crop_y)，若无法生成返回 None。
    随机选取范围：
        crop_x 范围：[ max(0, target_x2 - crop_size), min(target_x1, img_w - crop_size) ]
        crop_y 范围：[ max(0, target_y2 - crop_size), min(target_y1, img_h - crop_size) ]
    """
    _, tx1, ty1, tx2, ty2 = target_box
    min_crop_x = int(max(0, tx2 - crop_size))
    max_crop_x = int(min(tx1, img_w - crop_size))
    min_crop_y = int(max(0, ty2 - crop_size))
    max_crop_y = int(min(ty1, img_h - crop_size))
    if min_crop_x > max_crop_x or min_crop_y > max_crop_y:
        return None
    crop_x = random.randint(min_crop_x, max_crop_x)
    crop_y = random.randint(min_crop_y, max_crop_y)
    return (crop_x, crop_y)

def clip_bbox_to_crop(bbox, crop_x, crop_y, crop_size):
    """
    计算 bbox 与裁剪区域的交集，返回交集区域的坐标 (x1, y1, x2, y2)。
    如果没有交集，则返回 None。
    bbox 格式：(cls, x1, y1, x2, y2)
    裁剪区域为 (crop_x, crop_y, crop_x+crop_size, crop_y+crop_size)
    """
    _, x1, y1, x2, y2 = bbox
    crop_right = crop_x + crop_size
    crop_bottom = crop_y + crop_size
    ix1 = max(x1, crop_x)
    iy1 = max(y1, crop_y)
    ix2 = min(x2, crop_right)
    iy2 = min(y2, crop_bottom)
    if ix1 < ix2 and iy1 < iy2:
        return (ix1, iy1, ix2, iy2)
    else:
        return None

def iou_between_crops(crop1, crop2, crop_size):
    """
    计算两个裁剪区域（正方形，尺寸为 crop_size）的 IoU。
    crop1, crop2: (crop_x, crop_y)
    此函数仅用于相同尺寸区域间的去重判断。
    """
    x1, y1 = crop1
    x2, y2 = crop2
    left1, top1, right1, bottom1 = x1, y1, x1+crop_size, y1+crop_size
    left2, top2, right2, bottom2 = x2, y2, x2+crop_size, y2+crop_size
    inter_left = max(left1, left2)
    inter_top = max(top1, top2)
    inter_right = min(right1, right2)
    inter_bottom = min(bottom1, bottom2)
    if inter_right < inter_left or inter_bottom < inter_top:
        return 0.0
    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area = crop_size * crop_size
    union_area = 2 * area - inter_area
    return inter_area / union_area

def valid_crop_for_targets(crop_x, crop_y, crop_size, bboxes, candidate_idx, margin=15, img_w=None, img_h=None):
    """
    检查候选裁剪区域 (crop_x, crop_y, crop_size) 针对候选目标 candidate_idx 是否满足要求：
    - 候选目标必须被完整包含；对于不在图像边缘的部分，要求与裁剪边界至少留 margin 像素，
      如果目标边界距离图像边缘不足 margin，则该边有效 margin 设为 0。
    - 对于其他目标框，在裁剪区域内有交集的目标，要求交集区域与裁剪边界至少留 margin 像素。
    """
    crop_right = crop_x + crop_size
    crop_bottom = crop_y + crop_size
    target = bboxes[candidate_idx]
    _, tx1, ty1, tx2, ty2 = target
    eff_margin_left = margin if (tx1 >= margin) else 0
    eff_margin_top = margin if (ty1 >= margin) else 0
    eff_margin_right = margin if ((img_w - tx2) >= margin) else 0
    eff_margin_bottom = margin if ((img_h - ty2) >= margin) else 0
    if not (tx1 >= crop_x + eff_margin_left and ty1 >= crop_y + eff_margin_top and
            tx2 <= crop_right - eff_margin_right and ty2 <= crop_bottom - eff_margin_bottom):
        return False
    for i, bbox in enumerate(bboxes):
        if i == candidate_idx:
            continue
        inter = clip_bbox_to_crop(bbox, crop_x, crop_y, crop_size)
        if inter is not None:
            ix1, iy1, ix2, iy2 = inter
            if ix1 < crop_x + margin or iy1 < crop_y + margin or ix2 > crop_right - margin or iy2 > crop_bottom - margin:
                return False
    return True

def generate_candidate_crops_for_target(img_w, img_h, target_box, crop_size, bboxes, candidate_idx, object_num=3, margin=15, max_attempts=4000):
    """
    针对单个目标框生成候选裁剪区域，要求该候选区域完全包含该目标，
    同时对其他目标如果有交集，也要求满足 margin 要求。
    对候选区域进行去重：只有与已采纳区域的 IoU 小于当前阈值时才保留。
    返回候选裁剪区域列表（最多 object_num 个）。
    """
    candidate_crops = []
    threshold = 0.3
    attempts = 0
    while len(candidate_crops) < int(object_num) and attempts < max_attempts:
        crop = generate_crop_for_target(img_w, img_h, target_box, crop_size)
        attempts += 1
        if crop is None:
            continue
        crop_x, crop_y = crop
        if not valid_crop_for_targets(crop_x, crop_y, crop_size, bboxes, candidate_idx, margin, img_w, img_h):
            continue
        # 去重：只有当与已采纳区域的 IoU 小于当前阈值时才保留
        distinct = True
        for prev in candidate_crops:
            if iou_between_crops((crop_x, crop_y), (prev[0], prev[1]), crop_size) > threshold:
                distinct = False
                break
        if distinct:
            candidate_crops.append((crop_x, crop_y, "target", candidate_idx))
        if attempts % 3000 == 0 and threshold < 0.7:
            threshold += 0.1
        # 若候选区域足够则退出
    return candidate_crops

def crop_and_update_labels(image, bboxes, crop_x, crop_y, crop_size, candidate_idx=None, margin=15):
    """
    裁剪图像，并返回裁剪图像及更新后的标签（YOLO 格式归一化）。
    """
    cropped_image = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    new_labels = []
    crop_right = crop_x + crop_size
    crop_bottom = crop_y + crop_size
    for i, bbox in enumerate(bboxes):
        cls, x1, y1, x2, y2 = bbox
        if not (x2 <= crop_x or x1 >= crop_right or y2 <= crop_y or y1 >= crop_bottom):
            if candidate_idx is not None and i == candidate_idx:
                if x1 >= crop_x and y1 >= crop_y and x2 <= crop_right and y2 <= crop_bottom:
                    final_box = (x1, y1, x2, y2)
                else:
                    continue
            else:
                inter = clip_bbox_to_crop(bbox, crop_x, crop_y, crop_size)
                if inter is None:
                    continue
                ix1, iy1, ix2, iy2 = inter
                if ix1 < crop_x + margin or iy1 < crop_y + margin or ix2 > crop_right - margin or iy2 > crop_bottom - margin:
                    continue
                final_box = inter
            final_x1, final_y1, final_x2, final_y2 = final_box
            center_x = (final_x1 + final_x2) / 2 - crop_x
            center_y = (final_y1 + final_y2) / 2 - crop_y
            width = final_x2 - final_x1
            height = final_y2 - final_y1
            normalized = (center_x / crop_size, center_y / crop_size, width / crop_size, height / crop_size)
            new_labels.append((cls, *normalized))
    return cropped_image, new_labels

def main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=3, non_object_num=[5,6], crop_size=672, margin=15):
    random.seed(2000)
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        image_path = os.path.join(image_dir, filename)
        base_name = os.path.splitext(filename)[0]
        label_file = os.path.join(label_dir, base_name + ".txt")
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片 {image_path}")
            continue
        img_h, img_w = image.shape[:2]
        if img_w < crop_size or img_h < crop_size:
            print(f"图片尺寸过小，跳过 {filename}")
            continue

        bboxes = read_yolo_labels(label_file, img_w, img_h)
        crops = []  # 存储裁剪信息: (crop_x, crop_y, crop_type, candidate_idx)

        if bboxes:
            candidate_target_crops = []
            # 针对每个目标框生成候选裁剪（目标裁剪）
            for idx, target in enumerate(bboxes):
                candidate_crops = generate_candidate_crops_for_target(img_w, img_h, target, crop_size, bboxes, idx, object_num, margin, max_attempts=12000)
                if len(candidate_crops) < int(object_num):
                    print(f"目标 {idx} in {base_name} 仅生成 {len(candidate_crops)} 个候选裁剪")
                candidate_target_crops.extend(candidate_crops)
            crops.extend(candidate_target_crops)
            # 生成无目标裁剪
            no_target_crops = []
            attempts = 0
            while attempts < 10000 and len(no_target_crops) < int(non_object_num[0]):
                crop_x = random.randint(0, img_w - crop_size)
                crop_y = random.randint(0, img_h - crop_size)
                conflict = False
                for bbox in bboxes:
                    _, bx1, by1, bx2, by2 = bbox
                    crop_left, crop_top = crop_x, crop_y
                    crop_right, crop_bottom = crop_x + crop_size, crop_y + crop_size
                    inter_x1 = max(crop_left, bx1)
                    inter_y1 = max(crop_top, by1)
                    inter_x2 = min(crop_right, bx2)
                    inter_y2 = min(crop_bottom, by2)
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        conflict = True
                        break
                if not conflict:
                    candidate_crop = (crop_x, crop_y, "no_target", None)
                    distinct = True
                    for prev in no_target_crops:
                        if iou_between_crops((crop_x, crop_y), (prev[0], prev[1]), crop_size) > 0.05:
                            distinct = False
                            break
                    if distinct:
                        no_target_crops.append(candidate_crop)
                attempts += 1
            if len(no_target_crops) < int(non_object_num[0]):
                print(f"{base_name} 无法生成{non_object_num[0]}个无目标裁剪，仅生成 {len(no_target_crops)} 个")
            crops.extend(no_target_crops)
        else:
            # 没有目标框时的无目标裁剪
            no_target_crops = []
            attempts = 0
            while attempts < 10000 and len(no_target_crops) < int(non_object_num[1]):
                crop_x = random.randint(0, img_w - crop_size)
                crop_y = random.randint(0, img_h - crop_size)
                candidate_crop = (crop_x, crop_y, "no_target", None)
                distinct = True
                for prev in no_target_crops:
                    if iou_between_crops((crop_x, crop_y), (prev[0], prev[1]), crop_size) > 0.01:
                        distinct = False
                        break
                if distinct:
                    no_target_crops.append(candidate_crop)
                attempts += 1
            if len(no_target_crops) < int(non_object_num[1]):
                print(f"{base_name} 无法生成{non_object_num[1]}个无目标裁剪，仅生成 {len(no_target_crops)} 个")
            crops.extend(no_target_crops)

        # 记录当前图像已接受的裁剪区域（跨不同尺寸）
        if base_name not in global_accepted_crops:
            global_accepted_crops[base_name] = []

        # 输出统计信息
        target_crops = [crop for crop in crops if crop[2] == "target"]
        no_target = [crop for crop in crops if crop[2] == "no_target"]
        max_iou = 0.0
        if len(target_crops) > 1:
            for i in range(len(target_crops)):
                for j in range(i+1, len(target_crops)):
                    iou_val = iou_between_crops((target_crops[i][0], target_crops[i][1]), (target_crops[j][0], target_crops[j][1]), crop_size)
                    if iou_val > max_iou:
                        max_iou = iou_val
        print(f"{base_name}: 有目标裁剪 {len(target_crops)} 个, 最大 IoU: {max_iou:.2f}, 无目标裁剪 {len(no_target)} 个")

        # 遍历所有候选 crop，进行 IoU 策略选择
        for i, (crop_x, crop_y, crop_type, candidate_idx) in enumerate(crops):
            # 当前候选区域的矩形坐标
            cand_rect = (crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)
            accepted = None

            # 如果没有已接受区域，直接选取当前候选
            if not global_accepted_crops[base_name]:
                accepted = (crop_x, crop_y, crop_type, candidate_idx)
            else:
                # 先直接检查当前候选区域是否满足所有已接受区域 IoU < 0.6
                if all(iou_rect(cand_rect, prev_rect) < 0.6 for prev_rect in global_accepted_crops[base_name]):
                    accepted = (crop_x, crop_y, crop_type, candidate_idx)
                else:
                    # 尝试500次，以阈值0.6随机选取候选区域
                    attempts = 0
                    while attempts < 500:
                        cand_tmp = random.choice(crops)
                        tmp_rect = (cand_tmp[0], cand_tmp[1], cand_tmp[0] + crop_size, cand_tmp[1] + crop_size)
                        if all(iou_rect(tmp_rect, prev_rect) < 0.6 for prev_rect in global_accepted_crops[base_name]):
                            accepted = cand_tmp
                            break
                        attempts += 1
                    # 如果未成功，则尝试500次，阈值放宽到0.8
                    if accepted is None:
                        attempts = 0
                        while attempts < 500:
                            cand_tmp = random.choice(crops)
                            tmp_rect = (cand_tmp[0], cand_tmp[1], cand_tmp[0] + crop_size, cand_tmp[1] + crop_size)
                            if all(iou_rect(tmp_rect, prev_rect) < 0.8 for prev_rect in global_accepted_crops[base_name]):
                                accepted = cand_tmp
                                break
                            attempts += 1
                    # 如果仍未找到，则放弃 IoU 限制，直接接受当前 candidate
                    if accepted is None:
                        accepted = (crop_x, crop_y, crop_type, candidate_idx)

            # 将选定的 accepted 区域记录到全局字典中
            accepted_rect = (accepted[0], accepted[1], accepted[0] + crop_size, accepted[1] + crop_size)
            global_accepted_crops[base_name].append(accepted_rect)

            # 根据 accepted 保存裁剪图像及更新标签
            candidate_sel = accepted[3] if accepted[2] == "target" else None
            cropped_img, new_labels = crop_and_update_labels(image, bboxes, accepted[0], accepted[1], crop_size, candidate_sel, margin)
            out_img_path = os.path.join(out_image_dir, f"{base_name}_whole_crop_{i+1}_{crop_size}.jpg")
            cv2.imwrite(out_img_path, cropped_img)
            out_label_path = os.path.join(out_label_dir, f"{base_name}_whole_crop_{i+1}_{crop_size}.txt")
            with open(out_label_path, "w") as f:
                for label in new_labels:
                    cls, xc, yc, w_norm, h_norm = label
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")

if __name__ == '__main__':
    # # 对验证集
    # image_dir = r"I:\final_canola_dataset_method\final_labeled_dataset\images\val"
    # label_dir = r"I:\final_canola_dataset_method\final_labeled_dataset\labels\val"
    # out_image_dir = r"I:\final_canola_dataset_method\2560_crop_dataset\images\val"
    # out_label_dir = r"I:\final_canola_dataset_method\2560_crop_dataset\labels\val"
    # os.makedirs(out_image_dir, exist_ok=True)
    # os.makedirs(out_label_dir, exist_ok=True)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=1792,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=3328,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=2, non_object_num=[6, 7], crop_size=2560,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=2528,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=2592,
    #      margin=10)
    #
    # image_dir = r"I:\final_canola_dataset_method\final_labeled_dataset\images\train"
    # label_dir = r"I:\final_canola_dataset_method\final_labeled_dataset\labels\train"
    # out_image_dir = r"I:\final_canola_dataset_method\2560_crop_dataset\images\train"
    # out_label_dir = r"I:\final_canola_dataset_method\2560_crop_dataset\labels\train"
    # os.makedirs(out_image_dir, exist_ok=True)
    # os.makedirs(out_label_dir, exist_ok=True)
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=1792,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=3328,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=2, non_object_num=[6, 7], crop_size=2560,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=2528,
    #      margin=10)
    #
    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=1, non_object_num=[4, 5], crop_size=2592,
    #      margin=10)

    image_dir = r"J:\final_canola_dataset_method\final_labeled_dataset\images\val"
    label_dir = r"J:\final_canola_dataset_method\final_labeled_dataset\labels\val"
    out_image_dir = r"J:\final_canola_dataset_method\single_reduced_640_crop_dataset\images\val"
    out_label_dir = r"J:\final_canola_dataset_method\single_reduced_640_crop_dataset\labels\val"
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=2, non_object_num=[1, 2], crop_size=960,
    #      margin=10)

    image_dir = r"J:\final_canola_dataset_method\final_labeled_dataset\images\test"
    label_dir = r"J:\final_canola_dataset_method\final_labeled_dataset\labels\test"
    out_image_dir = r"G:\PhD All\paper\phd\paper 3\heatmap\images\test"
    out_label_dir = r"G:\PhD All\paper\phd\paper 3\heatmap\labels\test"
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)


    main(image_dir, label_dir, out_image_dir, out_label_dir, object_num=2, non_object_num=[1, 2], crop_size=960,
         margin=10)

