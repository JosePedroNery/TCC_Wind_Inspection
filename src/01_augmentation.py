import os
import random
import shutil
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_GENERATION_COUNT = 1220 
NEGATIVE_BACKGROUNDS_COUNT = 50 

# Random Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- PATHS (RELATIVE & DYNAMIC) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SOURCE_SYNTH_DIR = DATA_DIR / "raw_sintetico"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"     
OUTPUT_DIR = DATA_DIR / "synthetic_generated_pool"

# --- USER DEFINED PIPELINES (STRATEGY 5) ---

# 1. Pipeline to apply to the object BEFORE pasting
# (Simpler geometry transforms)
transform_object = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.8, border_mode=cv2.BORDER_CONSTANT, border_value=0),
])

# 2. Pipeline to apply to the FINAL image (Background + Pasted Object)
# STRATEGY 5: Blur and ISO Noise to bridge the Sim2Real gap
transform_final = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.2))

# --- HELPER FUNCTIONS ---

def read_yolo_label(label_path):
    bboxes = []
    classes = []
    if not label_path.exists():
        return [], []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls = int(parts[0])
                if cls == 0: # Assuming class 0 is the target
                    x, y, w, h = map(float, parts[1:5])
                    bboxes.append([x, y, w, h])
                    classes.append(cls)
            except ValueError:
                continue
    return bboxes, classes

def save_yolo_label(save_path, bboxes, classes):
    with open(save_path, 'w') as f:
        for cls, bbox in zip(classes, bboxes):
            x, y, w, h = [max(0.0, min(1.0, v)) for v in bbox]
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def yolo_to_pixel(bboxes, img_w, img_h):
    pixel_boxes = []
    for x, y, w, h in bboxes:
        x1 = int((x - w / 2) * img_w)
        y1 = int((y - h / 2) * img_h)
        x2 = int((x + w / 2) * img_w)
        y2 = int((y + h / 2) * img_h)
        pixel_boxes.append([x1, y1, x2, y2])
    return pixel_boxes

def pixel_to_yolo(bboxes, img_w, img_h):
    yolo_boxes = []
    for x1, y1, x2, y2 in bboxes:
        x_c = ((x1 + x2) / 2) / img_w
        y_c = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        yolo_boxes.append([x_c, y_c, w, h])
    return yolo_boxes

# --- MAIN EXECUTION ---

def main():
    print("🚀 Starting Synthetic Data Factory (Strategy 5: Blur/ISO)...")
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "images").mkdir(parents=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True)
    
    print("   > Indexing donor images...")
    donor_images = list(SOURCE_SYNTH_DIR.rglob("*.jpg")) + list(SOURCE_SYNTH_DIR.rglob("*.png")) + list(SOURCE_SYNTH_DIR.rglob("*.jpeg"))
    
    valid_donors = []
    for img_path in donor_images:
        label_path = None
        possibles = [
            img_path.with_suffix('.txt'),
            img_path.parents[1] / "labels" / f"{img_path.stem}.txt",
            img_path.parent / "labels" / f"{img_path.stem}.txt"
        ]
        for p in possibles:
            if p.exists():
                label_path = p
                break

        if label_path:
            _, classes = read_yolo_label(label_path)
            if 0 in classes:
                valid_donors.append((img_path, label_path))
    
    print(f"   > Donors: {len(valid_donors)}")
    
    backgrounds = list(BACKGROUNDS_DIR.glob("*.jpg")) + list(BACKGROUNDS_DIR.glob("*.png")) + list(BACKGROUNDS_DIR.glob("*.jpeg"))
    print(f"   > Backgrounds: {len(backgrounds)}")
    
    if not valid_donors or not backgrounds:
        print("❌ Error: Missing inputs.")
        return

    print(f"   > Generating {TARGET_GENERATION_COUNT} samples...")
    generated_count = 0
    pbar = tqdm(total=TARGET_GENERATION_COUNT)
    
    while generated_count < TARGET_GENERATION_COUNT:
        donor_img_path, donor_lbl_path = random.choice(valid_donors)
        bg_path = random.choice(backgrounds)
        
        try:
            img_donor = cv2.cvtColor(cv2.imread(str(donor_img_path)), cv2.COLOR_BGR2RGB)
            img_bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
        except: continue

        h_bg, w_bg, _ = img_bg.shape
        yolo_bboxes, _ = read_yolo_label(donor_lbl_path)
        pixel_bboxes = yolo_to_pixel(yolo_bboxes, img_donor.shape[1], img_donor.shape[0])
        
        if not pixel_bboxes: continue
        x1, y1, x2, y2 = random.choice(pixel_bboxes)
        if x2 <= x1 or y2 <= y1: continue
        
        object_crop = img_donor[y1:y2, x1:x2]
        if object_crop.size == 0: continue

        # --- 1. Apply Object Transform (Geometry) ---
        try:
            transformed = transform_object(image=object_crop)
            object_aug = transformed['image']
        except: continue
            
        h_obj, w_obj, _ = object_aug.shape
        
        if h_obj >= h_bg or w_obj >= w_bg:
            scale = min(h_bg/h_obj, w_bg/w_obj) * 0.8
            if scale <= 0: continue
            object_aug = cv2.resize(object_aug, (0,0), fx=scale, fy=scale)
            h_obj, w_obj, _ = object_aug.shape
            
        try:
            x_offset = random.randint(0, w_bg - w_obj)
            y_offset = random.randint(0, h_bg - h_obj)
        except ValueError: continue
        
        img_final = img_bg.copy()
        img_final[y_offset:y_offset+h_obj, x_offset:x_offset+w_obj] = object_aug
        
        new_bbox_pixel = [[x_offset, y_offset, x_offset+w_obj, y_offset+h_obj]]
        new_bbox_yolo = pixel_to_yolo(new_bbox_pixel, w_bg, h_bg)
        new_classes = [0] 
        
        # --- 2. Apply Final Transform (Blur/Noise Strategy) ---
        try:
            # NOTICE: We use 'category_ids' here because your pipeline defined it that way
            final_aug = transform_final(image=img_final, bboxes=new_bbox_yolo, category_ids=new_classes)
            
            img_to_save = final_aug['image']
            bboxes_to_save = final_aug['bboxes']
            classes_to_save = final_aug['category_ids'] # Retrieve from the correct key
            
            if len(bboxes_to_save) > 0:
                filename = f"synth_gen_{generated_count:05d}"
                cv2.imwrite(str(OUTPUT_DIR / "images" / f"{filename}.jpg"), cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                save_yolo_label(OUTPUT_DIR / "labels" / f"{filename}.txt", bboxes_to_save, classes_to_save)
                
                generated_count += 1
                pbar.update(1)
                
        except Exception as e:
            continue

    pbar.close()

    print(f"   > Injecting {NEGATIVE_BACKGROUNDS_COUNT} negatives...")
    for i in range(NEGATIVE_BACKGROUNDS_COUNT):
        bg_path = random.choice(backgrounds)
        filename = f"synth_negative_{i:03d}"
        shutil.copy(bg_path, OUTPUT_DIR / "images" / f"{filename}.jpg")
        with open(OUTPUT_DIR / "labels" / f"{filename}.txt", 'w') as f:
            pass

    print(f"✅ DONE! Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()