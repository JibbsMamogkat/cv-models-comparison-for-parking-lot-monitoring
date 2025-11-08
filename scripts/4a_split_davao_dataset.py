import os
import shutil
import random

def split_davao_dataset():
    """
    Splits the 50 annotated Davao images into a 'finetune_train' 
    and 'finetune_valid' set.
    
    It will create a new directory structure for this experiment.
    """
    print("--- Creating new Train/Valid split from Davao data ---")
    
    # --- 1. Define Paths ---
    source_root = os.path.join('data', 'real_world_test_set', 'davao_detector_test_set')
    source_images_dir = os.path.join(source_root, 'images')
    source_labels_dir = os.path.join(source_root, 'labels') # The .txt folder we created
    
    # Define the new directory for this experiment
    output_root = os.path.join('data', 'processed', 'davao_finetune_set')
    
    # Define the new train/valid paths
    train_img_path = os.path.join(output_root, 'images', 'train')
    train_lbl_path = os.path.join(output_root, 'labels', 'train')
    valid_img_path = os.path.join(output_root, 'images', 'valid')
    valid_lbl_path = os.path.join(output_root, 'labels', 'valid')
    
    # --- 2. Create New Directories ---
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(valid_img_path, exist_ok=True)
    os.makedirs(valid_lbl_path, exist_ok=True)
    
    # --- 3. Get All Image Files ---
    try:
        all_images = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]
        random.shuffle(all_images) # Shuffle them to get a random split
    except FileNotFoundError:
        print(f"[ERROR] Source images not found at: {source_images_dir}")
        return

    # --- 4. Define Split (e.g., 80% train, 20% valid) ---
    split_index = int(len(all_images) * 0.80)
    train_files = all_images[:split_index]
    valid_files = all_images[split_index:]
    
    print(f"Total images: {len(all_images)}")
    print(f"Splitting into {len(train_files)} training images and {len(valid_files)} validation images.")
    
    # --- 5. Copy Files to New Locations ---
    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        copied_labels = 0
        for img_filename in file_list:
            base_name = os.path.splitext(img_filename)[0]
            lbl_filename = base_name + '.txt'
            
            src_img = os.path.join(source_images_dir, img_filename)
            src_lbl = os.path.join(source_labels_dir, lbl_filename)
            
            dest_img = os.path.join(dest_img_dir, img_filename)
            dest_lbl = os.path.join(dest_lbl_dir, lbl_filename)
            
            # Copy image
            if os.path.exists(src_img):
                shutil.copy(src_img, dest_img)
            
            # Copy label (if it exists)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dest_lbl)
                copied_labels += 1
            else:
                # If no label file, create an empty one (for background images)
                open(dest_lbl, 'w').close()

        return copied_labels

    train_labels = copy_files(train_files, train_img_path, train_lbl_path)
    valid_labels = copy_files(valid_files, valid_img_path, valid_lbl_path)

    print(f"Copied {len(train_files)} train images ({train_labels} with labels).")
    print(f"Copied {len(valid_files)} valid images ({valid_labels} with labels).")
    print(f"\n✓ New fine-tuning dataset created at: {output_root}")
    
    return output_root

if __name__ == '__main__':
    split_davao_dataset()