import os
from ultralytics import YOLO

def evaluate_yolo_on_davao_set():
    """
    Loads the locally trained YOLOv8 model and evaluates it 
    on the custom Davao test set (now in YOLO .txt format).
    """
    print("--- Starting YOLOv8 Evaluation on Real-World Test Set ---")

    # --- 1. Define Paths ---
    
    # Path to your best trained model.
    # !! IMPORTANT: Check 'runs/detect/' folder and update 'train' if needed (e.g., 'train2').
    model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')

    # Path to the root of your test set.
    test_set_root = os.path.join('data', 'real_world_test_set', 'davao_detector_test_set')
    
    # Define a name for the new YAML config file we will create.
    yaml_config_path = 'yolo_davao_test_config.yaml'

    # --- 2. Check if Model File Exists ---
    if not os.path.exists(model_path):
        print(f"[FATAL ERROR] Model file not found at: {os.path.abspath(model_path)}")
        print("Please check the path to your 'best.pt' file.")
        return
        
    # Check if the 'labels' folder exists, which is created by the 3b script
    if not os.path.exists(os.path.join(test_set_root, 'labels')):
        print(f"[FATAL ERROR] 'labels' folder not found in '{test_set_root}'")
        print("Please run 'scripts/3b_prep_yolo_test_set.py' first to create the .txt label files.")
        return

    # --- 3. Create the YAML Configuration File ---
    # This file tells the YOLO library where to find your test data.
    # It now points to the 'images' and 'labels' folders as intended.
    
    print(f"Creating test config file: {yaml_config_path}")
    
    # We use os.path.abspath to give YOLO unambiguous paths
    yaml_content = f"""
path: {os.path.abspath(test_set_root)}
train: images  # 'train:' key is required, we can just point it to 'images'
val: images    # 'val:' is what we will use for our test split.
test: images   # Also add a 'test:' key for completeness

# Class definitions
nc: 1
names: ['occupied']
"""
    
    with open(yaml_config_path, 'w') as f:
        f.write(yaml_content)

    # --- 4. Load Model and Run Evaluation ---
    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)

    print("Starting evaluation...")
    
    # We use model.val() and point it to our new config file.
    # The 'split='val'' tells it to use the 'val' path from the YAML.
    # It will now automatically find the .txt files in the 'labels' folder.
    try:
        results = model.val(
            data=yaml_config_path,
            split='val', # We will use the 'val' split for this.
            batch=16 
        )

        print("\n--- Evaluation Complete ---")
        print("Metrics for your YOLOv8 model on the Davao Real-World Test Set:")
        
        # This is the corrected way to access the metrics
        print(f"  mAP50-95 (mean Average Precision @ .50-.95): {results.box.map:.4f}")
        print(f"  mAP50 (mean Average Precision @ .50):      {results.box.map50:.4f}")
        print(f"  Precision:                                {results.box.p[0]:.4f}")
        print(f"  Recall:                                   {results.box.r[0]:.4f}")
        
        print(f"\nResults saved to: {results.save_dir}")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during evaluation: {e}")
        print("Please ensure the 'labels' folder is populated with .txt files.")

if __name__ == '__main__':
    evaluate_yolo_on_davao_set()

