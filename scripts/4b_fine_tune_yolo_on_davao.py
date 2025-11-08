from ultralytics import YOLO
import os

def finetune_yolo_on_davao():
    """
    Fine-tunes the PRE-TRAINED yolov8n.pt model (NOT the one
    trained on 'archive') on our new, small Davao training set.
    """
    print("--- Starting YOLOv8 Fine-Tuning on Davao Dataset ---")

    # --- 1. Create a new YAML for this experiment ---
    davao_finetune_yaml = 'yolo_davao_finetune_config.yaml'
    
    # Get the absolute path to the new dataset
    data_root = os.path.abspath(os.path.join('data', 'processed', 'davao_finetune_set'))
    
    yaml_content = f"""
path: {data_root}
train: images/train
val: images/valid

# Class definitions
nc: 1
names: ['occupied']
"""
    with open(davao_finetune_yaml, 'w') as f:
        f.write(yaml_content)
        
    print(f"Created new config file: {davao_finetune_yaml}")

    # --- 2. Load the PRE-TRAINED Model ---
    # IMPORTANT: We start from 'yolov8n.pt', the general model,
    # NOT the 'best.pt' that was ruined by the archive dataset.
    model = YOLO('yolov8n.pt') 
    print("Loaded pre-trained 'yolov8n.pt' model.")

    # --- 3. Start Fine-Tuning ---
    # Since the dataset is tiny (40 images), this will be VERY fast.
    # We can use more epochs because the dataset is small.
    print("Starting fine-tuning...")
    try:
        model.train(
            data=davao_finetune_yaml,
            epochs=50, # 50 epochs on 40 images will be very fast
            batch=4,  # Use a small batch size for a small dataset
            imgsz=640,
            project='runs/detect',
            name='finetune_davao', # Save to a new folder
            exist_ok=True,
            device='cuda' # Explicitly use GPU
        )
        
        print("\n✓ YOLOv8 fine-tuning on Davao set complete.")
        print(f"New model saved in: runs/detect/finetune_davao/weights/best.pt")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during fine-tuning: {e}")


if __name__ == '__main__':
    finetune_yolo_on_davao()