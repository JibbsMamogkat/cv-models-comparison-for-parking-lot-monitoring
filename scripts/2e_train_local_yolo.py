# Subsystem 2: YOLOv8 Training Module

from ultralytics import YOLO
import os
import torch # <<< ADD THIS IMPORT AT THE TOP

def train_yolo():
    """
    Trains a YOLOv8 model using the ultralytics library.
    Handles automatic checkpointing and allows resuming.
    Includes a check to verify GPU usage.
    """
    print("--- Starting YOLOv8 Training ---")

    # --- 1. Define Paths and Parameters ---
    data_yaml_path = 'parking_data.yaml'
    model_name = 'yolov8n.pt'
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    IMG_SIZE = 640

    # --- 2. Load the Model ---
    model = YOLO(model_name)

    # --- 3. Verify Device ---
    # LEARNING POINT: Explicitly check if PyTorch can access the GPU.
    if torch.cuda.is_available():
        device = 'cuda' # Use the first available CUDA device (usually GPU 0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO] CUDA GPU is available!")
        print(f"[INFO] Using device: {device} ({gpu_name})")
    else:
        device = 'cpu'
        print("\n[WARNING] CUDA GPU not found or not configured correctly.")
        print("[WARNING] Training will proceed using the CPU. This will be very slow.")
    print("-" * 20)

    # --- 4. Check for Existing Checkpoints to Resume ---
    runs_dir = 'runs/detect'
    latest_run_path = None
    if os.path.exists(runs_dir):
        train_folders = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith('train')])
        if train_folders:
            latest_run_path = train_folders[-1]
            weights_path = os.path.join(latest_run_path, 'weights', 'last.pt')
            if os.path.exists(weights_path):
                print(f"INFO: Found previous run. Resuming training from: {weights_path}")
                model_name = weights_path
                model = YOLO(model_name)
            else:
                 print("INFO: Previous run found, but no 'last.pt' checkpoint. Starting fresh.")
                 model_name = 'yolov8n.pt'
                 model = YOLO(model_name)
        else:
             print("INFO: No previous training runs found. Starting fresh.")
    else:
        print("INFO: No 'runs' directory found. Starting fresh.")


    # --- 5. Start/Resume Training ---
    # The model.train() function handles everything.
    print(f"Starting training with model: {model_name}")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=NUM_EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            project='runs/detect',
            name='train',
            exist_ok=True,
            resume=os.path.exists(model_name) and model_name.endswith('last.pt'),
            device=device # <<< Explicitly tell Ultralytics which device to use
        )
        print("✓ YOLOv8 training completed successfully.")

    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User (Ctrl+C) ---")
        print("INFO: Training paused. Checkpoints are saved automatically.")
        print(f"To resume later, simply run this script again.")

    except Exception as e:
        print(f"\n--- An error occurred during training ---")
        print(e)
        print("INFO: Checkpoints may have been saved. You might be able to resume.")

if __name__ == '__main__':
    train_yolo()