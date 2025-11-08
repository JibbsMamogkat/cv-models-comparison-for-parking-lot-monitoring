from ultralytics import YOLO
import os

def evaluate_finetuned_yolo():
    """
    Evaluates our NEWLY fine-tuned model on the 10-image
    Davao validation set.
    """
    print("--- Evaluating Fine-Tuned YOLOv8 Model ---")

    # --- 1. Define Paths ---
    
    # Path to our NEWLY trained model
    model_path = os.path.join('runs', 'detect', 'finetune_davao', 'weights', 'best.pt')

    # Path to the YAML file we created in the fine-tuning script
    yaml_config_path = 'yolo_davao_finetune_config.yaml'

    if not os.path.exists(model_path):
        print(f"[ERROR] Fine-tuned model not found at: {model_path}")
        print("Please run '2f_finetune_yolo.py' first.")
        return

    if not os.path.exists(yaml_config_path):
        print(f"[ERROR] Config file not found: {yaml_config_path}")
        print("Please run '2f_finetune_yolo.py' first.")
        return

    # --- 2. Load Model and Run Evaluation ---
    print(f"Loading fine-tuned model from: {model_path}")
    model = YOLO(model_path)

    print("Starting evaluation on Davao validation set...")
    
    results = model.val(
        data=yaml_config_path,
        split='val' # This tells it to use the 'val' set (our 10 images)
    )

    print("\n--- Evaluation Complete ---")
    print("Metrics for FINE-TUNED YOLOv8 on Davao Set:")
    
    print(f"  mAP50-95 (mean Average Precision @ .50-.95): {results.box.map:.4f}")
    print(f"  mAP50 (mean Average Precision @ .50):      {results.box.map50:.4f}")
    print(f"  Precision:                                {results.box.p[0]:.4f}")
    print(f"  Recall:                                   {results.box.r[0]:.4f}")
    
    print(f"\nResults saved to: {results.save_dir}")

if __name__ == '__main__':
    evaluate_finetuned_yolo()