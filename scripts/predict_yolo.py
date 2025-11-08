from ultralytics import YOLO
import os
import cv2 # OpenCV for image handling

def predict_on_image(model_path, image_path, output_dir='runs/detect/predict'):
    """
    Runs YOLOv8 inference on a single image and saves the result.

    Args:
        model_path (str): Path to the trained model weights file (e.g., 'runs/detect/train/weights/best.pt').
        image_path (str): Path to the image file you want to predict on.
        output_dir (str): Directory where the output image with detections will be saved.
    """
    print(f"--- Running Prediction ---")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")

    # --- 1. Load the Trained Model ---
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Run Prediction ---
    try:
        # The 'predict' method handles everything.
        # Setting save=True automatically saves the output image.
        # You can customize confidence threshold, etc. here. See Ultralytics docs.
        results = model.predict(
            source=image_path, 
            save=True, 
            project=output_dir, 
            name='predict_results', 
            exist_ok=True,
            conf=0.10 # <--- ADD THIS: Set a lower threshold (e.g., 0.10 for 10%)
        )        
        # The results object contains detailed information, but save=True handles the visual output.
        # We find the path where the image was saved.
        saved_image_path = os.path.join(output_dir, 'predict_results', os.path.basename(image_path)) # Ultralytics saves in a subfolder
        if os.path.exists(saved_image_path):
             print(f"✓ Prediction successful!")
             print(f"Output image saved to: {saved_image_path}")
        else:
             # Fallback path if using older ultralytics version structure
             saved_image_path = os.path.join(output_dir, os.path.basename(image_path)) 
             if os.path.exists(saved_image_path):
                 print(f"✓ Prediction successful!")
                 print(f"Output image saved to: {saved_image_path}")
             else:
                 print(f"[WARNING] Prediction ran, but couldn't find the saved image output.")
                 print(f"Check the '{output_dir}' directory.")


    except Exception as e:
        print(f"An error occurred during prediction: {e}")

# --- 3. How to Use ---
if __name__ == '__main__':
    # --- IMPORTANT: Set these paths correctly! ---
    
    # Path to your trained model weights (use 'best.pt' for evaluation)
    path_to_weights = 'yolov8n.pt' # CHANGE 'train' if you have train2, train3 etc.
    
    # Path to the single image you want to test
    path_to_sample_image = 'data/real_world_test_set/earlier-pics/mcm-faculty1.jpg' # CHANGE this to an actual image filename

    # --- Optional: Output directory ---
    # Where the result image will be saved
    output_directory = 'runs/detect/predict_output' 

    # --- Run the prediction function ---
    predict_on_image(path_to_weights, path_to_sample_image, output_directory)