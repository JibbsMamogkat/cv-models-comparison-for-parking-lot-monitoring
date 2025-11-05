import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def predict_single_patch(model_path, image_path, model_name='mobilenet_v2'):
    """
    Loads a trained classifier model and predicts on a single image patch.

    Args:
        model_path (str): Path to the trained .pth file.
        image_path (str): Path to the single cropped parking spot image.
        model_name (str): The architecture ('mobilenet_v2' or 'efficientnet_b0').
    """
    print(f"--- Running Classifier Prediction ---")
    print(f"Model Type: {model_name}")
    print(f"Model Weights: {model_path}")
    print(f"Input Image: {image_path}")

    # --- 1. Load Model Architecture ---
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2() # Load the base architecture
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2) # Adjust final layer for 2 classes
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0() # Load the base architecture
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2) # Adjust final layer for 2 classes
    else:
        print(f"Error: Unknown model name '{model_name}'. Choose 'mobilenet_v2' or 'efficientnet_b0'.")
        return

    # --- 2. Load Trained Weights ---
    try:
        # Load the saved weights from your training run
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load to CPU for prediction
        model.eval() # Set the model to evaluation mode (important!)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at '{model_path}'.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # --- 3. Define Image Transformations ---
    # IMPORTANT: Use the EXACT same transformations as during training!
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 4. Load and Preprocess the Image Patch ---
    try:
        img = Image.open(image_path).convert('RGB') # Load image
        img_t = preprocess(img) # Apply transformations
        batch_t = torch.unsqueeze(img_t, 0) # Create a batch of 1 image
    except FileNotFoundError:
        print(f"Error: Input image file not found at '{image_path}'.")
        return
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # --- 5. Make Prediction ---
    with torch.no_grad(): # Turn off gradient calculations for inference
        outputs = model(batch_t)
        _, predicted_idx = torch.max(outputs, 1)

    # --- 6. Interpret the Result ---
    # Assuming your training data used folders 'occupied', 'vacant'
    # Check the actual class order from your training run if needed
    class_names = ['occupied', 'vacant'] 
    prediction = class_names[predicted_idx.item()]

    print(f"\nPrediction Complete:")
    print(f"The model predicts this spot is: {prediction.upper()}")


# --- How to Use ---
if __name__ == '__main__':
    # --- Configure these paths ---

    # Choose which model to test: 'mobilenet_v2' or 'efficientnet_b0'
    MODEL_TYPE_TO_TEST = 'mobilenet_v2' #<--- CHANGE THIS

    # Path to the corresponding trained model weights (.pth file)
    # Ensure this points to the correct file in your 'models' folder
    PATH_TO_WEIGHTS = f'models/mobilenet_v2.pth' 

    # Path to the SINGLE cropped image patch you want to test
    # Use the patches you created manually (e.g., 'campus_occupied_spot.jpg')
    PATH_TO_PATCH = 'data/real_world_test_set/classifier-earlier-pics/occupied-mcm-field1.jpg' #<--- CHANGE THIS

    # --- Run the prediction ---
    if not os.path.exists(PATH_TO_PATCH):
         print(f"[SETUP ERROR] Image patch not found at '{PATH_TO_PATCH}'")
         print("Please create the patch image first and update the PATH_TO_PATCH variable.")
    elif not os.path.exists(PATH_TO_WEIGHTS):
         print(f"[SETUP ERROR] Model weights not found at '{PATH_TO_WEIGHTS}'")
         print("Please ensure the model has been trained and the path is correct.")
    else:
        predict_single_patch(PATH_TO_WEIGHTS, PATH_TO_PATCH, MODEL_TYPE_TO_TEST)