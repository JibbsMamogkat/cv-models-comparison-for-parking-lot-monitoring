import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def evaluate_classifier(model_name, model_path, data_dir, test_set_name):
    """
    Loads a trained classifier model and evaluates it on a specified test dataset.
    Generates an accuracy score, a classification report, and a confusion matrix.
    
    Args:
        model_name (str): The architecture name (e.g., 'mobilenet_v2' or 'efficientnet_b0').
        model_path (str): Path to the saved .pth model weights file.
        data_dir (str): Path to the test dataset.
        test_set_name (str): A name for the test set for labeling (e.g., 'Public_Academic' or 'Real_World_Davao').
    """
    print(f"\n--- Evaluating Model: {model_name} on {test_set_name} Test Set ---")
    print(f"Loading weights from: {model_path}")
    print(f"Using test data from: {data_dir}")

    # --- 1. Load the Test Data ---
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image_dataset = ImageFolder(data_dir, data_transform)
    except FileNotFoundError:
        print(f"[ERROR] Test data directory not found at: {data_dir}")
        print("Please check the path and try again.")
        return

    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    class_names = image_dataset.classes
    print(f"Classes found in test set: {class_names} (0={class_names[0]}, 1={class_names[1]})")

    # --- 2. Load the Trained Model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = None
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    else:
        print(f"[ERROR] Unknown model name: {model_name}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"[ERROR] Model weights file not found: {model_path}")
        print("Did you train this model and save it to the 'models' folder?")
        return
        
    model = model.to(device)
    model.eval()

    # --- 3. Run Inference (Make Predictions) ---
    all_labels = []
    all_predictions = []

    print(f"Running predictions on the {test_set_name} test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {test_set_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. Calculate and Display Metrics ---
    print(f"\n--- Evaluation Complete for {model_name} on {test_set_name} ---")
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    print(report)

    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({test_set_name})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    chart_save_path = os.path.join('results', 'charts', f'cm_{model_name}_{test_set_name}.png')
    os.makedirs(os.path.dirname(chart_save_path), exist_ok=True)
    plt.savefig(chart_save_path)
    
    print(f"Confusion Matrix plot saved to: {chart_save_path}")
    print("-" * 30)

# --- How to Run This Script ---
if __name__ == '__main__':
    
    # --- 1. Define the paths to your two test sets ---
    
    # Path to the PUBLIC ACADEMIC test set (from your training data)
    public_test_path = os.path.join('data', 'processed', 'classifier', 'test')
    
    # Path to your REAL-WORLD DAVAO test set
    real_world_test_path = os.path.join('data', 'real_world_test_set', 'davao_classifier_test_set')

    # --- 2. Run all evaluations ---

    # --- Test MobileNetV2 ---
    print("="*40)
    print("NOW TESTING: MobileNetV2 on Public Academic Set")
    print("="*40)
    evaluate_classifier(
        model_name='mobilenet_v2',
        model_path=os.path.join('models', 'mobilenet_v2.pth'),
        data_dir=public_test_path,
        test_set_name='Public_Academic'
    )
    
    print("="*40)
    print("NOW TESTING: MobileNetV2 on Real-World Davao Set")
    print("="*40)
    evaluate_classifier(
        model_name='mobilenet_v2',
        model_path=os.path.join('models', 'mobilenet_v2.pth'),
        data_dir=real_world_test_path,
        test_set_name='Real_World_Davao'
    )

    # --- Test EfficientNetB0 ---
    print("="*40)
    print("NOW TESTING: EfficientNetB0 on Public Academic Set")
    print("="*40)
    evaluate_classifier(
        model_name='efficientnet_b0',
        model_path=os.path.join('models', 'efficientnet_b0.pth'),
        data_dir=public_test_path,
        test_set_name='Public_Academic'
    )
    
    print("="*40)
    print("NOW TESTING: EfficientNetB0 on Real-World Davao Set")
    print("="*40)
    evaluate_classifier(
        model_name='efficientnet_b0',
        model_path=os.path.join('models', 'efficientnet_b0.pth'),
        data_dir=real_world_test_path,
        test_set_name='Real_World_Davao'
    )

    print("\nAll classifier evaluations are complete.")

