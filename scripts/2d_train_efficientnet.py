# Subsystem 2: EfficientNetB0 Training Module
# NOTE: This script is 95% identical to the MobileNetV2 script.
# This demonstrates the power of a reusable training pipeline.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

def train_efficientnet():
    """
    Trains an EfficientNet-B0 model for binary classification (occupied vs. vacant).
    """
    print("--- Starting EfficientNet-B0 Training ---")

    # --- 1. Define Paths and Parameters ---
    # These paths are the same as before.
    data_dir = os.path.join('data', 'processed', 'classifier')
    
    # --- CHANGE #1: New model save path ---
    # We give our new model its own unique filename.
    model_save_path = os.path.join('models', 'efficientnet_b0.pth')
    
    # Hyperparameters are the same.
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # --- 2. Data Transformation and Loading ---
    # This entire section is IDENTICAL to the MobileNetV2 script. No changes needed.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Training set size: {dataset_sizes['train']} images")
    print(f"Test set size: {dataset_sizes['test']} images")

    # --- 3. Model Definition (Transfer Learning) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- CHANGE #2: Load the EfficientNet-B0 model ---
    # Instead of MobileNet, we now load the pre-trained EfficientNet-B0.
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # --- CHANGE #3: Modify the final layer for EfficientNet ---
    # The structure of EfficientNet's final layer is slightly different from MobileNet's.
    # We find the number of input features and replace the final layer with our new one.
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # --- 4. The Training Loop ---
    # This entire loop is IDENTICAL to the MobileNetV2 script. No changes needed.
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # --- 5. Save the Trained Model ---
    print("\nTraining complete. Saving model...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'")
    print("✓ EfficientNet-B0 training is complete.")

if __name__ == '__main__':
    train_efficientnet()
# ```

### **Your Workflow (The Hybrid Approach)**

# Follow the same successful pattern you used for the first model:

# 1.  **Local Test (Pre-flight Check):**
#     * In this new script, temporarily change `data_dir` to point to your small `classifier_sample` folder.
#     * Run `python scripts/2d_train_efficientnet.py` on your local machine.
#     * The script should finish in 1-2 minutes and save a new `efficientnet_b0.pth` file. This proves your code is correct.

# 2.  **Full Training (On Google Colab):**
#     * Revert the `data_dir` change in the script.
#     * In your Colab notebook, create a new code cell.
#     * Paste the final, correct code into it.
#     * **Crucially, update the paths** in the script to point to your data on Colab and the save location in your Google Drive, just like you did before:
#         ```python
#         data_dir = '/content/data/processed/classifier'
#         model_save_path = '/content/drive/MyDrive/path/to/your/project/models/efficientnet_b0.pth'
        
