# Subsystem 2: MobileNetV2 Training Module

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

def train_mobilenet():
    """
    Trains a MobileNetV2 model for binary classification (occupied vs. vacant).
    This function demonstrates a standard PyTorch training loop.
    """
    print("--- Starting MobileNetV2 Training ---")

    # --- 1. Define Paths and Parameters ---
    data_dir = os.path.join('data', 'processed', 'classifier')
    model_save_path = os.path.join('models', 'mobilenet_v2.pth')
    
    # Hyperparameters: These are the settings for our training process.
    NUM_EPOCHS = 10  # An epoch is one full pass through the entire training dataset.
    BATCH_SIZE = 32  # Process images in batches of 32 to fit in memory.
    LEARNING_RATE = 0.001

    # --- 2. Data Transformation and Loading (The "ETL" for the model) ---
    # LEARNING POINT: We must prepare our images before feeding them to the model.
    # a) Resize them all to the same size (224x224 pixels).
    # b) Convert them to Tensors (the data format PyTorch uses).
    # c) Normalize them (set pixel values to a standard range, which helps training).
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

    # LEARNING POINT: PyTorch's `ImageFolder` is a powerful tool.
    # It automatically finds our images and uses the folder names ('occupied', 'vacant') as labels.
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    # The DataLoader takes our dataset and serves it to the model in shuffled batches.
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Training set size: {dataset_sizes['train']} images")
    print(f"Test set size: {dataset_sizes['test']} images")


    # --- 3. Model Definition (Transfer Learning) ---
    # Check if a GPU is available, otherwise use CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # LEARNING POINT: This is Transfer Learning.
    # We load MobileNetV2 that was pre-trained on millions of images (ImageNet).
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # We then replace its final layer (the "classifier") with a new one
    # that is designed for our specific problem (2 outputs: occupied, vacant).
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

    # Move the model to the GPU if available.
    model = model.to(device)

    # Define the loss function (how we measure error) and the optimizer (how we update the model).
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


    # --- 4. The Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and a validation phase.
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using the dataloader.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients.
                optimizer.zero_grad()

                # Forward pass: make a prediction.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
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
    print("✓ MobileNetV2 training is complete.")

if __name__ == '__main__':
    train_mobilenet()

# ### **How to Run This**

# 1.  **Read the Code:** Take 10-15 minutes to read through the script and especially the comments I've labeled with **"LEARNING POINT."** This is where the key concepts are explained.
# 2.  **Run from Terminal:** Open your VS Code terminal (Git Bash), activate your virtual environment, and run the script:
#     ```bash
#     python scripts/2c_train_mobilenet.py
    
