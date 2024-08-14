import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
import os
from torch.utils.tensorboard import SummaryWriter
import logging
from util.crop import RandomResizedCrop

# Setup TensorBoard
log_dir = 'runs/experiment_1'  # Directory for TensorBoard logs
writer = SummaryWriter(log_dir=log_dir)

# Setup logging to console and file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class ResNet18For5Classes(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18For5Classes, self).__init__()
        # Load the pre-trained ResNet-18 model
        self.resnet18 = models.resnet18(weights=None)
        
        # Modify the final fully connected layer to output `num_classes` classes
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        logger.info(f'Validation Loss: {val_loss:.4f}')
        logger.info(f'Validation Accuracy: {val_accuracy:.2f}%')

def evaluate_model(model, val_loader, epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total * 100
    
    return epoch_loss, accuracy

def main():
    # Define transformations for data augmentation and normalization
    transform_train = transforms.Compose([
            RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Define paths to your dataset
    train_data_path = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/linprobe/train'
    val_data_path = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/linprobe/val'

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        logger.error("The specified dataset paths do not exist.")
        return

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Instantiate the model
    model = ResNet18For5Classes(num_classes=5)
    
    # Train and evaluate the model
    train_model(model, train_loader, val_loader, num_epochs=90, learning_rate=0.001)

    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
