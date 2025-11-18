import os 
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# DataLoader and Dataset (Clean Samples)
def data_loader(root_dir, image_size=(224, 224), batch_size=30):
    """
    Class to create Dataset and DataLoader from Image folder for lung dataset. 
    Args: 
        image_size -> size of the image after resize 
        batch_size 
        root_dir -> root directory of the dataset (your lung dataset) 

    return: 
        dataloader -> dict includes dataloader for test only (no train/val needed for evaluation)
        dataset -> dict includes dataset for test only
    """
    # Remove the hard-coded path - use the passed root_dir parameter
    # root_dir = "../XAI_VIT-main/data/lung/Test"  # Remove this line
    
    # For your lung dataset, you only have Test folder
    # Use the same transforms for test (no augmentation for evaluation)
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # Use your dataset's normalization values
        transforms.Normalize(mean=[0.58, 0.58, 0.58], std=[0.21, 0.21, 0.21])
    ])

    # Only test dataset needed for evaluation
    image_dataset = ImageFolder(root_dir, transform=data_transform)
    
    # Create test dataloader
    data_loaders = {}
    data_loaders['test'] = DataLoader(image_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0, drop_last=False)

    dataset_size = len(image_dataset)
    print(f'Number of test images: {dataset_size}')

    class_idx = image_dataset.class_to_idx
    print(f'Classes with index: {class_idx}')

    class_names = image_dataset.classes
    print(f'Class names: {class_names}')
    
    # Return as dictionary for compatibility
    return {'test': data_loaders['test']}, {'test': image_dataset}

# Dataloader and Dataset (Adversarial Samples)
def data_loader_attacks(root_dir, attack_name, image_size=(224, 224), batch_size=30): 
    """
    Class to create Dataset and DataLoader from Image folder for adversarial samples generated. 
    Args: 
        root_dir: root directory of generated adversarial samples.
        attack_name: attack name that has folder in root_dir.
        image_size: size of the image after resize (224,224)
        batch_size

    return: 
        dataloader: dataloader for the attack
        dataset: dataset for attack 
    """
    
    dirs = os.path.join(root_dir, f'Test_attacks_{attack_name}')
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # Use same normalization as clean data
        transforms.Normalize(mean=[0.58, 0.58, 0.58], std=[0.21, 0.21, 0.21])
    ])
    
    image_dataset = ImageFolder(dirs, transform=data_transform)
    data_loaders = DataLoader(image_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

    print(f'Number of adversarial images: {len(image_dataset)}')

    class_idx = image_dataset.class_to_idx
    print(f'Classes with index: {class_idx}')

    return data_loaders, image_dataset