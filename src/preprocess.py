import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Paths
RAW_DIR = "data/raw/"  # Path to raw dataset
PROCESSED_DIR = "data/processed/"  # Base directory for saving processed subsets
IMG_SIZE = (128, 128)  # Resize images to 128x128
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # Train, Validation, Test split ratios

# Preprocessing transformations
save_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize the image
    transforms.ToTensor(),  # Convert the image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])


# Function to save images to specific folder
# Function to save images to specific folder
def save_subset(data_subset, subset_dir, transform):
    """
    Save a subset of images (train/val/test) into a specific directory.

    Args:
        data_subset (list): List of (image path, label) tuples.
        subset_dir (str): Directory to save the processed subset.
        transform: Transformations to apply to each image.
    """
    for img_path, label in data_subset:
        # Open image
        img = Image.open(img_path)

        # Convert all images to RGB (eliminate alpha channel if present)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply preprocessing transformations
        img = transform(img)

        # Convert tensor to PIL image for saving
        img_to_save = transforms.ToPILImage()(img)  # Undo tensor transform for saving

        # Get class name based on label
        class_name = dataset.classes[label]
        class_dir = os.path.join(subset_dir, class_name)  # Output directory for the class
        os.makedirs(class_dir, exist_ok=True)  # Create class directory if it doesn't exist

        # Save the processed image
        filename = os.path.basename(img_path)  # Keep original filename
        output_path = os.path.join(class_dir, filename)
        img_to_save.save(output_path)
        print(f"Saved: {output_path}")


# Main function to preprocess and save subsets
if __name__ == "__main__":
    # Load raw dataset
    dataset = ImageFolder(RAW_DIR)
    print(f"Found {len(dataset)} images in {RAW_DIR}")

    # Split dataset into train, val, and test subsets
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size

    # Retrieve file paths and labels from the dataset
    file_paths, labels = zip(*dataset.imgs)

    # Split raw dataset paths into train, val, and test subsets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=(val_size + test_size), stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size, stratify=temp_labels, random_state=42
    )

    # Create full subsets with (path, label) pairs
    train_subset = list(zip(train_paths, train_labels))
    val_subset = list(zip(val_paths, val_labels))
    test_subset = list(zip(test_paths, test_labels))

    # Save subsets into processed directories
    print("\nSaving training data...")
    save_subset(train_subset, os.path.join(PROCESSED_DIR, "train"), save_transform)
    print("\nSaving validation data...")
    save_subset(val_subset, os.path.join(PROCESSED_DIR, "val"), save_transform)
    print("\nSaving testing data...")
    save_subset(test_subset, os.path.join(PROCESSED_DIR, "test"), save_transform)

    print(f"\nPreprocessed images saved to: {PROCESSED_DIR}")
