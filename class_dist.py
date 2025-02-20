import os
from collections import Counter
import matplotlib.pyplot as plt


def count_images_in_folders(root_dir):
    """
    Count the number of images in each class within the dataset splits (train, val, test).
    """
    dataset_counts = {}

    for split in ["train", "val", "test"]:  # Check for each dataset split
        split_path = os.path.join(root_dir, "data", "processed", split)
        class_counts = {}

        if os.path.exists(split_path):  # Ensure the split exists
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    class_counts[class_name] = len(os.listdir(class_path))

        dataset_counts[split] = class_counts

    return dataset_counts


# Set dataset root directory
root_dir = os.getcwd()  # Assuming script runs from project root

# Get counts
dataset_counts = count_images_in_folders(root_dir)

# Display the counts
for split, counts in dataset_counts.items():
    print(f"--- {split.upper()} SET ---")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")
    print("\n")

# Plot class distributions
for split, counts in dataset_counts.items():
    if counts:  # Ensure data exists for the split
        plt.figure(figsize=(8, 4))
        labels, values = zip(*counts.items())
        plt.bar(labels, values)
        plt.xlabel("Classes")
        plt.ylabel("Number of Images")
        plt.title(f"{split.capitalize()} Set Class Distribution")
        plt.xticks(rotation=45)
        plt.show()