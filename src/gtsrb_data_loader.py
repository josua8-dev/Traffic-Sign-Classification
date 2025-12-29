"""
GTSRB Data Loader - Fresh implementation for PNG images
Handles the full German Traffic Sign Recognition Benchmark dataset
"""

import numpy as np
import cv2
import os
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class GTSRBLoader:
    """
    Load and preprocess GTSRB dataset
    
    Expected structure: 
    GTSRB/
    ├── Train/
    │   ├── 0/
    │   │   ├── 00000.png
    │   │   ├── 00001.png
    │   │   └── ...
    │   ├── 1/
    │   └── ...  (up to 42)
    └── Test/  (optional)
    """
    
    def __init__(self, data_dir='GTSRB', img_size=(64, 64)):
        """
        Initialize the loader
        
        Args:
            data_dir:   Path to GTSRB folder
            img_size: Target size for all images (height, width)
        """
        self. data_dir = data_dir
        self.img_size = img_size
        self.train_dir = os.path.join(data_dir, 'Train')
        
        print(f"\n🚦 GTSRB Data Loader")
        print(f"   Data directory: {os.path.abspath(data_dir)}")
        print(f"   Target image size: {img_size}")
    
    def load_images_from_folder(self, folder_path, class_id):
        """
        Load all PNG images from a class folder
        
        Args:  
            folder_path: Path to class folder
            class_id:  Numeric class label
            
        Returns:
            List of images and labels
        """
        images = []
        labels = []
        
        folder = Path(folder_path)
        if not folder.exists():
            return images, labels
        
        # Use iterdir and filter for . png files
        image_files = [f for f in folder.iterdir() if f.suffix. lower() == '.png']
        
        for img_file in image_files: 
            try:
                # Read image
                img = cv2.imread(str(img_file))
                
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
                
                images.append(img)
                labels.append(class_id)
                
            except Exception as e: 
                print(f"   ⚠️  Error loading {img_file. name}: {e}")
                continue
        
        return images, labels
    
    def load_all_classes(self):
        """
        Load images from all 43 classes (0-42)
        
        Returns:
            images: numpy array of shape (N, height, width, 3)
            labels: numpy array of shape (N,)
        """
        print(f"\n📁 Loading images from class folders...")
        print(f"   Location: {os.path.abspath(self. train_dir)}\n")
        
        if not os.path.exists(self.train_dir):
            print(f"❌ Training directory not found: {self.train_dir}")
            return None, None
        
        all_images = []
        all_labels = []
        
        # Loop through all 43 classes (0-42)
        for class_id in range(43):
            class_folder = os.path.join(self.train_dir, str(class_id))
            
            images, labels = self.load_images_from_folder(class_folder, class_id)
            
            if len(images) > 0:
                all_images.extend(images)
                all_labels.extend(labels)
                print(f"   Class {class_id: 2d}:  {len(images):5d} images")
            else:
                print(f"   Class {class_id: 2d}: No images found")
        
        if len(all_images) == 0:
            print("\n❌ No images loaded!")
            return None, None
        
        # Convert to numpy arrays
        all_images = np.array(all_images, dtype=np.uint8)
        all_labels = np.array(all_labels, dtype=np.int32)
        
        print(f"\n✅ Total loaded:   {len(all_images)} images from 43 classes")
        
        return all_images, all_labels
    
    def apply_histogram_equalization(self, images):
        """
        Apply histogram equalization to improve contrast
        Helps with varying lighting conditions
        
        Args:  
            images: numpy array of uint8 images
            
        Returns:
            numpy array of equalized images
        """
        print("\n📊 Applying histogram equalization...")
        
        equalized_images = []
        
        for img in images:
            # Convert to YUV color space
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            
            # Equalize the Y channel (luminance)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[: , :, 0])
            
            # Convert back to RGB
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            
            equalized_images.append(img_eq)
        
        print("✅ Histogram equalization complete")
        
        return np.array(equalized_images, dtype=np.uint8)
    
    def normalize_images(self, images):
        """
        Normalize pixel values from [0, 255] to [0, 1]
        
        Args:  
            images: numpy array of uint8 images
            
        Returns:
            numpy array of float32 images with values in [0, 1]
        """
        print("📊 Normalizing images to [0, 1]...")
        normalized = images.astype(np. float32) / 255.0
        print(f"✅ Normalized - range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        return normalized
    
    def analyze_dataset(self, labels):
        """
        Print detailed dataset statistics
        """
        print("\n" + "="*70)
        print("📊 DATASET ANALYSIS")
        print("="*70)
        
        class_counts = Counter(labels)
        total_images = len(labels)
        
        print(f"\nTotal images: {total_images: ,}")
        print(f"Number of classes: {len(class_counts)}")
        
        print(f"\n{'Class':<8} {'Count':<8} {'Percentage':<12} {'Bar'}")
        print("-" * 70)
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / total_images) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            
            print(f"{class_id:<8} {count:<8} {percentage: >5.2f}%       {bar}")
        
        # Statistics
        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        avg_count = np.mean(counts)
        std_count = np.std(counts)
        
        print("\n" + "-" * 70)
        print(f"Statistics:")
        print(f"  Minimum samples per class: {min_count}")
        print(f"  Maximum samples per class: {max_count}")
        print(f"  Average samples per class: {avg_count:.1f}")
        print(f"  Std deviation:               {std_count:.1f}")
        print(f"  Imbalance ratio:           {max_count/min_count:.2f}x")
        
        if max_count / min_count > 3:
            print(f"  ⚠️  Dataset is imbalanced - will use class weights")
        else:
            print(f"  ✅ Dataset is reasonably balanced")
        
        print("="*70 + "\n")
    
    def visualize_samples(self, images, labels, num_samples=15, save_path='samples.png'):
        """
        Visualize random samples from the dataset
        """
        print(f"\n📸 Visualizing {num_samples} random samples...")
        
        # Select random indices
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        # Create subplot grid
        rows = 3
        cols = 5
        fig, axes = plt. subplots(rows, cols, figsize=(15, 9))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            img = images[idx]
            label = labels[idx]
            
            # Display image (handle both uint8 and float32)
            if img.max() <= 1.0:
                axes[i].imshow(img)
            else:
                axes[i]. imshow(img. astype(np.uint8))
            
            axes[i]. set_title(f'Class: {label}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('GTSRB Dataset - Random Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to '{save_path}'")
        plt.show()
    
    def visualize_class_distribution(self, labels, save_path='class_distribution.png'):
        """
        Visualize the class distribution as a bar chart
        """
        print(f"\n📊 Visualizing class distribution...")
        
        class_counts = Counter(labels)
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(classes, counts, color='steelblue', edgecolor='black')
        
        # Color the bars by height
        max_count = max(counts)
        min_count = min(counts)
        for bar, count in zip(bars, counts):
            normalized = (count - min_count) / (max_count - min_count) if max_count > min_count else 0.5
            bar.set_color(plt.cm.RdYlGn(normalized))
        
        plt.xlabel('Class ID', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
        plt.title('GTSRB Dataset - Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(classes, fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved distribution plot to '{save_path}'")
        plt.show()


def load_and_prepare_data(data_dir='GTSRB', img_size=(64, 64), test_size=0.2, 
                          use_histogram_eq=True, visualize=True):
    """
    Complete pipeline to load and prepare GTSRB data for training
    
    Args:  
        data_dir: Path to GTSRB dataset
        img_size: Target image size (height, width)
        test_size:   Fraction of data to use for validation (0.0 to 1.0)
        use_histogram_eq: Whether to apply histogram equalization
        visualize:   Whether to create visualizations
        
    Returns: 
        X_train, X_val, y_train, y_val - Ready for training
    """
    print("\n" + "="*70)
    print("🚀 GTSRB DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Initialize loader
    loader = GTSRBLoader(data_dir=data_dir, img_size=img_size)
    
    # Load all images
    images, labels = loader.load_all_classes()
    
    if images is None or len(images) == 0:
        print("\n❌ Failed to load data.   Check your directory structure.")
        return None, None, None, None
    
    # Analyze dataset
    loader.analyze_dataset(labels)
    
    # Visualize if requested
    if visualize:  
        loader.visualize_samples(images, labels, num_samples=15, save_path='gtsrb_samples.png')
        loader.visualize_class_distribution(labels, save_path='gtsrb_distribution.png')
    
    # Preprocessing
    print("\n" + "="*70)
    print("🔧 PREPROCESSING")
    print("="*70)
    
    if use_histogram_eq:
        images = loader.apply_histogram_equalization(images)
    
    images = loader.normalize_images(images)
    
    # Split into train and validation
    print("\n" + "="*70)
    print(f"📦 SPLITTING DATA (train={100*(1-test_size):.0f}% / val={100*test_size:.0f}%)")
    print("="*70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels  # Maintain class distribution
    )
    
    print(f"\n✅ Data split complete:")
    print(f"   Training set:     {X_train.shape} - {len(X_train):,} images")
    print(f"   Validation set: {X_val.shape} - {len(X_val):,} images")
    print(f"   Classes:  {len(np.unique(y_train))}")
    print(f"   Pixel range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Data type: {X_train.dtype}")
    
    print("\n" + "="*70)
    print("✅ DATA READY FOR TRAINING!")
    print("="*70 + "\n")
    
    return X_train, X_val, y_train, y_val


# =====================================================
# MAIN - Test the data loader
# =====================================================

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_val, y_train, y_val = load_and_prepare_data(
        data_dir='GTSRB',
        img_size=(64, 64),
        test_size=0.2,
        use_histogram_eq=True,
        visualize=True
    )
    
    if X_train is not None:  
        print("🎉 Data loading successful!")
        print(f"   Ready to train with {len(X_train):,} images")