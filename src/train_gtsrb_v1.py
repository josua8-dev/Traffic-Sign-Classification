"""
Traffic Sign Classifier - Training script for GTSRB
Clean implementation with no dependencies on old code
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.utils. class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from datetime import datetime

from gtsrb_data_loader import load_and_prepare_data


class TrafficSignCNN: 
    """
    Convolutional Neural Network for Traffic Sign Classification
    """
    
    def __init__(self, num_classes=43, img_size=(64, 64)):
        """
        Initialize the classifier
        
        Args:
            num_classes: Number of traffic sign classes (43 for GTSRB)
            img_size: Input image size (height, width)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        
        print(f"\n🏗️  Traffic Sign CNN Classifier")
        print(f"   Classes: {num_classes}")
        print(f"   Input size: {img_size}")
    
    def build_model(self):
        """
        Build CNN architecture optimized for traffic signs
        """
        print(f"\n🔨 Building CNN model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # Convolutional Block 1
            layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers. BatchNormalization(),
            layers. Conv2D(32, (5, 5), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers. MaxPooling2D((2, 2)),
            layers. Dropout(0.25),
            
            # Convolutional Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers. BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Convolutional Block 3
            layers. Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers. BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Convolutional Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self. num_classes, activation='softmax')
        ])
        
        self.model = model
        
        print("✅ Model architecture created")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        """
        print(f"\n⚙️  Compiling model...")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss:  Sparse Categorical Crossentropy")
        
        self.model.compile(
            optimizer=keras.optimizers. Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled")
    
    def summary(self):
        """
        Print model summary
        """
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        self.model.summary()
        print(f"\nTotal parameters: {self.model.count_params():,}")
        print("="*70 + "\n")
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator
        Increases effective dataset size and improves generalization
        """
        print("\n🔄 Creating data augmentation...")
        
        datagen = ImageDataGenerator(
            rotation_range=15,              # Rotate ±15 degrees
            width_shift_range=0.1,          # Shift horizontally 10%
            height_shift_range=0.1,         # Shift vertically 10%
            zoom_range=0.2,                 # Zoom ±20%
            shear_range=0.1,                # Shear transformation
            brightness_range=[0.7, 1.3],    # Brightness 70-130%
            fill_mode='nearest'             # Fill empty pixels
        )
        
        print("✅ Data augmentation configured:")
        print("   - Rotation: ±15°")
        print("   - Shifts: ±10%")
        print("   - Zoom: ±20%")
        print("   - Brightness: 70-130%")
        
        return datagen
    
    def compute_class_weights(self, y_train):
        """
        Compute class weights to handle imbalanced dataset
        Gives more importance to underrepresented classes
        """
        print("\n⚖️  Computing class weights...")
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))
        
        # Show sample weights
        print(f"   Example weights:")
        for cls in list(classes)[:5]: 
            print(f"     Class {cls}: {class_weights[cls]:.3f}")
        print(f"     ...")
        
        return class_weights
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=64, use_augmentation=True):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val:  Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            use_augmentation: Whether to use data augmentation
        """
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70)
        print(f"   Training samples:    {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Epochs:              {epochs}")
        print(f"   Batch size:         {batch_size}")
        print(f"   Data augmentation:   {use_augmentation}")
        print("="*70 + "\n")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'best_traffic_sign_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train with or without augmentation
        if use_augmentation:
            datagen = self.create_data_augmentation()
            datagen.fit(X_train)
            
            self.history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        else:
            self.history = self.model. fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
        
        print("\n✅ Training complete!")
    
    def plot_history(self, save_path='training_history.png'):
        """
        Plot training history
        """
        if self.history is None:
            print("⚠️  No training history available")
            return
        
        print(f"\n📊 Plotting training history...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], 
                label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax1.plot(self.history.history['val_accuracy'], 
                label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(self. history.history['loss'], 
                label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax2.plot(self.history. history['val_loss'], 
                label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved training history to '{save_path}'")
        plt.show()
    
    def evaluate(self, X_val, y_val):
        """
        Evaluate model on validation set
        """
        print("\n" + "="*70)
        print("📊 MODEL EVALUATION")
        print("="*70)
        
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        
        print(f"   Validation Loss:      {loss:.4f}")
        print(f"   Validation Accuracy: {accuracy*100:.2f}%")
        print("="*70)
        
        return loss, accuracy
    
    def plot_confusion_matrix(self, X_val, y_val, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        print(f"\n📈 Generating confusion matrix...")
        
        # Get predictions
        y_pred = self. model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        
        # Plot
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - All 43 Classes', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to '{save_path}'")
        plt.show()
    
    def show_predictions(self, X_val, y_val, num_samples=10, save_path='predictions.png'):
        """
        Show model predictions on random samples
        """
        print(f"\n🔮 Showing predictions on {num_samples} samples...")
        
        # Select random samples
        indices = np.random.choice(len(X_val), num_samples, replace=False)
        
        # Predict
        predictions = self.model.predict(X_val[indices], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            axes[i].imshow(X_val[indices[i]])
            
            true_label = y_val[indices[i]]
            pred_label = predicted_classes[i]
            confidence = predictions[i][pred_label] * 100
            
            # Green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            
            title = f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%"
            axes[i]. set_title(title, fontsize=10, color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Model Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved predictions to '{save_path}'")
        plt.show()
    
    def save_model(self, filepath='traffic_sign_model.keras'):
        """
        Save the trained model
        """
        self.model.save(filepath)
        print(f"\n💾 Model saved to '{filepath}'")
    
    def load_model(self, filepath='traffic_sign_model.keras'):
        """
        Load a saved model
        """
        self. model = keras.models.load_model(filepath)
        print(f"📂 Model loaded from '{filepath}'")


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def main():
    """
    Complete training pipeline for GTSRB traffic sign classification
    """
    print("\n" + "="*70)
    print("🚦 GTSRB TRAFFIC SIGN CLASSIFICATION")
    print("="*70)
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ==================== STEP 1: Load Data ====================
    print("\n📁 STEP 1: LOADING DATA")
    
    X_train, X_val, y_train, y_val = load_and_prepare_data(
        data_dir='GTSRB',
        img_size=(64, 64),
        test_size=0.2,
        use_histogram_eq=True,
        visualize=True
    )
    
    if X_train is None: 
        print("\n❌ Failed to load data.  Exiting.")
        return None
    
    # ==================== STEP 2: Build Model ====================
    print("\n🏗️  STEP 2: BUILDING MODEL")
    
    classifier = TrafficSignCNN(num_classes=43, img_size=(64, 64))
    classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    classifier.summary()
    
    # ==================== STEP 3: Train Model ====================
    print("\n🎓 STEP 3: TRAINING MODEL")
    
    classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=64,
        use_augmentation=True
    )
    
    # ==================== STEP 4: Evaluate ====================
    print("\n📊 STEP 4: EVALUATION")
    
    classifier.plot_history(save_path='training_history.png')
    loss, accuracy = classifier.evaluate(X_val, y_val)
    
    # ==================== STEP 5: Visualizations ====================
    print("\n📸 STEP 5: CREATING VISUALIZATIONS")
    
    classifier.show_predictions(X_val, y_val, num_samples=10, save_path='predictions.png')
    classifier.plot_confusion_matrix(X_val, y_val, save_path='confusion_matrix.png')
    
    # ==================== STEP 6: Save Model ====================
    print("\n💾 STEP 6: SAVING MODEL")
    
    classifier.save_model('traffic_sign_model.keras')
    
    # Save training info
    info = {
        'training_samples': int(len(X_train)),
        'validation_samples': int(len(X_val)),
        'num_classes': 43,
        'final_accuracy': float(accuracy),
        'final_loss': float(loss),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    print("💾 Training info saved to 'training_info.json'")
    
    # ==================== COMPLETE ====================
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n🎉 Final Validation Accuracy: {accuracy*100:.2f}%")
    print(f"\n📁 Generated files:")
    print("   - best_traffic_sign_model.keras  (best model)")
    print("   - traffic_sign_model.keras       (final model)")
    print("   - training_history.png           (accuracy/loss curves)")
    print("   - predictions.png                (sample predictions)")
    print("   - confusion_matrix.png           (confusion matrix)")
    print("   - gtsrb_samples.png              (dataset samples)")
    print("   - gtsrb_distribution.png         (class distribution)")
    print("   - training_info.json             (training metadata)")
    print("\n" + "="*70 + "\n")
    
    return classifier


if __name__ == "__main__":
    classifier = main()