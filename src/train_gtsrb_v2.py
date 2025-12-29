"""
GTSRB Traffic Sign Classifier - Optimized & Clean
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide warnings

class GTSRBClassifier:
    def __init__(self, num_classes=43, img_size=(64, 64)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build efficient CNN with residual connections"""
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Initial conv
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        
        # Block 1
        x = layers. Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers. Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers. MaxPooling2D(2)(x)
        x = layers. Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers. BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers. Dropout(0.4)(x)
        
        # Output (float32 for numerical stability)
        x = layers. Activation('linear', dtype='float32')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        self.model = models. Model(inputs, outputs)
        
        # Compile with aggressive learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n✅ Model built:  {self.model.count_params():,} parameters")
        return self.model
    
    def create_augmentation(self):
        """Fast data augmentation"""
        return keras.Sequential([
            layers. RandomRotation(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
        ])
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=128, use_augmentation=True):
        """Train with optimal settings"""
        
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING")
        print(f"{'='*70}")
        print(f"Training:    {len(X_train):,} samples")
        print(f"Validation: {len(X_val):,} samples")
        print(f"Batch size: {batch_size}")
        print(f"Epochs:      {epochs}")
        print(f"{'='*70}\n")
        
        # Compute class weights
        class_weights = self._compute_class_weights(y_train)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Convert to TF dataset for speed
        if use_augmentation:
            augmentation = self.create_augmentation()
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_ds = train_ds. shuffle(10000).batch(batch_size).prefetch(tf.data. AUTOTUNE)
            train_ds = train_ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            train_ds = tf.data.Dataset. from_tensor_slices((X_train, y_train))
            train_ds = train_ds. shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(batch_size).prefetch(tf.data. AUTOTUNE)
        
        # Train
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print(f"\n✅ Training complete!")
        return self.history
    
    def _compute_class_weights(self, y_train):
        """Compute balanced class weights"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"⚖️  Class weights computed (range: {weights.min():.2f} - {weights.max():.2f})")
        return class_weights
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION")
        print(f"{'='*70}\n")
        
        # Create dataset
        test_ds = tf. data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)
        
        # Evaluate
        loss, accuracy = self.model.evaluate(test_ds, verbose=0)
        print(f"Test Loss:      {loss:.4f}")
        print(f"Test Accuracy:  {accuracy*100:.2f}%")
        
        # Predictions
        y_pred = self.model.predict(test_ds, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Per-class metrics
        print(f"\n{'='*70}")
        print("PER-CLASS METRICS")
        print(f"{'='*70}\n")
        print(classification_report(y_test, y_pred_classes, digits=3))
        
        return accuracy, y_pred_classes
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history found")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✅ Saved training_history.png")
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, top_n=43):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm[: top_n, :top_n], annot=False, fmt='d', cmap='Blues', 
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix (Top {top_n} classes)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✅ Saved confusion_matrix.png")
        plt.show()


def main():
    print(f"\n{'='*70}")
    print(f"🚦 GTSRB TRAFFIC SIGN CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Load data
    print("📁 Loading data...")
    from gtsrb_data_loader import load_and_prepare_data
    
    X_train, X_val, y_train, y_val = load_and_prepare_data(
        data_dir='GTSRB',
        img_size=(64, 64),
        test_size=0.2,
        use_histogram_eq=True,
        visualize=False  # Disable for speed
    )
    
    if X_train is None: 
        print("❌ Failed to load data")
        return None
    
    # Build model
    print("\n🏗️  Building model...")
    classifier = GTSRBClassifier(num_classes=43, img_size=(64, 64))
    classifier.build_model()
    
    # Train
    classifier.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=100,
        batch_size=128,
        use_augmentation=True
    )
    
    # Evaluate
    accuracy, y_pred = classifier.evaluate(X_val, y_val)
    
    # Plots
    classifier.plot_history()
    classifier.plot_confusion_matrix(y_val, y_pred, top_n=43)
    
    # Save final model
    classifier.model.save('final_model.keras')
    print("\n✅ Saved final_model.keras")
    
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE")
    print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return classifier


if __name__ == "__main__":
    # Set memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🎮 Using GPU: {len(gpus)} device(s)")
    else:
        print("💻 Using CPU")
    
    classifier = main()