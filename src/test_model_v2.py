"""
Test the trained GTSRB model on test set
"""

import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import cv2

print("\n" + "="*70)
print("GTSRB MODEL TESTING")
print("="*70 + "\n")

# Load test data
test_dir = Path('GTSRB')
csv_path = test_dir / 'Test.csv'

print("Loading test data...")
test_df = pd.read_csv(csv_path)
print(f"Found {len(test_df)} images with labels\n")

X_test = []
y_test = []
filenames = []

for idx, row in test_df.iterrows():
    img_path = test_dir / row['Path']
    label = row['ClassId']
    
    img = cv2.imread(str(img_path))
    if img is not None: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        X_test.append(img)
        y_test.append(label)
        filenames.append(row['Path'])

X_test = np.array(X_test, dtype=np. uint8)
y_test = np.array(y_test, dtype=np.int32)

print(f"Loaded {len(X_test)} images")

# Preprocess
print("Preprocessing...")
X_test_eq = []
for img in X_test:
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[: , : , 0])
    X_test_eq.append(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))

X_test = np.array(X_test_eq, dtype=np. float32) / 255.0

# Load model
print("Loading model...")
model = keras.models.load_model('models_v2/best_model.keras')

# Evaluate
print("Evaluating.. .\n")
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=128, verbose=0)

print("="*70)
print("RESULTS")
print("="*70)
print(f"Test Accuracy:  {test_accuracy*100:.2f}%")
print(f"Test Loss:      {test_loss:.4f}")
print("="*70 + "\n")

# Generate predictions
print("Generating predictions...")
y_pred = model.predict(X_test, batch_size=128, verbose=0)
y_pred_classes = np. argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, digits=3))

# Save results to CSV
print("\nSaving results to CSV...")
results_df = pd.DataFrame({
    'Image': filenames,
    'True_Label': y_test,
    'Predicted_Label': y_pred_classes,
    'Confidence': [y_pred[i][y_pred_classes[i]] * 100 for i in range(len(y_pred))],
    'Correct': y_test == y_pred_classes
})
results_df.to_csv('test_results.csv', index=False)
print("Saved test_results.csv\n")

# Calculate per-class accuracy
class_accuracies = []
for class_id in range(43):
    mask = y_test == class_id
    if mask.sum() > 0:
        class_accuracies.append(accuracy_score(y_test[mask], y_pred_classes[mask]) * 100)
    else:
        class_accuracies.append(0)

# Generate visualizations
print("Generating visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Class', fontsize=11)
plt.ylabel('True Class', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_test.png', dpi=150, bbox_inches='tight')
plt.close()

# Per-Class Accuracy
plt.figure(figsize=(16, 5))
bars = plt.bar(range(43), class_accuracies, edgecolor='black', linewidth=0.5)
for bar, acc in zip(bars, class_accuracies):
    bar.set_color(plt.cm.RdYlGn(acc / 100))
plt.axhline(np.mean(class_accuracies), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(class_accuracies):.2f}%')
plt.xlabel('Class ID', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
plt.title('Per-Class Accuracy on Test Set', fontsize=13, fontweight='bold')
plt. xticks(range(43), fontsize=8)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()

# Sample Predictions
np.random.seed(42)
num_samples = min(15, len(X_test))
indices = np.random.choice(len(X_test), num_samples, replace=False)
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.ravel()

for i, idx in enumerate(indices):
    img = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred_classes[idx]
    confidence = y_pred[idx][pred_label] * 100
    
    axes[i]. imshow(img)
    color = 'green' if true_label == pred_label else 'red'
    axes[i]. set_title(f'True:  {true_label} | Pred: {pred_label}\n{confidence:.1f}%', 
                      fontsize=9, fontweight='bold', color=color)
    axes[i].axis('off')

for i in range(num_samples, 15):
    axes[i].axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved confusion_matrix_test.png")
print("Saved per_class_accuracy.png")
print("Saved sample_predictions.png")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total Test Samples:      {len(y_test):,}")
print(f"Correct Predictions:     {(y_test == y_pred_classes).sum():,}")
print(f"Incorrect Predictions:   {(y_test != y_pred_classes).sum():,}")
print(f"Mean Class Accuracy:     {np.mean(class_accuracies):.2f}%")
print(f"Best Class Accuracy:     {max(class_accuracies):.2f}%")
print(f"Worst Class Accuracy:    {min(class_accuracies):.2f}%")
print("="*70 + "\n")

print("Testing complete.\n")