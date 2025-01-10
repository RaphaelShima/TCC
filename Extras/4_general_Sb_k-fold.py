from sklearn.model_selection import KFold
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, matthews_corrcoef
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Define directory paths
DATA_DIR = '/home/antonio/Documents/pesquisa/TCC-Raphael/final/script_and_dataset/final_dataset'

# Parameters
IMG_HEIGHT = 69
IMG_WIDTH = 69
BATCH_SIZE = 32
MAX_EPOCHS = 1000
K_FOLDS = 5
INITIAL_LR = 0.001

# Learning rate scheduler
def lr_scheduler(epoch):
    initial_lr = 0.001
    decay_rate = 0.1
    decay_steps = 10
    lr = initial_lr * np.exp(-decay_rate * (epoch / decay_steps))
    return float(lr)  # Explicitly cast to float

# Augmentation data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the dataset
data_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=True
)

# Retrieve dataset information
X, y = data_generator.next()
for i in range(len(data_generator) - 1):
    X_batch, y_batch = data_generator.next()
    X = np.concatenate((X, X_batch), axis=0)
    y = np.concatenate((y, y_batch), axis=0)

# K-Fold Cross-Validation
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Store results
fold_metrics = {'accuracy': [], 'precision': [], 'f1': [], 'mcc': [], 'auc': []}

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Starting Fold {fold + 1}/{K_FOLDS}...")

    # Split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Data generators for this fold
    train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    val_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

    # Model architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),  # Dropout after first MaxPooling

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),  # Dropout after second MaxPooling

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Dropout before the final dense layers

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout between dense layers

        layers.Dense(1, activation='sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR), loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    lr_schedule = LearningRateScheduler(lr_scheduler)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=MAX_EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping, lr_schedule]
    )

    # Evaluation metrics
    y_pred = (model.predict(val_generator) > 0.5).astype("int32")
    y_true = y_val

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    print(f"Fold {fold + 1} Metrics: Accuracy={accuracy}, Precision={precision}, F1={f1}, MCC={mcc}, AUC={auc_score}")

    fold_metrics['accuracy'].append(accuracy)
    fold_metrics['precision'].append(precision)
    fold_metrics['f1'].append(f1)
    fold_metrics['mcc'].append(mcc)
    fold_metrics['auc'].append(auc_score)

# Summarize results across folds
print("\nCross-Validation Results:")
print(f"Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"F1-Score: {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
print(f"MCC: {np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}")
print(f"AUC: {np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}")
