import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import gc


def training(class_dict):
    # Files
    fingerprints_file = "tests_for_smrt/resources/smrt_fingerprints.csv"

    # Read fingerprints CSV
    df_fingerprints = pd.read_csv(fingerprints_file)

    # Filter out InChIs with rt <= 300
    df_fingerprints = df_fingerprints[df_fingerprints['rt'] > 400]

    # Prepare results
    results = []

    # Initialize RobustScaler for rt values
    scaler = RobustScaler()

    # Process each class
    for class_name, (freq, inchis) in class_dict.items():
        # Filter InChIs that are still present after rt filtering
        valid_inchis = [inchi for inchi in inchis if inchi in df_fingerprints['inchi'].values]
        if not valid_inchis:
            continue

        # Get corresponding fingerprint data
        class_df = df_fingerprints[df_fingerprints['inchi'].isin(valid_inchis)]
        X = class_df.filter(regex='^V\d+').values  # Fingerprint features
        y = class_df['rt'].values.reshape(-1, 1)  # Target (retention time), reshaped for scaler

        # Skip if insufficient data
        if len(X) < 5:  # Need enough samples for meaningful splits
            continue

        # 80/20 train-test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 80/20 train-validation split from train_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        # Calculate 10th and 90th quantiles for rt values
        q10 = np.percentile(y_train, 10)
        q90 = np.percentile(y_train, 90)
        rt_range = q90 - q10 if q90 != q10 else 1  # Avoid division by zero

        # Fit scaler on training rt values and transform train, val, test
        y_train_scaled = scaler.fit_transform(y_train)
        y_val_scaled = scaler.transform(y_val)

        # Define single neuron model
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(1)
        ], name="singleDense")

        # Define simple model
        """model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(512, activation='relu', kernel_initializer='he_normal'),
            Dense(512, activation='relu', kernel_initializer='he_normal'),
            Dense(512, activation='relu', kernel_initializer='he_normal'),
            Dense(1, activation="linear", name="rt_output")
        ], name="simpleModel")"""

        # Compile model
        optimizer = RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True,
            min_delta=0.001
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=30,
            min_lr=5e-6,
            min_delta=0.001
        )

        # Train model on scaled data
        model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=200000,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Predict on test set and inverse transform predictions
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)

        # Calculate MAE in original scale
        test_mae_original = mean_absolute_error(y_test, y_pred)

        # Calculate normalized MAE (MAE divided by range of 10th to 90th quantiles)
        test_mae_normalized = test_mae_original / rt_range

        # Store results
        results.append({
            'Class': class_name,
            'Frequency': len(valid_inchis),
            'Test_MAE': test_mae_original,  # Store MAE in original scale
            'Normalized_MAE': test_mae_normalized  # Store normalized MAE
        })

        #Clear GPU memory before training (Lucecitas)
        tf.keras.backend.clear_session()
        gc.collect()
        tf.config.experimental.reset_memory_stats('GPU:0')

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    return results_df
