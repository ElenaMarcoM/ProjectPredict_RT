import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score, \
    mean_absolute_percentage_error
# from sktime.performance_metrics.forecasting import mean_squared_percentage_error, median_absolute_percentage_error
import tensorflow as tf
import gc


def training(class_dict):
    # Files
    fingerprints_file = "resources/smrt_fingerprints.csv"

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
        y_test_scaled = scaler.transform(y_test)

        """
        # Define single neuron model
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(1)
        ], name="singleDense")
        
        # Define simple model
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(1)], name="Dense")
        """
        # Model with division
        s1, s2, s3 = 1024, 166, 1024
        total = s1 + s2 + s3
        inp = Input(shape=(total,))

        a = Lambda(lambda x: x[:, :s1])(inp)
        b = Lambda(lambda x: x[:, s1:s1 + s2])(inp)
        c = Lambda(lambda x: x[:, s1 + s2:])(inp)
        # A: 1024 → 1024 → 128
        xa = Dense(1024, activation="relu")(a)
        xa = Dense(128, activation="relu")(xa)
        # B: 166 → 166 → 128
        xb = Dense(166, activation="relu")(b)
        xb = Dense(128, activation="relu")(xb)
        # C: igual que B
        xc = Dense(1024, activation="relu")(c)
        xc = Dense(128, activation="relu")(xc)
        # Merge
        m = Concatenate()([xa, xb, xc])
        # MLP final
        h = Dense(128, activation="relu")(m)
        h = Dense(64, activation="relu")(h)
        out = Dense(1)(h)
        model = Model(inp, out)
        model.summary()

        # Compile model
        optimizer = RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            min_delta=0.001
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=15,
            min_lr=5e-6,
            min_delta=0.001
        )

        # Train model on scaled data
        history = model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # TODO: NEW
        def plot_training_history(history, title="Model Loss (MAE)", figsize=(10, 6), group_name = None):
            """
            Plots training and validation loss over epochs.

            Parameters:
            - history: the history object returned by model.fit()
            - title: title of the plot
            - figsize: size of the figure
            """
            plt.figure(figsize=figsize)
            # TODO: NEW
            if group_name:
                title = f"{title} – {group_name}"

            # Plot training & validation loss
            plt.plot(history.history['loss'], label='Training Loss (MAE)', linewidth=2)
            plt.plot(history.history['val_loss'], label='Validation Loss (MAE)', linewidth=2)

            plt.title(title, fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Mean Absolute Error', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            # Optional: add a marker for the best validation loss
            best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
            plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                        label=f'Best Val Loss (Epoch {best_epoch})')

            plt.legend()
            plt.tight_layout()
            #TODO: NEW
            plt.savefig(os.path.join("results1", f"{title}.png"))
            plt.show()

        plot_training_history(history, group_name=class_name)
        """
        # Predict on test set and inverse transform predictions
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        """
        # Predictions and test unscaled
        pred_scaled = model.predict([X_test], batch_size=32, verbose=0).flatten()
        y_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

        # Calculate MAE in original scale
        test_mae_original = mean_absolute_error(y_test, y_pred)

        # Calculate normalized MAE (MAE divided by range of 10th to 90th quantiles)
        test_mae_normalized = test_mae_original / rt_range

        # Linear regression between y_test and y_pred
        linreg = LinearRegression()
        linreg.fit(y_test.reshape(-1, 1), y_pred.reshape(-1, 1))
        y_pred_lin = linreg.predict(y_test.reshape(-1, 1)).flatten()

        # Calculate all th metrics
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_pred, y_pred_lin)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        #TODO: NEW
        # mspe = mean_squared_percentage_error(y_test, y_pred) ERROR: This function does NOT exist in sklearn.metrics
        # medape = median_absolute_percentage_error(y_test, y_pred) ERROR: This function does NOT exist in sklearn.metrics
        print("\nMAE:", mae,
              "\nMEDAE:", medae,
              "\nMSE:", mse,
              "\nRMSE:", r2,
              "\nMAPE:", mape)

        def plot_scatter(y_true, y_pred, y_pred_lin, r2, save_path, title_prefix=""):
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5, s=1)
            plt.plot(y_true, y_pred_lin, color='green', label=f'Linear fit (R² = {r2:.3f})')
            plt.title(f'{title_prefix} Scatter plot, R² = {r2:.3f}')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.legend()
            plt.savefig(save_path)
            plt.show()
            plt.close()
            # Store results
            results.append({
                'Class': class_name,
                'Frequency': len(valid_inchis),
                'Test_MAE': test_mae_original,  # Store MAE in original scale
                'Normalized_MAE': test_mae_normalized  # Store normalized MAE
            })

            # Clear GPU memory before training (Lucecitas)
            tf.keras.backend.clear_session()
            gc.collect()

            # Looks for a GPU to clear
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.reset_memory_stats('GPU:0')
                except ValueError:
                    # Por si el nombre del dispositivo no cuadra exacto
                    pass
        #TODO: NEW
        plot_scatter(y_test, y_pred, y_pred_lin, r2, os.path.join('results1',f'{class_name}_ScatterPlot.png' ), title_prefix = class_name)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def test_group_size(s1, s2, s3):
    total = s1 + s2 + s3
    inp = Input(shape=(total,))

    a = Lambda(lambda x: x[:, :s1])(inp)
    b = Lambda(lambda x: x[:, s1:s1 + s2])(inp)
    c = Lambda(lambda x: x[:, s1 + s2:])(inp)
    # A: 1024 → 1024 → 128
    xa = Dense(1024, activation="relu")(a)
    xa = Dense(128, activation="relu")(xa)
    # B: 166 → 166 → 128
    xb = Dense(166, activation="relu")(b)
    xb = Dense(128, activation="relu")(xb)
    # C: igual que B
    xc = Dense(1024, activation="relu")(c)
    xc = Dense(128, activation="relu")(xc)
    # Merge
    m = Concatenate()([xa, xb, xc])
    # MLP final
    h = Dense(128, activation="relu")(m)
    h = Dense(64, activation="relu")(h)
    out = Dense(1)(h)
    model = Model(inp, out)
    model.summary()

