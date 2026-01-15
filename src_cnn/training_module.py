import os
import shutil
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
from rdkit.DataStructs import ConvertToNumpyArray
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Conv1D, Lambda, Dense, Concatenate, Reshape,
    BatchNormalization, ReLU, Add, Flatten, Dropout
)
from tensorflow.keras.models import Model, Sequential, load_model
# from scr_training.functions import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from kerastuner import RandomSearch
import sys
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.callbacks import ReduceLROnPlateau

def save_backup_prevModel():
    """
    Saves the tuner_dir folder as backup to avoid loosing information on previous trained
    models when overwritten
    """
    tuner_dir = "tuner_dir"

    backup_root = os.path.join("results", "tuner_backups")
    os.makedirs(backup_root, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(backup_root, f"tuner_dir_subgroup_backup_{timestamp}")

    if os.path.exists(tuner_dir):
        shutil.copytree(tuner_dir, backup_dir)
        print(f"Folder '{tuner_dir}' saved: '{backup_dir}'")
    else:
        print(f"Could not find folder '{tuner_dir}'")

# --- PERSONALIZED BLOCKS ---
def resnet_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Add()([x, shortcut])
    return ReLU()(x)

def dense_block(input, dropout_val, TOTAL_LENGTH):
    dense = Dense(1024, activation='relu')(input)
    dropout = Dropout(dropout_val)(dense)
    dense = Dense(512, activation='relu')(dropout)
    dropout = Dropout(dropout_val)(dense)
    dense = Dense(256, activation='relu')(dropout)
    return dense
# ---------------------------

# --- MODEL ---
def create_build_model(input):
    def build_model(hp):
        # Separation of fingerprint into type-sections
        NUM_SUBGROUPS = 4
        TOTAL_LENGTH = input.shape[1]
        SUBGROUP_SIZE = TOTAL_LENGTH // NUM_SUBGROUPS
        input_layer = Input(shape=(TOTAL_LENGTH, 1))
        subgroup_outputs = []

        # hyperparameters
        res_blocks_min, res_blocks_max = 1, 30
        dropout_min, dropout_max = 0, 0.5

        use_dense = hp.Boolean('use_dense')
        dropout_val = hp.Float('dropout_val', min_value=dropout_min, max_value=dropout_max, step=0.1)
        conv_filters = hp.Choice('conv_filters', values=[8, 16, 32])
        num_blocks_res = hp.Int('blocks_resnet', min_value=res_blocks_min, max_value=res_blocks_max)

        # Separation of input into small groups
        for i in range(NUM_SUBGROUPS):
            segment = Lambda(lambda t: t[:, i * SUBGROUP_SIZE:(i + 1) * SUBGROUP_SIZE, :])(input_layer)
            if use_dense:
                x = Flatten()(segment)
                x = dense_block(x, dropout_val, TOTAL_LENGTH)
                x = Reshape((x.shape[1], -1))(x)
            else:
                x = Conv1D(filters=conv_filters, kernel_size=7, padding='same', activation='relu')(segment)
            subgroup_outputs.append(x)

        concatenated = Concatenate()(subgroup_outputs)

        resnet_out = concatenated
        for _ in range(num_blocks_res):
            resnet_out = resnet_block(resnet_out, filters=32)
        reduced_output = Dense(64, activation='relu')(Flatten()(resnet_out))

        final_block = Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        output = final_block(reduced_output)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        model.summary()
        return model
    return build_model
# -------------

# --- OBTAIN X AND Y FROM DATASET ---
def dataset_from_file(file_path):
    """
    Reads data from pkl file and separates the fingerprints from the rt for the model.

    :param pkl_path: path to pkl file with required data
    :return:
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        fingerprints = pd.read_csv(file_path)
    elif ext == ".pkl":
        fingerprints = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use .csv or .pkl")

    feature_cols = [col for col in fingerprints.columns if col.startswith('V')]
    mask = fingerprints['rt'] >= 300

    filtered_data = fingerprints[mask]
    X = filtered_data[feature_cols].values.astype(np.float32)
    y = filtered_data['rt'].values.astype(np.float32)

    X = X.reshape(-1, len(feature_cols), 1)

    # X[:100], y[:100]

    return X, y
# -----------------------------------

# --- CREATING MODEL ---
def create_model(smrt_path):
    """
    Creates, trains and tests the model given the rt data file.
    :param smrt_path: path to file with fingerprints and rt
    """
    X, y = dataset_from_file(smrt_path)

    y_binned = pd.qcut(y, q=10, labels=False)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = RobustScaler()
    build_model = create_build_model(X)

    # Tuner and Callbacks
    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y_binned)):
        fold_num = fold_idx + 1
        print(f"\nFold {fold_num}")
        if fold_num > 1:
            break

        project_dir = os.path.join("tuner_dir", f"resnet_dense_tune_fold{fold_num}")
        os.makedirs(project_dir, exist_ok=True)

        fold_dir = os.path.join("results", f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val_raw, y_test = y[train_idx], y[test_idx]

        y_train_val = scaler.fit_transform(y_train_val_raw.reshape(-1, 1)).flatten()

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        tuner = RandomSearch(
            build_model,
            objective='val_mae',
            max_trials=10,
            executions_per_trial=3,
            directory=project_dir,
            project_name=f'resnet_dense{fold_num}',
            overwrite=False
        )

        tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop, reduce_lr]
        )

        best_model = tuner.get_best_models(num_models=1)[0]

        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            verbose=1,
            callbacks=[early_stop, reduce_lr]
        )

        best_model.save(os.path.join(fold_dir, "best_model.keras"))
        joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

        mae_graphic(history,fold_dir)

        test_model(X_test, y_test, os.path.join(fold_dir, "scaler.pkl"), os.path.join(fold_dir, "best_model.h5"), fold_dir)

def test_model(test_data, rt_data, scaler_path, model_path, result_path):
    """
    Test the viability of the model given the rt test data file. The results are then presented in a
    scatter plot for comparison and better visualization of the results
    :param test_data: Fingerprints used for testing
    :param rt_data: Actual rt of the given fingerprints
    :param scaler_path: scaler path used for model
    :param model_path: stored model path
    :param result_path: path where the results will be stored
    :return:
    """
    model = load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    scaler = joblib.load(scaler_path)
    y_pred_scaled = model.predict(test_data).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    print(y_pred_orig.shape)
    print(rt_data.shape)
    scatter_plot_test(rt_data, y_pred_orig, result_path)

# ----------------------

# --- GRAPHS ---
def mae_graphic(history, results_path):
    """
    Creates a graphic reresentation of the evolution of the mae through the training and validation
    process for comparison
    :param history: MAE records during training and validation steps
    :param results_path:  Path where the results will be stored
    """
    hist = history.history if hasattr(history, "history") else history
    mae_key = 'mae' if 'mae' in hist else 'mean_absolute_error'
    val_mae_key = 'val_mae' if 'val_mae' in hist else 'val_mean_absolute_error'
    plt.figure(figsize=(10, 4))
    plt.plot(hist[mae_key], label='MAE training')
    plt.plot(hist[val_mae_key], label='MAE validation')
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(f"MAE Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "mae_plot.png"))
    plt.close()

def scatter_plot_test(og_rt, predicted_rt, output_path):
    """
    Generates a scatter plot for comparison and better visualization of the results from
    the model and the expected results
    :param og_rt: Expected rt value
    :param predicted_rt: Observed rt value
    :param output_path: Path where the results will be stored
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(og_rt, predicted_rt, alpha=0.6, edgecolors='k')
    plt.plot([min(og_rt), max(og_rt)], [min(og_rt), max(og_rt)],
             'r--', label='Ideal')
    plt.xlabel("RT real")
    plt.ylabel("RT predicted")
    plt.title(f"Real vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "scatter_plot.png"))
    plt.close()
# --------------

# --- NEW FUNCTION ---
def plot_training_history(history, title="Model Loss (MAE)", figsize=(10, 6)):
    """
    Plots training and validation loss over epochs.

    Parameters:
    - history: the history object returned by model.fit()
    - title: title of the plot
    - figsize: size of the figure
    """
    plt.figure(figsize=figsize)

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
    plt.show()
# --------------------