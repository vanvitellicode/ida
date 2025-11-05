import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalMaxPooling1D, Input, Add, Activation, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# Caricamento e preprocessamento dei dati
df = pd.read_csv(r"FinalDataset.csv")
df = df.drop(columns=['posts'], errors='ignore')
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns='type').values
y = df['type'].values

# Standardizzazione delle feature
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding del target
num_classes = len(np.unique(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=20)

input_dim = X_train.shape[1]

# Definizione di 6 modelli deep learning

def build_shallow_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_deep_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_dropout_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_batchnorm_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(128, input_shape=(input_dim,)),
        BatchNormalization(),
        Activation('relu'),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_cnn_1d(input_dim, num_classes):
    model = Sequential([
        Reshape((input_dim, 1), input_shape=(input_dim,)),
        Conv1D(32, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_residual_mlp(input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    x = Dense(64)(inputs)
    x = Activation('relu')(x)
    res = Dense(64)(x)
    x = Add()([x, res])
    x = Activation('relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

models = {
    'Shallow MLP': build_shallow_mlp(input_dim, num_classes),
    'Deep MLP': build_deep_mlp(input_dim, num_classes),
    'Dropout MLP': build_dropout_mlp(input_dim, num_classes),
    'BatchNorm MLP': build_batchnorm_mlp(input_dim, num_classes),
    'CNN 1D': build_cnn_1d(input_dim, num_classes),
    'Residual MLP': build_residual_mlp(input_dim, num_classes)
}

# Funzione di valutazione
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Training
    t0 = time.time()
    model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early], verbose=0)
    train_time = time.time() - t0

    # Predizione
    t1 = time.time()
    y_pred_prob = model.predict(X_test, verbose=0)
    pred_time = time.time() - t1

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Training Time (s)': train_time,
        'Prediction Time (s)': pred_time
    }


results = []
for name, model in models.items():
    print(f"Training {name}...")
    results.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))

results_df = pd.DataFrame(results)
print("\nRisultati Deep Learning Models:")
print(results_df)

# Salvataggio CSV
results_df.to_csv('dl_model_comparison.csv', index=False)
