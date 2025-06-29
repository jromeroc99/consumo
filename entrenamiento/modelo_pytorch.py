import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys


from datos.procesamiento import ProcesadorDatos, MONGO_URI, DB_NAME, LATITUD, LONGITUD


# Configuración
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_STATE = 42

# 1. Cargar datos limpios
def cargar_datos():
    if ProcesadorDatos is not None and MONGO_URI is not None:
        procesador = ProcesadorDatos(MONGO_URI, DB_NAME, LATITUD, LONGITUD)
        df = procesador.obtener_df_limpia()
    elif os.path.exists('../datos_limpios.csv'):
        df = pd.read_csv('../datos_limpios.csv')
    else:
        raise FileNotFoundError('No se encontraron datos limpios.')
    return df

def preparar_datos(df):
    # Quitar columnas objetivo y no numéricas
    target_cols = ['fase0', 'fase1', 'fase2']
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    # Eliminar columnas no numéricas
    X = X.select_dtypes(include=[np.number])
    return X, y

class ConsumoDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def entrenar():
    df = cargar_datos()
    X, y = preparar_datos(df)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=RANDOM_STATE)
    train_ds = ConsumoDataset(pd.DataFrame(X_train), pd.DataFrame(y_train))
    test_ds = ConsumoDataset(pd.DataFrame(X_test), pd.DataFrame(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim=X.shape[1], output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

    # Evaluación
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_ds)
    print(f"Test Loss: {test_loss:.4f}")

    # Guardar modelo y scalers
    torch.save(model.state_dict(), 'modelo_consumo.pth')
    import joblib
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("Modelo y scalers guardados.")

if __name__ == "__main__":
    entrenar() 