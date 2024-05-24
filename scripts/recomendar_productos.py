import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos desde archivos CSV
compras_df = pd.read_csv("/home/penguin/Hackaton/data/compras.csv")
inventario_df = pd.read_csv("/home/penguin/Hackaton/data/inventario.csv")

# Función para preprocesar datos
def preprocesar_datos(compras_df, inventario_df):
    # Fusionar datos de compras con información de inventario
    compras_agrupadas = compras_df.merge(inventario_df[['id_producto', 'nombre_producto']], on='id_producto')
    
    # Codificar nombres de productos como números
    encoder = LabelEncoder()
    compras_agrupadas['producto_encoded'] = encoder.fit_transform(compras_agrupadas['nombre_producto'])
    
    # Escalar la cantidad de productos comprados
    scaler = StandardScaler()
    compras_agrupadas[['cantidad']] = scaler.fit_transform(compras_agrupadas[['cantidad']])
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X = compras_agrupadas[['id_cliente', 'cantidad']]
    y = compras_agrupadas['producto_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoder.classes_

# Función para recomendar varios productos a un cliente
def recomendar_varios_productos(modelo, X_cliente, nombre_productos, n=5):
    # Convertir datos del cliente a tensor
    X_tensor = torch.tensor(X_cliente.values, dtype=torch.float32)
    
    # Obtener las predicciones del modelo
    with torch.no_grad():
        output = modelo(X_tensor.unsqueeze(0))
        _, indices = torch.topk(output, n)
    
    # Obtener los nombres de los productos predichos
    productos_predichos = [nombre_productos[idx] for idx in indices[0].tolist()]
    
    return productos_predichos

# Definir el modelo
class RecomendadorProductos(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RecomendadorProductos, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc(x)
        return out

# Preprocesar datos
X_train, X_test, y_train, y_test, nombre_productos = preprocesar_datos(compras_df, inventario_df)

# Convertir datos a tensores PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.int64)

# Parámetros del modelo
input_dim = X_train.shape[1]
output_dim = len(nombre_productos)

# Crear instancia del modelo
modelo = RecomendadorProductos(input_dim, output_dim)

# Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# Entrenar el modelo
def entrenar_modelo(modelo, X_train_tensor, y_train_tensor, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = modelo(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')

entrenar_modelo(modelo, X_train_tensor, y_train_tensor, criterion, optimizer)

# Recomendar productos para múltiples clientes y guardar en un archivo CSV
clientes = [1, 2, 3, 4, 5]
recomendaciones = []

for cliente in clientes:
    # Filtrar datos del cliente
    X_cliente = X_test[X_test['id_cliente'] == cliente]
    if X_cliente.empty:  # Si no hay datos del cliente en el conjunto de prueba
        X_cliente = X_train[X_train['id_cliente'] == cliente]
    
    if not X_cliente.empty:
        # Obtener recomendaciones para el cliente
        productos_recomendados = recomendar_varios_productos(modelo, X_cliente.iloc[0], nombre_productos, n=5)
        recomendaciones.append({'id_cliente': cliente, 'recomendaciones': productos_recomendados})
        print(f"Recomendaciones para el cliente {cliente}: {productos_recomendados}")
    else:
        print(f"No hay suficientes datos para el cliente {cliente} en los conjuntos de entrenamiento y prueba.")

# Guardar recomendaciones en un archivo CSV
recomendaciones_df = pd.DataFrame(recomendaciones)
recomendaciones_df.to_csv("/home/penguin/Hackaton/data/recomendaciones.csv", index=False)

print("Recomendaciones generadas y guardadas en recomendaciones.csv")




