import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cargar_datos():
    clientes_df = pd.read_csv('data/clientes.csv')
    compras_df = pd.read_csv('data/compras.csv')
    inventario_df = pd.read_csv('data/inventario.csv')
    return clientes_df, compras_df, inventario_df

def recomendar_productos(cliente_id, compras_df, inventario_df):
    # Crear una matriz de conteo de productos comprados
    compras_df['cantidad'] = 1
    matriz_productos = compras_df.pivot_table(index='id_cliente', columns='id_producto', values='cantidad', fill_value=0)

    # Calcular la similitud de coseno entre clientes
    matriz_similitud = cosine_similarity(matriz_productos)

    # Encontrar clientes similares
    cliente_indice = matriz_productos.index.get_loc(cliente_id)
    similitud_cliente = matriz_similitud[cliente_indice]

    # Recomendación de productos basada en clientes similares
    recomendacion_productos = matriz_productos.T.dot(similitud_cliente)
    recomendacion_productos = recomendacion_productos.sort_values(ascending=False)

    return recomendacion_productos.head(10)

if __name__ == "__main__":
    clientes_df, compras_df, inventario_df = cargar_datos()
    cliente_id = 1  # ID del cliente para el cual generar recomendaciones
    productos_recomendados = recomendar_productos(cliente_id, compras_df, inventario_df)
    print("Productos recomendados para el cliente", cliente_id)
    print(productos_recomendados)
