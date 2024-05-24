from flask import Flask, render_template, redirect, url_for
from cargar_datos import cargar_datos
from analizar_habitos import analizar_habitos, visualizar_datos
from recomendar_productos import recomendar_varios_productos
from enviar_correos import enviar_correo

app = Flask(__name__)

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para cargar datos
@app.route('/cargar_datos')
def cargar_datos_route():
    clientes_df, compras_df, inventario_df = cargar_datos()
    # Aquí procesamos los datos y los pasamos a la plantilla
    # Por ejemplo:
    # procesar_datos(clientes_df, compras_df, inventario_df)
    return render_template('cargar_datos.html', mensaje="¡Datos cargados correctamente!")

# Ruta para analizar hábitos
@app.route('/analizar_habitos')
def analizar_habitos_route():
    clientes_df, compras_df, inventario_df = cargar_datos()
    frecuencia_compra, valor_medio_compra, productos_populares = analizar_habitos(clientes_df, compras_df, inventario_df)
    visualizar_datos(frecuencia_compra, valor_medio_compra, productos_populares)
    # Aquí pasamos los resultados del análisis a la plantilla
    return render_template('analizar_habitos.html', mensaje="¡Hábitos analizados correctamente!")

# Ruta para recomendar productos
@app.route('/recomendar_productos')
def recomendar_productos_route():
    # Aquí llamamos a la función para recomendar productos
    # y pasamos los resultados a la plantilla
    return render_template('recomendar_productos.html', mensaje="¡Productos recomendados correctamente!")

# Ruta para enviar correos
@app.route('/enviar_correos')
def enviar_correos_route():
    # Aquí llamamos a la función para enviar correos
    # y pasamos los resultados a la plantilla
    return render_template('enviar_correos.html', mensaje="¡Correos enviados correctamente!")

# Ruta para regresar a la página anterior
@app.route('/regresar')
def regresar():
    # Redirigimos a la página principal
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

