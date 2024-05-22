import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from jinja2 import Template
from recomendar_productos import recomendar_productos

def cargar_datos():
    # Cargar los datos desde los archivos CSV
    clientes_df = pd.read_csv('data/clientes.csv')
    compras_df = pd.read_csv('data/compras.csv')
    inventario_df = pd.read_csv('data/inventario.csv')
    return clientes_df, compras_df, inventario_df

def enviar_correo(cliente_email, productos_recomendados):
    # Configurar el correo electrónico saliente
    from_email = "iapenguinhackaton@hotmail.com"
    from_password = "ABC45678"  # Por favor, reemplazar por la contraseña real
    subject = "Recomendaciones de Productos Personalizadas"

    # Cargar la plantilla de correo
    with open('templates/email_template.html') as file_:
        template = Template(file_.read())

    # Renderizar la plantilla con los productos recomendados
    email_content = template.render(productos=productos_recomendados.index)

    # Crear el mensaje de correo
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = cliente_email
    msg['Subject'] = subject
    msg.attach(MIMEText(email_content, 'html'))

    # Enviar el correo electrónico usando SMTP
    server = smtplib.SMTP('smtp.office365.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    server.send_message(msg)
    server.quit()

def enviar_correos_a_todos_los_clientes(clientes_df, compras_df, inventario_df):
    # Recorrer todos los clientes y enviar recomendaciones
    for _, cliente in clientes_df.iterrows():
        cliente_id = cliente['id_cliente']
        cliente_email = cliente['correo_electronico']
        
        # Obtener recomendaciones de productos para el cliente actual
        productos_recomendados = recomendar_productos(cliente_id, compras_df, inventario_df)
        
        # Enviar correo con las recomendaciones
        enviar_correo(cliente_email, productos_recomendados)

if __name__ == "__main__":
    # Cargar los datos
    clientes_df, compras_df, inventario_df = cargar_datos()
    
    # Verificar si el nombre de la columna 'correo_electronico' está presente
    if 'correo_electronico' in clientes_df.columns:
        # Enviar correos a todos los clientes
        enviar_correos_a_todos_los_clientes(clientes_df, compras_df, inventario_df)
    else:
        print("El nombre de la columna 'correo_electronico' no está presente en clientes_df.")
