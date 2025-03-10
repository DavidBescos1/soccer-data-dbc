import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
from datetime import datetime
import pdfkit
import plotly.io as pio
import os

# Función para exportar a PDF
def create_pdf_report(title, description, fig_data, filters=None):
    """
    Crea un informe PDF con título, descripción, gráficos y filtros aplicados.
    
    Args:
        title (str): Título del informe
        description (str): Descripción o resumen
        fig_data (list): Lista de tuplas (figura_plotly, título_figura)
        filters (dict, optional): Diccionario de filtros aplicados
        
    Returns:
        bytes: Contenido del PDF en formato bytes
    """
    # Crear directorio temporal para las imágenes si no existe
    temp_dir = os.path.join(os.getcwd(), "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generar HTML para el informe
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{ font-family: 'Arial', sans-serif; margin: 30px; color: #333; }}
            h1 {{ color: #00BFFF; text-align: center; margin-bottom: 20px; }}
            h2 {{ color: #333; border-bottom: 1px solid #00BFFF; padding-bottom: 5px; }}
            .header {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
            .logo {{ max-width: 150px; height: auto; }}
            .fecha {{ text-align: right; color: #666; font-style: italic; }}
            .descripcion {{ margin-bottom: 25px; }}
            .filtros {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .filtro {{ margin: 5px 0; }}
            .figura {{ margin: 25px 0; text-align: center; }}
            .imagen {{ max-width: 100%; height: auto; }}
            .pie-pagina {{ text-align: center; margin-top: 30px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>{title}</h1>
            </div>
            <div class="fecha">
                Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            </div>
        </div>
        
        <div class="descripcion">
            <p>{description}</p>
        </div>
    """
    
    # Añadir filtros si existen
    if filters:
        html_content += '<div class="filtros"><h2>Filtros aplicados</h2>'
        for key, value in filters.items():
            html_content += f'<div class="filtro"><strong>{key}:</strong> {value}</div>'
        html_content += '</div>'
    
    # Añadir figuras
    image_paths = []
    for i, (fig, fig_title) in enumerate(fig_data):
        # Guardar figura como imagen
        img_path = os.path.join(temp_dir, f"figure_{i}.png")
        pio.write_image(fig, img_path, scale=2, width=900, height=500)
        image_paths.append(img_path)
        
        # Añadir imagen al HTML
        html_content += f"""
        <div class="figura">
            <h2>{fig_title}</h2>
            <img src="{img_path}" class="imagen" alt="{fig_title}">
        </div>
        """
    
    # Cerrar HTML
    html_content += """
        <div class="pie-pagina">
            <p>SOCCER DATA DBC - Análisis de Rendimiento Deportivo</p>
        </div>
    </body>
    </html>
    """
    
    # Convertir HTML a PDF
    pdf = pdfkit.from_string(html_content, False)
    
    # Limpiar imágenes temporales
    for img_path in image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)
    
    return pdf

# Componente para el botón de exportación
def export_button(id_prefix, title_default="Informe de Rendimiento"):
    """
    Crea un componente con un botón de exportación a PDF y un modal para configuración.
    
    Args:
        id_prefix (str): Prefijo para los IDs de los componentes
        title_default (str): Título predeterminado para el informe
        
    Returns:
        html.Div: Componente con botón de exportación y modal
    """
    modal_id = f"{id_prefix}-modal"
    button_id = f"{id_prefix}-button"
    title_id = f"{id_prefix}-title"
    description_id = f"{id_prefix}-description"
    download_id = f"{id_prefix}-download"
    
    return html.Div([
        # Botón de exportación
        dbc.Button(
            [html.I(className="fas fa-file-pdf mr-2"), " Exportar a PDF"],
            id=button_id,
            color="info",
            className="mb-3",
            style={"backgroundColor": "#00BFFF"}
        ),
        
        # Modal para configurar el PDF
        dbc.Modal(
            [
                dbc.ModalHeader("Configurar informe PDF"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Título del informe:"),
                                dbc.Input(
                                    id=title_id,
                                    type="text",
                                    value=title_default,
                                    placeholder="Introduce un título para el informe"
                                ),
                            ])
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Descripción:"),
                                dbc.Textarea(
                                    id=description_id,
                                    placeholder="Añade una descripción o notas sobre el informe",
                                    style={"height": "100px"}
                                ),
                            ])
                        ])
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Cancelar", id=f"{modal_id}-close", className="mr-auto"),
                    dbc.Button("Generar PDF", id=f"{modal_id}-generate", color="primary")
                ]),
            ],
            id=modal_id,
            centered=True,
        ),
        
        # Componente para descargar el PDF
        dcc.Download(id=download_id)
    ])

# Callbacks para manejar el modal y la generación de PDF
# NOTA: Estos callbacks deben ser implementados en cada página donde se use el componente
"""
# Ejemplo de callbacks para una página específica:

# Callback para abrir el modal
@callback(
    Output("rendimiento-pdf-modal", "is_open"),
    Input("rendimiento-pdf-button", "n_clicks"),
    Input("rendimiento-pdf-modal-close", "n_clicks"),
    Input("rendimiento-pdf-modal-generate", "n_clicks"),
    State("rendimiento-pdf-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

# Callback para generar y descargar el PDF
@callback(
    Output("rendimiento-pdf-download", "data"),
    Input("rendimiento-pdf-modal-generate", "n_clicks"),
    [State("rendimiento-pdf-title", "value"),
     State("rendimiento-pdf-description", "value"),
     State("filtered-data-store", "data")],  # Aquí obtenemos los datos filtrados
    prevent_initial_call=True
)
def generate_pdf(n_clicks, title, description, filtered_data):
    if not n_clicks:
        raise PreventUpdate
    
    # Obtener las figuras actuales (esto dependerá de cada página)
    fig1 = go.Figure()  # Reemplazar con la figura real
    fig2 = go.Figure()  # Reemplazar con la figura real
    
    # Crear informe PDF
    pdf_data = create_pdf_report(
        title=title or "Informe de Rendimiento",
        description=description or "Análisis de rendimiento de jugadores",
        fig_data=[(fig1, "Gráfico 1"), (fig2, "Gráfico 2")],
        filters={"Posición": "Todas", "Liga": "Todas"}  # Reemplazar con filtros reales
    )
    
    # Devolver los datos para descargar
    return dict(
        content=pdf_data,
        filename=f"informe_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        type="application/pdf"
    )
"""