from dash import dcc, html, callback, Input, Output, State
from dash import callback_context as ctx
from dash.exceptions import PreventUpdate
from pdf_export import export_button, create_pdf_report
from components.data_manager import DataManager
from components.sidebar import create_sidebar
from datetime import datetime
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir métricas nutricionales principales (basadas en las columnas del CSV)
metricas_nutricionales = [
    "PROTEINAS_DIARIAS_G",
    "CARBOHIDRATOS_DIARIOS_G",
    "HIDRATACION_DIARIA_L",
    "CALORIAS_DIARIAS",
    "PLAN_NUTRICION"
]

# Layout de la página
layout = html.Div([
    # Sidebar
    create_sidebar(),
    
    # Contenido principal
    html.Div(
        className="content-container fade-in",
        children=[
            # Título
            html.H1("Análisis Nutricional de Jugadores", 
                   style={"color": "#00BFFF", "margin-bottom": "20px"}),
            
            # Filtros
            html.Div(
                className="filter-container",
                children=[
                    # Primera fila de filtros - los 3 dropdowns en la parte superior
                    dbc.Row([
                        # Filtro de posición
                        dbc.Col([
                            html.Div(className="filter-title", children="Posición"),
                            dcc.Dropdown(
                                id="posicion-filter-nutricion",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona posiciones",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de plan nutricional
                        dbc.Col([
                            html.Div(className="filter-title", children="Plan Nutricional"),
                            dcc.Dropdown(
                                id="plan-filter-nutricion",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona plan",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de rango de proteínas
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Proteínas (g)"),
                            dcc.RangeSlider(
                                id="proteinas-range-slider",
                                min=100, max=300, step=10,
                                marks={i: str(i) for i in range(100, 301, 50)},
                                value=[100, 300],
                                className="filter-slider"
                            ),
                        ], width=4),
                    ], className="mb-4"),
                    
                    # Segunda fila de filtros 
                    dbc.Row([
                        # Rango de carbohidratos
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Carbohidratos (g)"),
                            dcc.RangeSlider(
                                id="carbohidratos-range-slider",
                                min=200, max=500, step=10,
                                marks={i: str(i) for i in range(200, 501, 50)},
                                value=[200, 500],
                                className="filter-slider"
                            ),
                        ], width=6),
                        
                        # Rango de calorías
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Calorías"),
                            dcc.RangeSlider(
                                id="calorias-range-slider",
                                min=2000, max=5000, step=100,
                                marks={i: str(i) for i in range(2000, 5001, 500)},
                                value=[2000, 5000],
                                className="filter-slider"
                            ),
                        ], width=6),
                    ], className="mb-4"),
                    
                    # Tercera fila - dropdown de métricas a ancho completo
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="filter-title", children="Métricas Nutricionales"),
                            dcc.Dropdown(
                                id="metricas-filter-nutricion",
                                options=[
                                    {"label": "Proteínas Diarias (g)", "value": "PROTEINAS_DIARIAS_G"},
                                    {"label": "Carbohidratos Diarios (g)", "value": "CARBOHIDRATOS_DIARIOS_G"},
                                    {"label": "Hidratación Diaria (L)", "value": "HIDRATACION_DIARIA_L"},
                                    {"label": "Calorías Diarias", "value": "CALORIAS_DIARIAS"}
                                ],
                                value=["PROTEINAS_DIARIAS_G", "CARBOHIDRATOS_DIARIOS_G", "HIDRATACION_DIARIA_L"],  # Valores por defecto
                                multi=True,
                                placeholder="Selecciona métricas",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Botón de aplicar filtros centrado
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "APLICAR FILTROS",
                                id="aplicar-filtros-btn-nutricion",
                                className="login-button mt-2",
                                style={"width": "200px"}
                            ),
                        ], width={"size": 2, "offset": 5}),
                    ], className="mb-4"),
                ]
            ),
            
            # Botón de exportación PDF
            dbc.Row([
                dbc.Col([
                    export_button("nutricion-jugadores-pdf", "Informe de Nutrición"),
                ], width={"size": 4, "offset": 4}), 
            ], className="mb-4"),
            
            # Contenido de la página
            html.Div(
                id="nutricion-content",
                children=[
                    # Primera fila: Tabla de perfiles nutricionales
                    dbc.Row([
                        # Ranking de perfiles nutricionales a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Perfiles Nutricionales de Jugadores", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="perfiles-nutricionales-table")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Segunda fila: Gráfico de dispersión ocupando todo el ancho
                    dbc.Row([
                        # Gráfico de dispersión a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    # Selector de métricas para el gráfico
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div(className="filter-title", children="Métrica X"),
                                            dcc.Dropdown(
                                                id="scatter-x-metric-nutricion",
                                                options=[],  # Se llenará dinámicamente
                                                placeholder="Selecciona métrica para eje X",
                                                className="filter-dropdown mb-2",
                                                style={"color": "black"}
                                            ),
                                        ], width=5),
                                        dbc.Col([
                                            html.Div(className="filter-title", children="Métrica Y"),
                                            dcc.Dropdown(
                                                id="scatter-y-metric-nutricion",
                                                options=[],  # Se llenará dinámicamente
                                                placeholder="Selecciona métrica para eje Y",
                                                className="filter-dropdown mb-2",
                                                style={"color": "black"}
                                            ),
                                        ], width=5),
                                        dbc.Col([
                                            html.Button(
                                                "ACTUALIZAR GRÁFICO",
                                                id="actualizar-grafico-btn-nutricion",
                                                className="login-button mt-4",
                                                style={"width": "100%"}
                                            ),
                                        ], width=2),
                                    ]),
                                    html.Div(id="scatter-metricas-nutricion")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Tercera fila: Comparativa por plan nutricional
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Comparativa por Plan Nutricional", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="comparativa-planes-nutricion")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Cuarta fila: Distribución de métricas
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Distribución de Métricas Nutricionales", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="distribucion-metricas-nutricion")
                                ]
                            )
                        ], width=12),
                    ]),
                ]
            ),
            
            # Store para los datos filtrados
            dcc.Store(id="filtered-nutricion-data-store"),
        ]
    )
])

# Callback para cargar opciones de filtros iniciales
@callback(
    [Output("posicion-filter-nutricion", "options"),
     Output("plan-filter-nutricion", "options")],
    Input("posicion-filter-nutricion", "id")  # Trigger dummy
)
def load_nutrition_filter_options(dummy):
    # Obtener todos los datos nutricionales
    df_nutrition = data_manager.get_nutrition_data()
    
    # Obtener datos principales para las posiciones (si están disponibles)
    df_main = data_manager.get_data()
    
    # Obtener opciones de posición desde los datos principales
    posiciones = []
    if not df_main.empty and 'POSICIÓN' in df_main.columns:
        posiciones = [{"label": pos, "value": pos} for pos in sorted(df_main['POSICIÓN'].unique())]
    
    # Si no hay posiciones en los datos principales, usar el DataFrame de nutrición si tiene esa columna
    if not posiciones and not df_nutrition.empty and 'POSICIÓN' in df_nutrition.columns:
        posiciones = [{"label": pos, "value": pos} for pos in sorted(df_nutrition['POSICIÓN'].unique())]
    
    # Obtener opciones de plan nutricional
    planes = []
    if not df_nutrition.empty and 'PLAN_NUTRICION' in df_nutrition.columns:
        # Asegurarse de eliminar valores nulos o vacíos
        planes_unicos = [plan for plan in df_nutrition['PLAN_NUTRICION'].dropna().unique() if plan and str(plan).strip()]
        # Crear opciones para el dropdown
        planes = [{"label": plan, "value": plan} for plan in sorted(planes_unicos)]
    
    return posiciones, planes

# Callback para actualizar opciones de métricas en los dropdowns del gráfico
@callback(
    [Output("scatter-x-metric-nutricion", "options"),
     Output("scatter-y-metric-nutricion", "options"),
     Output("scatter-x-metric-nutricion", "value"),
     Output("scatter-y-metric-nutricion", "value")],
    Input("metricas-filter-nutricion", "value")
)
def update_scatter_metric_options_nutricion(selected_metrics):
    # Si no hay métricas seleccionadas, mostrar mensaje de error
    if not selected_metrics or len(selected_metrics) == 0:
        options = [{"label": "Selecciona métricas en el filtro", "value": ""}]
        return options, options, None, None
    
    # Crear opciones solo para las métricas seleccionadas
    options = []
    for metric in selected_metrics:
        # Crear etiquetas más legibles para las métricas
        if metric == "PROTEINAS_DIARIAS_G":
            label = "Proteínas Diarias (g)"
        elif metric == "CARBOHIDRATOS_DIARIOS_G":
            label = "Carbohidratos Diarios (g)"
        elif metric == "HIDRATACION_DIARIA_L":
            label = "Hidratación Diaria (L)"
        elif metric == "CALORIAS_DIARIAS":
            label = "Calorías Diarias"
        else:
            label = metric.replace("_", " ").title()
        options.append({"label": label, "value": metric})
    
    # Seleccionar por defecto las dos primeras métricas seleccionadas
    x_default = selected_metrics[0] if len(selected_metrics) > 0 else None
    y_default = selected_metrics[1] if len(selected_metrics) > 1 else selected_metrics[0]
    
    return options, options, x_default, y_default

# Callback para aplicar filtros y actualizar datos
@callback(
    Output("filtered-nutricion-data-store", "data"),
    [Input("aplicar-filtros-btn-nutricion", "n_clicks")],
    [State("posicion-filter-nutricion", "value"),
     State("plan-filter-nutricion", "value"),
     State("proteinas-range-slider", "value"),
     State("carbohidratos-range-slider", "value"),
     State("calorias-range-slider", "value"),
     State("metricas-filter-nutricion", "value")],
    prevent_initial_call=False
)
def filter_nutrition_data(n_clicks, posiciones, planes, rango_proteinas, rango_carbohidratos, rango_calorias, metricas_seleccionadas):
    # Obtener datos de nutrición
    df_nutrition = data_manager.get_nutrition_data()
    
    if df_nutrition.empty:
        return {"data": [], "selected_metrics": []}
    
    # Crear una copia para filtrarlo
    filtered_df = df_nutrition.copy()
    
    # Si no es la carga inicial (se ha hecho clic en el botón), aplicar todos los filtros
    if n_clicks:
        # Aplicar filtro de posición si hay datos de posición disponibles
        if posiciones and len(posiciones) > 0:
            # Obtener datos principales para cruzar con posiciones
            df_main = data_manager.get_data()
            
            if not df_main.empty and 'NOMBRE' in df_main.columns and 'POSICIÓN' in df_main.columns:
                # Crear un diccionario de posiciones por jugador
                posicion_dict = df_main.set_index('NOMBRE')['POSICIÓN'].to_dict()
                
                # Filtrar jugadores de nutrición que existen en el diccionario de posiciones
                # y tienen una de las posiciones seleccionadas
                filtered_df = filtered_df[filtered_df['NOMBRE'].apply(
                    lambda x: x in posicion_dict and posicion_dict[x] in posiciones)]
        
        # Aplicar filtro de plan nutricional
        if planes and len(planes) > 0 and 'PLAN_NUTRICION' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['PLAN_NUTRICION'].isin(planes)]
        
        # Aplicar filtros de rangos numéricos
        if 'PROTEINAS_DIARIAS_G' in filtered_df.columns:
            min_proteinas, max_proteinas = rango_proteinas
            filtered_df = filtered_df[(filtered_df['PROTEINAS_DIARIAS_G'] >= min_proteinas) & 
                                     (filtered_df['PROTEINAS_DIARIAS_G'] <= max_proteinas)]
        
        if 'CARBOHIDRATOS_DIARIOS_G' in filtered_df.columns:
            min_carbohidratos, max_carbohidratos = rango_carbohidratos
            filtered_df = filtered_df[(filtered_df['CARBOHIDRATOS_DIARIOS_G'] >= min_carbohidratos) & 
                                     (filtered_df['CARBOHIDRATOS_DIARIOS_G'] <= max_carbohidratos)]
        
        if 'CALORIAS_DIARIAS' in filtered_df.columns:
            min_calorias, max_calorias = rango_calorias
            filtered_df = filtered_df[(filtered_df['CALORIAS_DIARIAS'] >= min_calorias) & 
                                     (filtered_df['CALORIAS_DIARIAS'] <= max_calorias)]
    
    # Seleccionar columnas básicas siempre necesarias
    cols_to_include = ['NOMBRE']
    
    # Incluir columna de plan nutricional si está disponible
    if 'PLAN_NUTRICION' in filtered_df.columns:
        cols_to_include.append('PLAN_NUTRICION')
    
    # Incluir solo las métricas nutricionales seleccionadas
    if metricas_seleccionadas and len(metricas_seleccionadas) > 0:
        # Añadir solo las métricas seleccionadas que existen en el DataFrame
        for col in metricas_seleccionadas:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    else:
        # Si no hay métricas seleccionadas, incluir todas las disponibles
        for col in metricas_nutricionales:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    
    # Seleccionar solo columnas disponibles y eliminar duplicados
    cols_to_include = list(dict.fromkeys([col for col in cols_to_include if col in filtered_df.columns]))
    
    # Seleccionar solo las columnas deseadas y filtrar filas con valores faltantes
    filtered_df = filtered_df[cols_to_include]
    filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 1)  # Permitir algunos valores faltantes
    
    # Guardar métricas seleccionadas en los datos (para usar en otros callbacks)
    selected_metrics_data = {
        'data': filtered_df.to_dict('records'),
        'selected_metrics': metricas_seleccionadas if metricas_seleccionadas else metricas_nutricionales
    }
    
    return selected_metrics_data

# Callback para crear tabla de perfiles nutricionales
@callback(
    Output("perfiles-nutricionales-table", "children"),
    Input("filtered-nutricion-data-store", "data")
)
def create_perfiles_nutricionales_table(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_nutricionales
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.",
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # IMPORTANTE: Limitar el DataFrame a solo 10 filas por defecto, agrupados por plan nutricional
    # Primero ordenamos por plan nutricional, luego tomamos los primeros 10
    if 'PLAN_NUTRICION' in df.columns:
        # Obtener hasta 2 jugadores por cada plan nutricional
        plans = df['PLAN_NUTRICION'].unique()
        limited_df = pd.DataFrame()
        for plan in plans:
            plan_players = df[df['PLAN_NUTRICION'] == plan].head(2)
            limited_df = pd.concat([limited_df, plan_players])
            
            # Si ya tenemos 10 jugadores, detenemos el proceso
            if len(limited_df) >= 10:
                limited_df = limited_df.head(10)
                break
                
        # Si no llegamos a 10 jugadores, añadimos más hasta completar
        if len(limited_df) < 10 and len(df) > 10:
            remaining = df[~df['NOMBRE'].isin(limited_df['NOMBRE'])].head(10 - len(limited_df))
            limited_df = pd.concat([limited_df, remaining])
    else:
        # Si no hay columna de plan nutricional, simplemente limitamos a 10
        limited_df = df.head(10)
    
    # Usamos limited_df en lugar de df para el resto de la función
    df = limited_df
    
    # Seleccionar columnas para mostrar
    display_cols = ['NOMBRE']
    
    # Añadir plan nutricional si está disponible
    if 'PLAN_NUTRICION' in df.columns:
        display_cols.append('PLAN_NUTRICION')
    
    # Añadir las métricas seleccionadas disponibles
    metrics_to_show = [m for m in selected_metrics if m in df.columns]
    display_cols.extend(metrics_to_show)
    
    # Crear los encabezados de la tabla
    header_cells = []
    for col in display_cols:
        # Formatear nombres de columnas para mostrar
        if col == "PROTEINAS_DIARIAS_G":
            header_text = "Proteínas (g)"
        elif col == "CARBOHIDRATOS_DIARIOS_G":
            header_text = "Carbohidratos (g)"
        elif col == "HIDRATACION_DIARIA_L":
            header_text = "Hidratación (L)"
        elif col == "CALORIAS_DIARIAS":
            header_text = "Calorías"
        elif col == "PLAN_NUTRICION":
            header_text = "Plan Nutricional"
        else:
            header_text = col
        
        header_cells.append(
            html.Th(header_text, style={
                'backgroundColor': '#00BFFF',
                'color': 'white',
                'padding': '10px',
                'textAlign': 'center',
                'fontWeight': 'bold'
            })
        )
    
    header_row = html.Tr(header_cells)
    
    # Crear las filas de la tabla
    table_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Alternar colores de fondo
        bg_color = '#0E2B3D' if i % 2 == 0 else '#1A1A1A'
        row_cells = []
        
        # Añadir celdas con datos
        for col in display_cols:
            value = row[col]
            # Formatear valores correctamente
            if isinstance(value, (int, float)) and not pd.isna(value):
                if col == 'PROTEINAS_DIARIAS_G' or col == 'CARBOHIDRATOS_DIARIOS_G':
                    formatted = f"{value:.1f} g"
                elif col == 'HIDRATACION_DIARIA_L':
                    formatted = f"{value:.1f} L"
                elif col == 'CALORIAS_DIARIAS':
                    formatted = f"{int(value)}"
                else:
                    formatted = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
            else:
                formatted = str(value) if not pd.isna(value) else "-"
            
            # Crear celda
            row_cells.append(html.Td(formatted, style={'padding': '10px'}))
        
        # Añadir fila a la tabla
        table_rows.append(html.Tr(row_cells, style={'backgroundColor': bg_color}))
    
    # Añadir texto informativo sobre la limitación
    info_text = html.Div(
        f"Mostrando {len(df)} de {len(raw_data)} jugadores. Utiliza los filtros para ver más resultados específicos.",
        style={"font-style": "italic", "text-align": "center", "margin-top": "10px", "font-size": "12px", "color": "#ccc"}
    )
    
    # Crear tabla completa
    table = html.Table(
        [html.Thead(header_row), html.Tbody(table_rows)],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'color': 'white'
        }
    )
    
    return html.Div([table, info_text])

# Callback para crear gráfico de dispersión
@callback(
    Output("scatter-metricas-nutricion", "children"),
    [Input("actualizar-grafico-btn-nutricion", "n_clicks"),
     Input("filtered-nutricion-data-store", "data")],
    [State("scatter-x-metric-nutricion", "value"),
     State("scatter-y-metric-nutricion", "value")]
)
def create_scatter_plot_nutricion(n_clicks, filtered_data, x_metric, y_metric):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.",
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Verificar si se han proporcionado las métricas X e Y
    if not x_metric or not y_metric:
        return html.Div("Selecciona métricas en el filtro y haz clic en 'ACTUALIZAR GRÁFICO'", 
                       style={"text-align": "center", "padding": "20px", "font-weight": "bold", "color": "#00BFFF"})
    
    # Verificar métricas disponibles
    if x_metric in df.columns and y_metric in df.columns:
        # Crear etiquetas legibles para los ejes
        x_label = "Proteínas Diarias (g)" if x_metric == "PROTEINAS_DIARIAS_G" else \
                 "Carbohidratos Diarios (g)" if x_metric == "CARBOHIDRATOS_DIARIOS_G" else \
                 "Hidratación Diaria (L)" if x_metric == "HIDRATACION_DIARIA_L" else \
                 "Calorías Diarias" if x_metric == "CALORIAS_DIARIAS" else \
                 x_metric.replace("_", " ").title()
        
        y_label = "Proteínas Diarias (g)" if y_metric == "PROTEINAS_DIARIAS_G" else \
                 "Carbohidratos Diarios (g)" if y_metric == "CARBOHIDRATOS_DIARIOS_G" else \
                 "Hidratación Diaria (L)" if y_metric == "HIDRATACION_DIARIA_L" else \
                 "Calorías Diarias" if y_metric == "CALORIAS_DIARIAS" else \
                 y_metric.replace("_", " ").title()
        
        # Agregar columna de plan nutricional como color si está disponible
        color_column = "PLAN_NUTRICION" if "PLAN_NUTRICION" in df.columns else None
        
        # Crear gráfico de dispersión
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            color=color_column,
            hover_name='NOMBRE',
            template="plotly_dark",
            labels={
                x_metric: x_label,
                y_metric: y_label,
                'PLAN_NUTRICION': 'Plan Nutricional'
            },
            opacity=0.9,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Personalizar hover
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         f"{x_label}: %{{x:.1f}}<br>" +
                         f"{y_label}: %{{y:.1f}}<br>" +
                         ("%{marker.color}<extra></extra>" if color_column else "<extra></extra>")
        )
        
        # Mejorar el diseño del gráfico
        fig.update_layout(
            title={
                'text': f"{x_label} vs. {y_label}",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': 'white', 'family': 'Arial, sans-serif'}
            },
            font=dict(
                family="Arial, sans-serif",
                size=13,
                color="white"
            ),
            legend=dict(
                title="Plan Nutricional" if color_column else "",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text=x_label, font=dict(size=16, color="white")),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
                zerolinewidth=1
            ),
            yaxis=dict(
                title=dict(text=y_label, font=dict(size=16, color="white")),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.1)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.2)",
                zerolinewidth=1
            ),
            height=550,
            margin=dict(l=10, r=10, t=80, b=10)
        )
        
        # Agregar líneas de referencia de cuadrantes (valores medios)
        x_mean = df[x_metric].mean()
        y_mean = df[y_metric].mean()
        
        # Línea vertical en la media de x
        fig.add_shape(
            type="line",
            x0=x_mean,
            y0=df[y_metric].min(),
            x1=x_mean,
            y1=df[y_metric].max(),
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
        )
        
        # Línea horizontal en la media de y
        fig.add_shape(
            type="line",
            x0=df[x_metric].min(),
            y0=y_mean,
            x1=df[x_metric].max(),
            y1=y_mean,
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
        )
        
        # Añadir anotaciones para los cuadrantes
        fig.add_annotation(
            x=df[x_metric].max() - (df[x_metric].max() - df[x_metric].min())*0.1,
            y=df[y_metric].max() - (df[y_metric].max() - df[y_metric].min())*0.1,
            text="Alto rendimiento",
            showarrow=False,
            font=dict(color="rgba(0, 255, 0, 0.7)", size=12)
        )
        
        # Añadir una explicación del gráfico
        explanation = html.Div([
            html.H4("Análisis de dispersión", style={"color": "#00BFFF", "margin-bottom": "10px"}),
            html.P([
                "Este gráfico muestra la relación entre ", 
                html.Strong(x_label), 
                " y ", 
                html.Strong(y_label),
                " para cada jugador. ",
                "Los colores representan los diferentes planes nutricionales." if color_column else ""
            ]),
            html.P([
                "Las líneas punteadas muestran los valores promedio, dividiendo el gráfico en cuadrantes.",
                " Los jugadores en el cuadrante superior derecho tienen valores por encima del promedio en ambas métricas."
            ]),
        ], style={"margin-top": "20px", "padding": "15px", "background-color": "rgba(0,0,0,0.2)", "border-radius": "5px"})
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False}),
            explanation
        ])
    else:
        return html.Div(f"Las métricas seleccionadas no están disponibles en los datos filtrados.",
                       style={"text-align": "center", "padding": "20px", "color": "orange"})

# Callback para crear comparativa por plan nutricional
@callback(
    Output("comparativa-planes-nutricion", "children"),
    Input("filtered-nutricion-data-store", "data")
)
def create_comparativa_planes(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_nutricionales
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    # Verificar si existe la columna PLAN_NUTRICION
    if df.empty or 'PLAN_NUTRICION' not in df.columns:
        return html.Div("No hay datos de plan nutricional disponibles para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Seleccionar métricas disponibles de entre las seleccionadas
    metrics = [m for m in selected_metrics if m in df.columns]
    
    # Limitar a 3 métricas para mejor visualización
    metrics = metrics[:3]
    
    if not metrics:
        return html.Div("No hay métricas nutricionales disponibles para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Formatear nombres de métricas para mostrar
    formatted_metrics = []
    for m in metrics:
        if m == "PROTEINAS_DIARIAS_G":
            formatted_metrics.append("Proteínas (g)")
        elif m == "CARBOHIDRATOS_DIARIOS_G":
            formatted_metrics.append("Carbohidratos (g)")
        elif m == "HIDRATACION_DIARIA_L":
            formatted_metrics.append("Hidratación (L)")
        elif m == "CALORIAS_DIARIAS":
            formatted_metrics.append("Calorías")
        else:
            formatted_metrics.append(m.replace("_", " ").title())
    
    # Agrupar por plan y calcular la media de las métricas
    plan_stats = df.groupby('PLAN_NUTRICION')[metrics].mean().reset_index()
    
    # Definir colores personalizados
    custom_colors = ['#00BFFF', '#4CAF50', '#FFFF00', '#FF5733']
    
    # Crear gráfico de barras para comparar planes
    fig = go.Figure()
    
    for i, (metric, formatted_metric) in enumerate(zip(metrics, formatted_metrics)):
        fig.add_trace(go.Bar(
            x=plan_stats['PLAN_NUTRICION'],
            y=plan_stats[metric],
            name=formatted_metric,
            marker_color=custom_colors[i % len(custom_colors)],
            hovertemplate='<b>%{x}</b><br>' +
                         f'{formatted_metric}: %{{y:.1f}}<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='group',
        title={
            'text': "Comparativa de Métricas por Plan Nutricional",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'white', 'family': 'Arial, sans-serif'}
        },
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Plan Nutricional", font=dict(size=14, color="white")),
            tickangle=0,
            tickfont=dict(size=12),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text="Valor Promedio", font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(size=12)
        ),
        height=450,
        margin=dict(l=10, r=10, t=80, b=50)
    )
    
    # Añadir un título y explicación
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.P("Este gráfico muestra el promedio por plan nutricional para las métricas seleccionadas.",
                 style={"font-style": "italic", "margin-top": "10px"})
        ])
    ])

# Callback para crear distribución de métricas nutricionales
@callback(
    Output("distribucion-metricas-nutricion", "children"),
    Input("filtered-nutricion-data-store", "data")
)
def create_distribucion_metricas_nutricion(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_nutricionales
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.",
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Seleccionar la primera métrica disponible de las seleccionadas
    metric = None
    for m in selected_metrics:
        if m in df.columns:
            metric = m
            break
    
    if not metric:
        return html.Div("No hay métricas nutricionales disponibles para mostrar la distribución.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Formatear nombre de la métrica para mostrar
    if metric == "PROTEINAS_DIARIAS_G":
        metric_title = "Proteínas Diarias (g)"
    elif metric == "CARBOHIDRATOS_DIARIOS_G":
        metric_title = "Carbohidratos Diarios (g)"
    elif metric == "HIDRATACION_DIARIA_L":
        metric_title = "Hidratación Diaria (L)"
    elif metric == "CALORIAS_DIARIAS":
        metric_title = "Calorías Diarias"
    else:
        metric_title = metric.replace("_", " ").title()
    
    # Crear histograma
    color_column = "PLAN_NUTRICION" if "PLAN_NUTRICION" in df.columns else None
    
    fig = px.histogram(
        df,
        x=metric,
        color=color_column,
        title=f"Distribución de {metric_title}",
        template="plotly_dark",
        marginal="box",
        opacity=0.7,
        labels={
            metric: metric_title,
            'PLAN_NUTRICION': 'Plan Nutricional'
        },
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Añadir línea de valor medio
    mean_value = df[metric].mean()
    
    fig.add_shape(
        type="line",
        x0=mean_value,
        x1=mean_value,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=mean_value,
        y=0.95,
        yref="paper",
        text=f"Media: {mean_value:.1f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="red",
        borderwidth=1
    )
    
    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text=metric_title, font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            title=dict(text="Número de Jugadores", font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(
            title="Plan Nutricional" if color_column else "",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(size=11)
        ),
        height=450,
        margin=dict(l=10, r=10, t=50, b=50)
    )
    
    # Añadir un título y explicación
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.P([
                "Este histograma muestra la distribución de ",
                html.Strong(metric_title),
                " entre los jugadores. La línea roja representa el valor promedio."
            ], style={"font-style": "italic", "margin-top": "10px"})
        ])
    ])

# Callback para exportar a PDF
@callback(
    Output("nutricion-pdf-download", "data"),
    Input("nutricion-pdf-modal-generate", "n_clicks"),
    [State("nutricion-pdf-title", "value"),
     State("nutricion-pdf-description", "value"),
     State("filtered-nutricion-data-store", "data")],
    prevent_initial_call=True
)
def generate_nutricion_pdf(n_clicks, title, description, filtered_data):
    if not n_clicks:
        raise PreventUpdate
    
    import base64
    import io
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # Crear un buffer para guardar el PDF
    buffer = io.BytesIO()
    
    # Crear el documento PDF con orientación horizontal
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                           leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Lista para los elementos del PDF
    elements = []
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Crear estilo personalizado para el título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        textColor=colors.cyan,
        alignment=1,  # Centrado
        fontSize=28,
        fontName='Helvetica-Bold',
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        textColor=colors.white,
        fontSize=14,
        fontName='Helvetica-Bold',
        alignment=1  # Centrado
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        textColor=colors.white,
        fontSize=12,
        fontName='Helvetica'
    )
    
    # Agregar título
    elements.append(Paragraph(title or "Informe de Nutrición", title_style))
    elements.append(Spacer(1, 15))
    
    # Agregar fecha
    from datetime import datetime
    elements.append(Paragraph(f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Agregar descripción si existe
    if description:
        elements.append(Paragraph(description, normal_style))
        elements.append(Spacer(1, 10))
    
    # Extraer datos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        df = pd.DataFrame(filtered_data['data'])
        metrics = filtered_data.get('selected_metrics', [])
    else:
        df = pd.DataFrame()
        metrics = []
    
    # Agregar filtros aplicados
    if metrics:
        metrics_text = f"Métricas analizadas: {', '.join(metrics[:3])}"
        if len(metrics) > 3:
            metrics_text += "..."
        metrics_text += f" | Jugadores analizados: {len(df)}"
        
        elements.append(Paragraph(metrics_text, ParagraphStyle(
            'CustomFilter',
            parent=styles['Normal'],
            textColor=colors.yellow,
            fontSize=11,
            fontName='Helvetica',
            alignment=1  # Centrado
        )))
        elements.append(Spacer(1, 15))
    
    # Crear tabla de perfiles nutricionales
    if not df.empty:
        # Seleccionar columnas de interés
        headers = ["Nombre"]
        
        # Añadir plan nutricional si existe
        if 'PLAN_NUTRICION' in df.columns:
            headers.append("Plan Nutricional")
        
        # Añadir métricas nutricionales principales
        metrics_to_show = []
        metrics_headers = []
        
        for metric in metrics[:4]:  # Limitar a 4 métricas
            if metric in df.columns:
                metrics_to_show.append(metric)
                
                # Formatear nombres de columnas
                if metric == "PROTEINAS_DIARIAS_G":
                    metrics_headers.append("Proteínas (g)")
                elif metric == "CARBOHIDRATOS_DIARIOS_G":
                    metrics_headers.append("Carbohidratos (g)")
                elif metric == "HIDRATACION_DIARIA_L":
                    metrics_headers.append("Hidratación (L)")
                elif metric == "CALORIAS_DIARIAS":
                    metrics_headers.append("Calorías")
                else:
                    metrics_headers.append(metric.replace("_", " ").title())
        
        headers.extend(metrics_headers)
        
        # Preparar datos para la tabla
        table_data = [headers]
        
        # Añadir filas de datos
        for _, row in df.head(15).iterrows():  # Limitar a 15 jugadores
            row_data = [row["NOMBRE"]]
            
            # Añadir plan nutricional si existe
            if 'PLAN_NUTRICION' in df.columns:
                row_data.append(row["PLAN_NUTRICION"])
            
            # Añadir métricas
            for metric in metrics_to_show:
                value = row[metric]
                if isinstance(value, (int, float)):
                    if metric == "PROTEINAS_DIARIAS_G" or metric == "CARBOHIDRATOS_DIARIOS_G":
                        row_data.append(f"{value:.1f} g")
                    elif metric == "HIDRATACION_DIARIA_L":
                        row_data.append(f"{value:.1f} L")
                    elif metric == "CALORIAS_DIARIAS":
                        row_data.append(f"{int(value)}")
                    else:
                        row_data.append(f"{value:.1f}")
                else:
                    row_data.append(str(value) if not pd.isna(value) else "-")
            
            table_data.append(row_data)
        
        # Crear tabla con anchos personalizados
        col_widths = [120]  # Ancho para nombre
        
        if 'PLAN_NUTRICION' in df.columns:
            col_widths.append(150)  # Ancho para plan nutricional
        
        # Ancho para métricas
        for _ in metrics_to_show:
            col_widths.append(90)
        
        table = Table(table_data, colWidths=col_widths)
        
        # Estilo de tabla
        table_style = [
            # Cabecera
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Cuerpo de tabla
            ('BACKGROUND', (0, 1), (-1, -1), colors.navy),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 1), (1, -1), 'LEFT'),      # Alineación izquierda para nombre y plan
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'),   # Alineación central para métricas
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
        ]
        
        # Alternar colores de filas
        for i in range(1, len(table_data), 2):
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.darkblue))
        
        table.setStyle(TableStyle(table_style))
        elements.append(table)
    else:
        elements.append(Paragraph("No hay datos disponibles", normal_style))
    
    # Agregar pie de página
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("SOCCER DATA DBC - Análisis Nutricional", 
                             ParagraphStyle('Footer', parent=styles['Normal'],
                                           textColor=colors.cyan, alignment=1, fontSize=10)))
    
    # Crear una clase para manejar el fondo negro
    class BlackBackground(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            SimpleDocTemplate.__init__(self, *args, **kwargs)
        
        def beforePage(self):
            self.canv.setFillColor(colors.black)
            self.canv.rect(0, 0, self.width + 60, self.height + 60, fill=1)
    
    # Usar nuestra clase personalizada
    doc = BlackBackground(buffer, pagesize=landscape(letter),
                         leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Construir el PDF
    doc.build(elements)
    
    # Obtener los datos del buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Codificar en base64 para enviarlo
    pdf_base64 = base64.b64encode(pdf_data).decode('ascii')
    
    # Devolver el PDF para descarga
    return {
        "content": pdf_base64,
        "filename": f"informe_nutricion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }

# Callback para abrir el modal de PDF
@callback(
    Output("nutricion-jugadores-pdf-modal", "is_open"),
    [Input("nutricion-jugadores-pdf-button", "n_clicks"),
     Input("nutricion-jugadores-pdf-modal-close", "n_clicks"),
     Input("nutricion-jugadores-pdf-modal-generate", "n_clicks")],
    [State("nutricion-jugadores-pdf-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_nutricion_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

# Callback para generar y descargar el PDF
@callback(
    Output("nutricion-jugadores-pdf-download", "data"),
    Input("nutricion-jugadores-pdf-modal-generate", "n_clicks"),
    [State("nutricion-jugadores-pdf-title", "value"),
     State("nutricion-jugadores-pdf-description", "value"),
     State("filtered-nutricion-data-store", "data")],
    prevent_initial_call=True
)
def generate_nutricion_pdf(n_clicks, title, description, filtered_data):
    if not n_clicks:
        raise PreventUpdate
    
    import base64
    import io
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # Crear un buffer para guardar el PDF
    buffer = io.BytesIO()
    
    # Crear el documento PDF con orientación horizontal
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), 
                           leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Lista para los elementos del PDF
    elements = []
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Crear estilo personalizado para el título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        textColor=colors.cyan,
        alignment=1,  # Centrado
        fontSize=28,
        fontName='Helvetica-Bold',
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        textColor=colors.white,
        fontSize=14,
        fontName='Helvetica-Bold',
        alignment=1  # Centrado
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        textColor=colors.white,
        fontSize=12,
        fontName='Helvetica'
    )
    
    # Agregar título
    elements.append(Paragraph(title or "Informe de Nutrición", title_style))
    elements.append(Spacer(1, 15))
    
    # Agregar fecha
    from datetime import datetime
    elements.append(Paragraph(f"Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Agregar descripción si existe
    if description:
        elements.append(Paragraph(description, normal_style))
        elements.append(Spacer(1, 10))
    
    # Extraer datos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        df = pd.DataFrame(filtered_data['data'])
        metrics = filtered_data.get('selected_metrics', [])
    else:
        df = pd.DataFrame()
        metrics = []
    
    # Agregar filtros aplicados
    if metrics:
        metrics_text = f"Métricas analizadas: {', '.join(metrics[:3])}"
        if len(metrics) > 3:
            metrics_text += "..."
        metrics_text += f" | Jugadores analizados: {len(df)}"
        
        elements.append(Paragraph(metrics_text, ParagraphStyle(
            'CustomFilter',
            parent=styles['Normal'],
            textColor=colors.yellow,
            fontSize=11,
            fontName='Helvetica',
            alignment=1  # Centrado
        )))
        elements.append(Spacer(1, 15))
    
    # Crear tabla de perfiles nutricionales
    if not df.empty:
        # Seleccionar columnas de interés
        headers = ["Nombre"]
        
        # Añadir plan nutricional si existe
        if 'PLAN_NUTRICION' in df.columns:
            headers.append("Plan Nutricional")
        
        # Añadir métricas nutricionales principales
        metrics_to_show = []
        metrics_headers = []
        
        for metric in metrics[:4]:  # Limitar a 4 métricas
            if metric in df.columns:
                metrics_to_show.append(metric)
                
                # Formatear nombres de columnas
                if metric == "PROTEINAS_DIARIAS_G":
                    metrics_headers.append("Proteínas (g)")
                elif metric == "CARBOHIDRATOS_DIARIOS_G":
                    metrics_headers.append("Carbohidratos (g)")
                elif metric == "HIDRATACION_DIARIA_L":
                    metrics_headers.append("Hidratación (L)")
                elif metric == "CALORIAS_DIARIAS":
                    metrics_headers.append("Calorías")
                else:
                    metrics_headers.append(metric.replace("_", " ").title())
        
        headers.extend(metrics_headers)
        
        # Preparar datos para la tabla
        table_data = [headers]
        
        # Añadir filas de datos
        for _, row in df.head(15).iterrows():  # Limitar a 15 jugadores
            row_data = [row["NOMBRE"]]
            
            # Añadir plan nutricional si existe
            if 'PLAN_NUTRICION' in df.columns:
                row_data.append(row["PLAN_NUTRICION"])
            
            # Añadir métricas
            for metric in metrics_to_show:
                value = row[metric]
                if isinstance(value, (int, float)):
                    if metric == "PROTEINAS_DIARIAS_G" or metric == "CARBOHIDRATOS_DIARIOS_G":
                        row_data.append(f"{value:.1f} g")
                    elif metric == "HIDRATACION_DIARIA_L":
                        row_data.append(f"{value:.1f} L")
                    elif metric == "CALORIAS_DIARIAS":
                        row_data.append(f"{int(value)}")
                    else:
                        row_data.append(f"{value:.1f}")
                else:
                    row_data.append(str(value) if not pd.isna(value) else "-")
            
            table_data.append(row_data)
        
        # Crear tabla con anchos personalizados
        col_widths = [120]  # Ancho para nombre
        
        if 'PLAN_NUTRICION' in df.columns:
            col_widths.append(150)  # Ancho para plan nutricional
        
        # Ancho para métricas
        for _ in metrics_to_show:
            col_widths.append(90)
        
        table = Table(table_data, colWidths=col_widths)
        
        # Estilo de tabla
        table_style = [
            # Cabecera
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            
            # Cuerpo de tabla
            ('BACKGROUND', (0, 1), (-1, -1), colors.navy),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 1), (1, -1), 'LEFT'),      # Alineación izquierda para nombre y plan
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'),   # Alineación central para métricas
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.darkblue),
        ]
        
        # Alternar colores de filas
        for i in range(1, len(table_data), 2):
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.darkblue))
        
        table.setStyle(TableStyle(table_style))
        elements.append(table)
    else:
        elements.append(Paragraph("No hay datos disponibles", normal_style))
    
    # Agregar pie de página
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("SOCCER DATA DBC - Análisis Nutricional", 
                             ParagraphStyle('Footer', parent=styles['Normal'],
                                           textColor=colors.cyan, alignment=1, fontSize=10)))
    
    # Crear una clase para manejar el fondo negro
    class BlackBackground(SimpleDocTemplate):
        def __init__(self, *args, **kwargs):
            SimpleDocTemplate.__init__(self, *args, **kwargs)
        
        def beforePage(self):
            self.canv.setFillColor(colors.black)
            self.canv.rect(0, 0, self.width + 60, self.height + 60, fill=1)
    
    # Usar nuestra clase personalizada
    doc = BlackBackground(buffer, pagesize=landscape(letter),
                         leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    
    # Construir el PDF
    doc.build(elements)
    
    # Obtener los datos del buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Codificar en base64 para enviarlo
    pdf_base64 = base64.b64encode(pdf_data).decode('ascii')
    
    # Devolver el PDF para descarga
    return {
        "content": pdf_base64,
        "filename": f"informe_nutricion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }