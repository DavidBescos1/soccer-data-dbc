from dash import dcc, html, callback, Input, Output, State, ALL
from pdf_export import export_button, create_pdf_report
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64

from components.data_manager import DataManager
from components.sidebar import create_sidebar

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir métricas ofensivas principales
metricas_ofensivas = [
    "TIROS A PORTERÍA", 
    "TIROS FUERA", 
    "TIROS BLOQUEADOS", 
    "TIROS EXITOSOS",
    "TIROS LARGA DISTANCIA", 
    "TIROS MEDIA DISTANCIA", 
    "TIROS CORTA DISTANCIA",
    "REMATES CABEZA", 
    "TIROS UNO CONTRA UNO", 
    "PENALTIS TIRADOS", 
    "TIROS LIBRES DIRECTOS",
    "TIROS DENTRO ÁREA", 
    "TIROS DESDE ÚLTIMO TERCIO", 
    "XG POR TIROS", 
    "XG TIROS LEJANOS",
    "XG TIROS MEDIA DISTANCIA", 
    "XG TIROS CERCANOS", 
    "XG REMATES CABEZA", 
    "XG PENALTIS",
    "GOLES POR 90", 
    "GOLES TIRO LARGA DISTANCIA", 
    "GOLES TIRO MEDIA DISTANCIA",
    "GOLES TIRO CORTA DISTANCIA", 
    "GOLES CABEZA", 
    "GOLES UNO CONTRA UNO", 
    "GOLES PENALTI",
    "GOLES TIRO LIBRE", 
    "GOLES PUERTA VACÍA", 
    "GOLES DENTRO ÁREA", 
    "GOLES DESDE ÚLTIMO TERCIO",
    "GOLES DESDE CARRIL CENTRAL", 
    "GOLES EN POSESIÓN POR 90", 
    "GOLES CONTRAATAQUE POR 90",
    "GOLES JUGADA PARADA POR 90", 
    "XG POST TIRO", 
    "XG DENTRO ÁREA", 
    "XG DESDE ÚLTIMO TERCIO",
    "XG DESDE CARRIL CENTRAL", 
    "XG EN POSESIÓN", 
    "XG EN CONTRAATAQUE", 
    "XG EN JUGADAS BALÓN PARADO"
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
            html.H1("Análisis de Rendimiento Ofensivo", 
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
                                id="posicion-filter-ofensivo",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona posiciones",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de liga
                        dbc.Col([
                            html.Div(className="filter-title", children="Liga"),
                            dcc.Dropdown(
                                id="liga-filter-ofensivo",
                                options=[],  # Se llenará dinámicamente
                                multi=True,
                                placeholder="Selecciona ligas",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Filtro de equipos (dependiente de las ligas seleccionadas)
                        dbc.Col([
                            html.Div(className="filter-title", children="Equipo"),
                            dcc.Dropdown(
                                id="equipo-filter-ofensivo",
                                options=[],  # Se llenará dinámicamente según las ligas seleccionadas
                                multi=True,
                                placeholder="Selecciona equipos",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                    ], className="mb-4"),
                    
                    # Segunda fila de filtros - los 2 sliders
                    dbc.Row([
                        # Rango de edad
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Edad"),
                            dcc.RangeSlider(
                                id="edad-range-slider-ofensivo",
                                min=16, max=45, step=1,
                                marks={i: str(i) for i in range(16, 46, 5)},
                                value=[18, 35],
                                className="filter-slider"
                            ),
                        ], width=6),
                        
                        # Minutos jugados
                        dbc.Col([
                            html.Div(className="filter-title", children="Minutos Jugados (mínimo)"),
                            dcc.Slider(
                                id="minutos-slider-ofensivo",
                                min=0,
                                max=3000,
                                step=100,
                                value=500,
                                marks={i: str(i) for i in range(0, 3001, 500)},
                                className="filter-slider"
                            ),
                        ], width=6),
                    ], className="mb-4"),
                    
                    # Tercera fila - dropdown de métricas a ancho completo
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="filter-title", children="Métricas Ofensivas"),
                            dcc.Dropdown(
                                id="metricas-filter-ofensivo",
                                options=[
                                    {"label": "TIROS A PORTERÍA", "value": "TIROS A PORTERÍA"},
                                    {"label": "TIROS FUERA", "value": "TIROS FUERA"},
                                    {"label": "TIROS BLOQUEADOS", "value": "TIROS BLOQUEADOS"},
                                    {"label": "TIROS EXITOSOS", "value": "TIROS EXITOSOS"},
                                    {"label": "TIROS LARGA DISTANCIA", "value": "TIROS LARGA DISTANCIA"},
                                    {"label": "TIROS MEDIA DISTANCIA", "value": "TIROS MEDIA DISTANCIA"},
                                    {"label": "TIROS CORTA DISTANCIA", "value": "TIROS CORTA DISTANCIA"},
                                    {"label": "REMATES CABEZA", "value": "REMATES CABEZA"},
                                    {"label": "TIROS UNO CONTRA UNO", "value": "TIROS UNO CONTRA UNO"},
                                    {"label": "PENALTIS TIRADOS", "value": "PENALTIS TIRADOS"},
                                    {"label": "GOLES POR 90", "value": "GOLES POR 90"},
                                    {"label": "GOLES TIRO LARGA DISTANCIA", "value": "GOLES TIRO LARGA DISTANCIA"},
                                    {"label": "GOLES TIRO MEDIA DISTANCIA", "value": "GOLES TIRO MEDIA DISTANCIA"},
                                    {"label": "GOLES TIRO CORTA DISTANCIA", "value": "GOLES TIRO CORTA DISTANCIA"},
                                    {"label": "GOLES CABEZA", "value": "GOLES CABEZA"},
                                    {"label": "GOLES UNO CONTRA UNO", "value": "GOLES UNO CONTRA UNO"},
                                    {"label": "GOLES PENALTI", "value": "GOLES PENALTI"},
                                    {"label": "GOLES DENTRO ÁREA", "value": "GOLES DENTRO ÁREA"},
                                    {"label": "XG POR TIROS", "value": "XG POR TIROS"},
                                    {"label": "XG TIROS LEJANOS", "value": "XG TIROS LEJANOS"},
                                    {"label": "XG TIROS MEDIA DISTANCIA", "value": "XG TIROS MEDIA DISTANCIA"},
                                    {"label": "XG TIROS CERCANOS", "value": "XG TIROS CERCANOS"},
                                    {"label": "XG REMATES CABEZA", "value": "XG REMATES CABEZA"},
                                    {"label": "XG PENALTIS", "value": "XG PENALTIS"}
                                ],
                                value=["GOLES POR 90", "TIROS A PORTERÍA", "XG POR TIROS"],  # Valores por defecto
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
                                id="aplicar-filtros-btn-ofensivo",
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
                    export_button("ofensivo-pdf", "Informe de Rendimiento Ofensivo"),
                ], width={"size": 4, "offset": 4}),
            ], className="mb-4"),
            
            # Contenido de la página
            html.Div(
                id="offensive-content",
                children=[
                    # Primera fila: Tabla de los top goleadores
                    dbc.Row([
                        # Ranking de mejores atacantes a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Top 10 Goleadores", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="top-goleadores-table"),
                                    # Mensaje indicando que se puede seleccionar jugadores
                                    html.Div(
                                        "Haz clic en un jugador para ver análisis detallado",
                                        style={"text-align": "center", "font-style": "italic", "margin-top": "10px", "color": "#00BFFF"}
                                    )
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Segunda fila: Información del jugador seleccionado
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                id="jugador-seleccionado-info",
                                className="graph-container",
                                style={"display": "none"}  # Inicialmente oculto
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Tercera fila: Gráfico de mapa de tiros en campo
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                id="campo-tiros-container",
                                className="graph-container",
                                children=[
                                    html.H3("Distribución de Tiros y Goles", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="campo-tiros-chart")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Cuarta fila: Gráfico de radar comparativo y tendencia temporal
                    dbc.Row([
                        # Radar comparativo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Perfil Ofensivo Comparativo", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="radar-comparativo-chart")
                                ]
                            )
                        ], width=6),
                        
                        # Tendencia temporal
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Tendencia y Eficiencia", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="tendencia-eficiencia-chart")
                                ]
                            )
                        ], width=6),
                    ]),
                ]
            ),
            
            # Stores para los datos
            dcc.Store(id="filtered-data-store-ofensivo"),
            dcc.Store(id="jugador-seleccionado-store"),
        ]
    )
])

# Callback para llenar los dropdowns al cargar la página
@callback(
    [Output("posicion-filter-ofensivo", "options"),
     Output("liga-filter-ofensivo", "options")],
    Input("offensive-content", "children"),
    prevent_initial_call=False
)
def load_initial_dropdown_options(_):
    df = data_manager.get_data()
    
    # Opciones para posición
    posiciones = []
    if 'POSICIÓN' in df.columns:
        posiciones = [{"label": pos, "value": pos} for pos in sorted(df['POSICIÓN'].unique())]
    
    # Opciones para liga
    ligas = []
    if 'LIGA' in df.columns:
        ligas = [{"label": liga, "value": liga} for liga in sorted(df['LIGA'].unique())]
        # Añadir opción de todas las ligas
        ligas.insert(0, {"label": "Todas", "value": "todas"})
    
    return posiciones, ligas

# Callback para actualizar equipos basados en ligas seleccionadas
@callback(
    Output("equipo-filter-ofensivo", "options"),
    Input("liga-filter-ofensivo", "value"),
    prevent_initial_call=False
)
def update_equipos_options(ligas_seleccionadas):
    df = data_manager.get_data()
    
    equipos = []
    equipo_col = None
    
    # Buscar columna de equipo
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in df.columns:
            equipo_col = col
            break
    
    if equipo_col:
        # Si hay ligas seleccionadas y no incluye 'todas'
        if ligas_seleccionadas and 'todas' not in ligas_seleccionadas:
            filtered_df = df[df['LIGA'].isin(ligas_seleccionadas)]
            equipos = [{"label": eq, "value": eq} for eq in sorted(filtered_df[equipo_col].unique())]
        else:
            equipos = [{"label": eq, "value": eq} for eq in sorted(df[equipo_col].unique())]
        
        # Añadir opción de todos los equipos
        equipos.insert(0, {"label": "Todos", "value": "todos"})
    
    return equipos

# Callback para inicializar los datos al cargar y aplicar filtros cuando se hace clic
@callback(
    Output("filtered-data-store-ofensivo", "data"),
    [Input("aplicar-filtros-btn-ofensivo", "n_clicks")],
    [State("posicion-filter-ofensivo", "value"),
     State("liga-filter-ofensivo", "value"),
     State("equipo-filter-ofensivo", "value"),
     State("metricas-filter-ofensivo", "value"),
     State("edad-range-slider-ofensivo", "value"),
     State("minutos-slider-ofensivo", "value")],
    prevent_initial_call=False  # Permite que se ejecute al cargar
)
def filter_data(n_clicks, posiciones, ligas, equipos, metricas_seleccionadas, rango_edad, min_minutos):
    df = data_manager.get_data()
    
    if df.empty:
        return []
    
    # Hacer una copia de los datos
    filtered_df = df.copy()
    
    # Aplicar filtros básicos...
    if posiciones and len(posiciones) > 0:
        filtered_df = filtered_df[filtered_df['POSICIÓN'].isin(posiciones)]
    
    # Aplicar filtro de liga
    if ligas and len(ligas) > 0 and 'LIGA' in filtered_df.columns and 'todas' not in ligas:
        filtered_df = filtered_df[filtered_df['LIGA'].isin(ligas)]
    
    # Aplicar filtro de equipos
    if equipos and len(equipos) > 0 and 'todos' not in equipos:
        # Buscar columna de equipo disponible
        equipo_col = None
        for col in ['EQUIPO', 'CLUB', 'TEAM']:
            if col in filtered_df.columns:
                equipo_col = col
                break
        
        if equipo_col:
            filtered_df = filtered_df[filtered_df[equipo_col].isin(equipos)]
    
    # Aplicar filtro de edad
    if 'FECHA NACIMIENTO' in filtered_df.columns:
        min_edad, max_edad = rango_edad
        try:
            fecha_nacimiento = pd.to_datetime(filtered_df['FECHA NACIMIENTO'], errors='coerce')
            filtered_df['EDAD'] = pd.to_datetime('now').year - fecha_nacimiento.dt.year
            filtered_df = filtered_df[(filtered_df['EDAD'] >= min_edad) & (filtered_df['EDAD'] <= max_edad)]
        except Exception as e:
            print(f"Error al calcular edades: {e}")
    
    # Aplicar filtro de minutos jugados
    if 'MINUTOS JUGADOS' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['MINUTOS JUGADOS'] >= min_minutos]
    
    # CAMBIO IMPORTANTE: Incluir todas las columnas necesarias para las visualizaciones
    # Lista completa de columnas básicas y métricas específicas
    cols_to_include = ['NOMBRE', 'POSICIÓN', 'EDAD', 'MINUTOS JUGADOS']
    
    # Columnas de equipo y liga
    if 'LIGA' in filtered_df.columns:
        cols_to_include.append('LIGA')
    
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in filtered_df.columns:
            cols_to_include.append(col)
            break
    
    # Métricas específicas siempre necesarias
    essential_metrics = [
        'TIROS A PORTERÍA', 'TIROS FUERA', 'TIROS BLOQUEADOS', 'TIROS EXITOSOS',
        'TIROS LARGA DISTANCIA', 'TIROS MEDIA DISTANCIA', 'TIROS CORTA DISTANCIA',
        'REMATES CABEZA', 'TIROS UNO CONTRA UNO', 'PENALTIS TIRADOS', 
        'GOLES POR 90', 
        'GOLES TIRO LARGA DISTANCIA', 'GOLES TIRO MEDIA DISTANCIA', 'GOLES TIRO CORTA DISTANCIA',
        'GOLES CABEZA', 'GOLES UNO CONTRA UNO', 'GOLES PENALTI',
        'XG POR TIROS', 'XG TIROS LEJANOS', 'XG TIROS MEDIA DISTANCIA', 'XG TIROS CERCANOS'
    ]
    
    # Asegurar que todas estas métricas estén incluidas si existen
    for metric in essential_metrics:
        if metric in filtered_df.columns and metric not in cols_to_include:
            cols_to_include.append(metric)
    
    # También incluir las métricas seleccionadas por el usuario
    if metricas_seleccionadas and len(metricas_seleccionadas) > 0:
        for metric in metricas_seleccionadas:
            if metric in filtered_df.columns and metric not in cols_to_include:
                cols_to_include.append(metric)
    
    # Filtrar solo columnas que existen en el DataFrame
    cols_to_include = [col for col in cols_to_include if col in filtered_df.columns]
    
    # Seleccionar columnas y filtrar filas con valores faltantes
    # CAMBIO: No filtrar por columnas, mantener todo el DataFrame
    # filtered_df = filtered_df[cols_to_include]
    filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)  # Permitir algunos valores faltantes
    
    # Convertir a diccionario y devolver
    return {
        'data': filtered_df.to_dict('records'),
        'selected_metrics': metricas_seleccionadas if metricas_seleccionadas else essential_metrics
    }

# Callback para crear tabla interactiva de top goleadores
@callback(
    Output("top-goleadores-table", "children"),
    Input("filtered-data-store-ofensivo", "data")
)
def create_top_goleadores_table(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_ofensivas
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Crear un score ofensivo basado en las métricas relacionadas con goles
    metric_weights = {
        'GOLES POR 90': 1.0,
        'GOLES TIRO LARGA DISTANCIA': 0.5,
        'GOLES TIRO MEDIA DISTANCIA': 0.5,
        'GOLES TIRO CORTA DISTANCIA': 0.5,
        'GOLES CABEZA': 0.5,
        'GOLES UNO CONTRA UNO': 0.5,
        'GOLES PENALTI': 0.3,
        'GOLES TIRO LIBRE': 0.5,
        'GOLES PUERTA VACÍA': 0.2,
        'GOLES DENTRO ÁREA': 0.5,
        'XG POR TIROS': 0.7
    }
    
    # Inicializar score ofensivo
    offense_score = pd.Series(0, index=df.index)
    metrics_used = []
    
    # Buscar cualquier métrica de goles disponible
    goles_metric = None
    for metric in ['GOLES POR 90', 'GOLES TIRO LARGA DISTANCIA', 'GOLES TIRO MEDIA DISTANCIA', 
                  'GOLES TIRO CORTA DISTANCIA', 'GOLES CABEZA', 'GOLES UNO CONTRA UNO',
                  'GOLES PENALTI', 'GOLES TIRO LIBRE', 'GOLES PUERTA VACÍA', 'GOLES DENTRO ÁREA']:
        if metric in df.columns:
            goles_metric = metric
            break
    
    # Si no se encuentra ninguna métrica de goles, usar la primera métrica ofensiva disponible
    if not goles_metric:
        for metric in selected_metrics:
            if metric in df.columns:
                goles_metric = metric
                break
    
    if not goles_metric:
        return html.Div("No se encontraron métricas de goles en los datos disponibles.", 
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Calcular el score ofensivo
    for metric, weight in metric_weights.items():
        if metric in df.columns:
            metrics_used.append(metric)
            # Normalizar la métrica (0-1) y añadirla al score con su peso
            if df[metric].max() > df[metric].min():
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                offense_score += normalized * weight
    
    # Si no hay métricas con pesos, usar el goles_metric
    if not metrics_used:
        metrics_used.append(goles_metric)
        if df[goles_metric].max() > df[goles_metric].min():
            normalized = (df[goles_metric] - df[goles_metric].min()) / (df[goles_metric].max() - df[goles_metric].min())
            offense_score += normalized
    
    df['Offense_Score'] = offense_score
    
    # Ordenar por score ofensivo y tomar los 10 mejores
    top_10 = df.sort_values('Offense_Score', ascending=False).head(10)
    
    # Seleccionar columnas para mostrar, añadiendo FOTO al principio
    display_cols = ['FOTO', 'NOMBRE', 'POSICIÓN']
    
    # Columnas de datos (excluir FOTO que es virtual)
    data_cols = ['NOMBRE', 'POSICIÓN']
    
    # Añadir EQUIPO si está disponible
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in top_10.columns:
            display_cols.append(col)
            data_cols.append(col)
            break
    
    # Añadir LIGA si está disponible
    if 'LIGA' in top_10.columns:
        display_cols.append('LIGA')
        data_cols.append('LIGA')
    
    # Añadir EDAD si está disponible
    if 'EDAD' in top_10.columns:
        display_cols.append('EDAD')
        data_cols.append('EDAD')
    
    # Añadir la métrica principal de goles
    display_cols.append(goles_metric)
    data_cols.append(goles_metric)
    
    # Añadir otras métricas seleccionadas relacionadas con eficiencia
    efficiency_metrics = ['XG POR TIROS', 'TIROS A PORTERÍA', 'TIROS EXITOSOS']
    for metric in efficiency_metrics:
        if metric in top_10.columns and metric not in display_cols:
            display_cols.append(metric)
            data_cols.append(metric)
    
    # Crear los encabezados de la tabla
    header_cells = []
    
    # Añadir encabezado vacío para la columna de fotos
    header_cells.append(
        html.Th("", style={
            'backgroundColor': '#00BFFF',
            'color': 'white',
            'padding': '10px',
            'textAlign': 'center',
            'fontWeight': 'bold',
            'width': '60px'
        })
    )
    
    # Añadir resto de encabezados
    for col in data_cols:
        header_cells.append(
            html.Th(col, style={
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
    for i, (_, row) in enumerate(top_10.iterrows()):
        # Alternar colores de fondo
        bg_color = '#0E2B3D' if i % 2 == 0 else '#1A1A1A'
        row_cells = []
        
        # Añadir celda con foto del jugador
        player_name = row['NOMBRE']
        imagen_url = data_manager.get_player_image_url(player_name)
        
        row_cells.append(
            html.Td(
                html.Img(
                    src=imagen_url if imagen_url else "/assets/players/placeholder.png",
                    style={
                        "width": "40px",
                        "height": "40px",
                        "border-radius": "50%",
                        "object-fit": "cover",
                        "border": "2px solid #00BFFF",
                    }
                ),
                style={"padding": "5px", "text-align": "center", "width": "60px"}
            )
        )
        
        # Añadir el resto de celdas con datos
        for col in data_cols:
            value = row[col]
            # Formatear valores correctamente
            if isinstance(value, (int, float)) and not pd.isna(value):
                if col in efficiency_metrics or 'GOLES' in col or 'XG' in col:
                    formatted = f"{value:.2f}"
                elif col == 'EDAD':
                    formatted = f"{int(value)}" if not pd.isna(value) else "-"
                else:
                    formatted = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
            else:
                formatted = str(value) if not pd.isna(value) else "-"
            
            # Crear celda
            row_cells.append(html.Td(formatted, style={'padding': '10px'}))
        
        # Crear fila completa con ID único para identificación al hacer clic
        player_row = html.Tr(
            row_cells, 
            id={"type": "jugador-row", "index": player_name},
            style={
                'backgroundColor': bg_color,
                'cursor': 'pointer',  # Cambiar cursor a mano
                'transition': 'background-color 0.3s'
            },
            # Añadir efecto hover
            n_clicks=0
        )
        
        table_rows.append(player_row)
    
    # Crear tabla completa
    table = html.Table(
        [html.Thead(header_row), html.Tbody(table_rows)],
        style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'color': 'white'
        },
        id="tabla-jugadores"
    )
    
    return table

# Callback para manejar clics en filas de jugadores
@callback(
    [Output("jugador-seleccionado-store", "data"),
     Output("jugador-seleccionado-info", "children"),
     Output("jugador-seleccionado-info", "style")],
    [Input({"type": "jugador-row", "index": ALL}, "n_clicks")],
    [State({"type": "jugador-row", "index": ALL}, "id"),
     State("filtered-data-store-ofensivo", "data")]
)
def handle_player_click(n_clicks, row_ids, filtered_data):
    # Verificar si hay algún clic
    if not any(n_clicks) or not filtered_data:
        return None, None, {"display": "none"}
    
    # Encontrar el jugador que recibió el clic
    clicked_indices = [i for i, clicks in enumerate(n_clicks) if clicks > 0]
    
    if not clicked_indices:
        return None, None, {"display": "none"}
    
    # Tomar el último jugador clickeado
    latest_click = max(clicked_indices, key=lambda i: n_clicks[i])
    player_name = row_ids[latest_click]["index"]
    
    # Obtener datos del jugador
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    df = pd.DataFrame(raw_data)
    
    # Filtrar para obtener solo los datos del jugador seleccionado
    player_data = df[df["NOMBRE"] == player_name]
    
    if player_data.empty:
        return None, None, {"display": "none"}
    
    # Crear tarjeta de información del jugador
    player_info = player_data.iloc[0]
    
    # Obtener la URL de la imagen
    imagen_url = data_manager.get_player_image_url(player_name)
    
    # Encontrar columnas por categoría
    goles_cols = [col for col in player_data.columns if 'GOLES' in col]
    tiros_cols = [col for col in player_data.columns if ('TIRO' in col or 'TIROS' in col)]
    xg_cols = [col for col in player_data.columns if 'XG' in col]
    
    # Crear la tarjeta de información
    info_card = html.Div([
        html.Div([
            # Columna de la imagen y nombre
            html.Div([
                html.Img(
                    src=imagen_url if imagen_url else "/assets/players/placeholder.png",
                    style={
                        "width": "120px",
                        "height": "120px",
                        "border-radius": "50%",
                        "object-fit": "cover",
                        "border": "3px solid #00BFFF",
                        "margin-bottom": "10px"
                    }
                ),
                html.H3(player_name, style={"color": "#00BFFF", "margin-top": "10px"})
            ], style={"display": "flex", "flex-direction": "column", "align-items": "center", "width": "25%"}),
            
            # Columna de información básica
            html.Div([
                html.Div([
                    html.Strong("Posición: "),
                    html.Span(player_info.get("POSICIÓN", "No disponible"))
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Equipo: "),
                    html.Span(player_info.get("EQUIPO", player_info.get("CLUB", "No disponible")))
                ], style={"margin-bottom": "8px"}) if "EQUIPO" in player_info or "CLUB" in player_info else None,
                
                html.Div([
                    html.Strong("Liga: "),
                    html.Span(player_info.get("LIGA", "No disponible"))
                ], style={"margin-bottom": "8px"}) if "LIGA" in player_info else None,
                
                html.Div([
                    html.Strong("Edad: "),
                    html.Span(f"{int(player_info['EDAD'])}" if "EDAD" in player_info and not pd.isna(player_info["EDAD"]) else "No disponible")
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Minutos jugados: "),
                    html.Span(f"{int(player_info['MINUTOS JUGADOS'])}" if "MINUTOS JUGADOS" in player_info else "No disponible")
                ], style={"margin-bottom": "8px"}),
            ], style={"width": "25%", "padding": "0 15px"}),
            
            # Columna de métricas de goles
            html.Div([
                html.H4("Goles", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('GOLES', '').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in goles_cols[:5]]  # Limitar a 5 métricas
            ], style={"width": "25%", "padding": "0 15px"}),
            
            # Columna de métricas de tiros y xG
            html.Div([
                html.H4("Tiros y xG", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('TIROS', '').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in tiros_cols[:3]],  # Limitar a 3 métricas de tiros
                
                *[html.Div([
                    html.Strong(f"{col.replace('XG', 'xG').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in xg_cols[:2]]  # Limitar a 2 métricas de xG
            ], style={"width": "25%", "padding": "0 15px"}),
        ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),
        
        # Instrucción para ver gráficos detallados
        html.Div([
            html.P([
                "Se han generado visualizaciones detalladas para ",
                html.Strong(player_name),
                ". Desplázate hacia abajo para ver el análisis completo."
            ], style={"font-style": "italic", "text-align": "center", "margin-top": "10px"})
        ])
    ])
    
    return {"player_name": player_name, "player_data": player_data.to_dict('records')}, info_card, {"display": "block"}

# Callback para crear visualización de campo con tiros
@callback(
    Output("campo-tiros-chart", "children"),
    Input("jugador-seleccionado-store", "data"),
    State("filtered-data-store-ofensivo", "data")
)
def create_campo_tiros(jugador_data, filtered_data):
    if not jugador_data or not filtered_data:
        return html.Div("Selecciona un jugador de la tabla para ver su distribución de tiros y goles.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    player_name = jugador_data.get("player_name")
    if not player_name:
        return html.Div("No se ha seleccionado ningún jugador.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Obtener datos del jugador
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    df = pd.DataFrame(raw_data)
    player_df = df[df["NOMBRE"] == player_name]
    
    if player_df.empty:
        return html.Div(f"No se encontraron datos para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # EXTRAER DATOS DIRECTAMENTE DE LAS COLUMNAS EXACTAS - MANTENER LOS VALORES DECIMALES
    # Tiros por zona
    tiros_corta = float(player_df["TIROS CORTA DISTANCIA"].values[0]) if "TIROS CORTA DISTANCIA" in player_df.columns else 0
    tiros_media = float(player_df["TIROS MEDIA DISTANCIA"].values[0]) if "TIROS MEDIA DISTANCIA" in player_df.columns else 0
    tiros_larga = float(player_df["TIROS LARGA DISTANCIA"].values[0]) if "TIROS LARGA DISTANCIA" in player_df.columns else 0
    tiros_cabeza = float(player_df["REMATES CABEZA"].values[0]) if "REMATES CABEZA" in player_df.columns else 0
    penaltis_tirados = float(player_df["PENALTIS TIRADOS"].values[0]) if "PENALTIS TIRADOS" in player_df.columns else 0
    
    # Goles por zona
    goles_corta = float(player_df["GOLES TIRO CORTA DISTANCIA"].values[0]) if "GOLES TIRO CORTA DISTANCIA" in player_df.columns else 0
    goles_media = float(player_df["GOLES TIRO MEDIA DISTANCIA"].values[0]) if "GOLES TIRO MEDIA DISTANCIA" in player_df.columns else 0
    goles_larga = float(player_df["GOLES TIRO LARGA DISTANCIA"].values[0]) if "GOLES TIRO LARGA DISTANCIA" in player_df.columns else 0
    goles_cabeza = float(player_df["GOLES CABEZA"].values[0]) if "GOLES CABEZA" in player_df.columns else 0
    goles_penalti = float(player_df["GOLES PENALTI"].values[0]) if "GOLES PENALTI" in player_df.columns else 0
    
    # Imprimir valores extraídos para depuración
    print(f"Valores extraídos para {player_name}:")
    print(f"TIROS CORTA DISTANCIA: {tiros_corta}")
    print(f"TIROS MEDIA DISTANCIA: {tiros_media}")
    print(f"TIROS LARGA DISTANCIA: {tiros_larga}")
    print(f"REMATES CABEZA: {tiros_cabeza}")
    print(f"PENALTIS TIRADOS: {penaltis_tirados}")
    print(f"GOLES TIRO CORTA DISTANCIA: {goles_corta}")
    print(f"GOLES TIRO MEDIA DISTANCIA: {goles_media}")
    print(f"GOLES TIRO LARGA DISTANCIA: {goles_larga}")
    print(f"GOLES CABEZA: {goles_cabeza}")
    print(f"GOLES PENALTI: {goles_penalti}")
    
    # Calcular eficiencias - usar valores exactos, no redondear
    eficiencia_corta = (goles_corta / tiros_corta * 100) if tiros_corta > 0 else 0
    eficiencia_media = (goles_media / tiros_media * 100) if tiros_media > 0 else 0
    eficiencia_larga = (goles_larga / tiros_larga * 100) if tiros_larga > 0 else 0
    eficiencia_cabeza = (goles_cabeza / tiros_cabeza * 100) if tiros_cabeza > 0 else 0
    eficiencia_penalti = (goles_penalti / penaltis_tirados * 100) if penaltis_tirados > 0 else 0
    
    # Calcular eficiencia global
    suma_tiros = tiros_corta + tiros_media + tiros_larga + tiros_cabeza
    suma_goles = goles_corta + goles_media + goles_larga + goles_cabeza + goles_penalti
    
    eficiencia_general = (suma_goles / suma_tiros * 100) if suma_tiros > 0 else 0
    
    # Mostrar todos los valores en una tabla simple
    tabla_valores = html.Table(
        # Encabezado
        [html.Thead(html.Tr([
            html.Th("Zona", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "left"}),
            html.Th("Tiros (por 90')", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
            html.Th("Goles (por 90')", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
            html.Th("Eficiencia", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
        ], style={"backgroundColor": "rgba(0,191,255,0.2)"}))
        ] +
        # Cuerpo
        [html.Tbody([
            html.Tr([
                html.Td("Corta distancia", style={"border": "1px solid #00BFFF", "padding": "8px"}),
                html.Td(f"{tiros_corta:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{goles_corta:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{eficiencia_corta:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Media distancia", style={"border": "1px solid #00BFFF", "padding": "8px"}),
                html.Td(f"{tiros_media:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{goles_media:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{eficiencia_media:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
            ], style={"backgroundColor": "rgba(0,0,0,0.2)"}),
            html.Tr([
                html.Td("Larga distancia", style={"border": "1px solid #00BFFF", "padding": "8px"}),
                html.Td(f"{tiros_larga:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{goles_larga:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{eficiencia_larga:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Remates de cabeza", style={"border": "1px solid #00BFFF", "padding": "8px"}),
                html.Td(f"{tiros_cabeza:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{goles_cabeza:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{eficiencia_cabeza:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
            ], style={"backgroundColor": "rgba(0,0,0,0.2)"}),
            html.Tr([
                html.Td("Penaltis", style={"border": "1px solid #00BFFF", "padding": "8px"}),
                html.Td(f"{penaltis_tirados:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{goles_penalti:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"}),
                html.Td(f"{eficiencia_penalti:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("TOTAL", style={"border": "1px solid #00BFFF", "padding": "8px", "fontWeight": "bold"}),
                html.Td(f"{suma_tiros:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center", "fontWeight": "bold"}),
                html.Td(f"{suma_goles:.2f}", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center", "fontWeight": "bold"}),
                html.Td(f"{eficiencia_general:.1f}%", style={"border": "1px solid #00BFFF", "padding": "8px", "textAlign": "center", "fontWeight": "bold"})
            ], style={"backgroundColor": "rgba(0,191,255,0.1)"})
        ])],
        style={"width": "100%", "borderCollapse": "collapse", "marginTop": "20px"}
    )
    
    # Añadir análisis de texto
    analysis = html.Div([
        html.H4(f"Análisis de Tiros y Goles de {player_name}", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"El jugador ha registrado un total de {suma_tiros:.2f} tiros por 90 minutos con una eficiencia global del ",
            html.Strong(f"{eficiencia_general:.1f}%", style={"color": "#FFFF00"}),
            "."
        ]),
        html.P([
            "Sus zonas más productivas son: ",
            html.Strong(
                "corta distancia" if eficiencia_corta >= max(eficiencia_media, eficiencia_larga, eficiencia_cabeza, eficiencia_penalti) else
                "media distancia" if eficiencia_media >= max(eficiencia_corta, eficiencia_larga, eficiencia_cabeza, eficiencia_penalti) else
                "larga distancia" if eficiencia_larga >= max(eficiencia_corta, eficiencia_media, eficiencia_cabeza, eficiencia_penalti) else
                "remates de cabeza" if eficiencia_cabeza >= max(eficiencia_corta, eficiencia_media, eficiencia_larga, eficiencia_penalti) else
                "penaltis",
                style={"color": "#FFFF00"}
            ),
            f" ({max(eficiencia_corta, eficiencia_media, eficiencia_larga, eficiencia_cabeza, eficiencia_penalti):.1f}% de eficiencia)."
        ]),
        html.P([
            "Nota: Todos los valores se expresan por 90 minutos jugados, no son totales absolutos."
        ], style={"fontStyle": "italic", "fontSize": "0.9em", "color": "#aaaaaa"}),
        tabla_valores
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    # Si queremos mantener el gráfico de campo, podemos ajustarlo también...
    # Pero por ahora, solo mostramos la tabla y el análisis
    
    return analysis
    
# Callback para crear radar comparativo
@callback(
    Output("radar-comparativo-chart", "children"),
    Input("jugador-seleccionado-store", "data"),
    State("filtered-data-store-ofensivo", "data")
)
def create_radar_comparativo(jugador_data, filtered_data):
    if not jugador_data or not filtered_data:
        return html.Div("Selecciona un jugador de la tabla para ver su perfil comparativo.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    player_name = jugador_data.get("player_name")
    if not player_name:
        return html.Div("No se ha seleccionado ningún jugador.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Obtener datos completos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    df = pd.DataFrame(raw_data)
    
    # Filtrar para el jugador seleccionado
    player_df = df[df["NOMBRE"] == player_name]
    
    if player_df.empty:
        return html.Div(f"No se encontraron datos para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Obtener la posición del jugador para comparar con similares
    player_position = player_df["POSICIÓN"].values[0] if "POSICIÓN" in player_df.columns else None
    
    if not player_position:
        return html.Div(f"No se encontró la posición para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Filtrar jugadores de la misma posición
    similar_players = df[df["POSICIÓN"] == player_position].copy()
    
    # Eliminar al jugador seleccionado de la lista de similares
    similar_players = similar_players[similar_players["NOMBRE"] != player_name]
    
    if len(similar_players) < 2:
        return html.Div(f"No hay suficientes jugadores en la posición {player_position} para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Identificar métricas ofensivas relevantes disponibles
    offensive_metrics = []
    for category, keyword in [
        ("Goles", "GOLES"),
        ("Tiros", "TIRO"),
        ("xG", "XG"),
        ("Regates", "REGAT"),
        ("Asistencias", "ASIST")
    ]:
        # Buscar columnas que contengan la palabra clave
        category_cols = [col for col in df.columns if keyword in col]
        if category_cols:
            # Tomar la primera métrica de cada categoría
            offensive_metrics.append(category_cols[0])
    
    # Asegurarse de que existan valores para todas las métricas
    for metric in offensive_metrics:
        if metric not in player_df.columns or pd.isna(player_df[metric].values[0]):
            # Buscar métricas alternativas o usar un valor por defecto
            if "GOLES" in metric and "GOLES POR 90" in player_df.columns:
                player_df[metric] = player_df["GOLES POR 90"]
            elif "TIROS" in metric and "TIROS A PORTERÍA" in player_df.columns:
                player_df[metric] = player_df["TIROS A PORTERÍA"]
            elif "XG" in metric and "XG POR TIROS" in player_df.columns:
                player_df[metric] = player_df["XG POR TIROS"]
            else:
                # Usar un valor por defecto bajo
                player_df[metric] = 0.1  # Valor pequeño para no distorsionar el gráfico
    
    # Si no hay suficientes métricas, mostrar mensaje
    if len(offensive_metrics) < 3:
        # Usar métricas comunes si no hay suficientes
        typical_metrics = ["GOLES POR 90", "TIROS A PORTERÍA", "XG POR TIROS"]
        offensive_metrics = [m for m in typical_metrics if m in df.columns]
        
        if len(offensive_metrics) < 3:
            return html.Div("No se encontraron suficientes métricas ofensivas para crear el radar comparativo.", 
                        style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Limitar a 5 métricas para mejor visualización
    offensive_metrics = offensive_metrics[:5]
    
    # Calcular similitud para encontrar jugadores comparables
    # Método simple: distancia euclidiana normalizada
    
    # Normalizar métricas
    for metric in offensive_metrics:
        if metric in similar_players.columns:
            metric_min = df[metric].min()
            metric_max = df[metric].max()
            if metric_max > metric_min:
                similar_players[f"{metric}_norm"] = (similar_players[metric] - metric_min) / (metric_max - metric_min)
            else:
                similar_players[f"{metric}_norm"] = 0.5
    
    # Normalizar métricas para el jugador seleccionado
    player_normalized = {}
    for metric in offensive_metrics:
        if metric in player_df.columns:
            metric_min = df[metric].min()
            metric_max = df[metric].max()
            if metric_max > metric_min:
                player_normalized[f"{metric}_norm"] = (player_df[metric].values[0] - metric_min) / (metric_max - metric_min)
            else:
                player_normalized[f"{metric}_norm"] = 0.5
    
    # Calcular distancia euclidiana
    distances = []
    for idx, row in similar_players.iterrows():
        distance = 0
        for metric in offensive_metrics:
            norm_metric = f"{metric}_norm"
            if norm_metric in player_normalized and norm_metric in row:
                distance += (player_normalized[norm_metric] - row[norm_metric]) ** 2
        distances.append((row["NOMBRE"], (distance ** 0.5)))
    
    # Ordenar por similitud (menor distancia)
    distances.sort(key=lambda x: x[1])
    
    # Seleccionar los 2 jugadores más similares
    similar_names = [d[0] for d in distances[:2]]
    
    # Preparar datos para el radar
    radar_data = []
    
    # Añadir jugador seleccionado
    player_values = []
    for metric in offensive_metrics:
        if metric in player_df.columns:
            player_values.append(player_df[metric].values[0])
        else:
            player_values.append(0)
    
    radar_data.append({
        "player": player_name,
        "values": player_values,
        "metrics": offensive_metrics
    })
    
    # Añadir jugadores similares
    for similar_name in similar_names:
        similar_df = df[df["NOMBRE"] == similar_name]
        if not similar_df.empty:
            similar_values = []
            for metric in offensive_metrics:
                if metric in similar_df.columns:
                    similar_values.append(similar_df[metric].values[0])
                else:
                    similar_values.append(0)
            
            radar_data.append({
                "player": similar_name,
                "values": similar_values,
                "metrics": offensive_metrics
            })
    
    # Crear gráfico de radar
    fig = go.Figure()
    
    # Formatear nombres de métricas para mejor visualización
    formatted_metrics = [m.replace("_", " ").title() for m in offensive_metrics]
    
    # Colores para cada jugador
    colors = ['#00BFFF', '#FFFF00', '#4CAF50']
    
    # Añadir un trace para cada jugador
    for i, player_data in enumerate(radar_data):
        # Cerrar el polígono repitiendo el primer valor
        values = player_data["values"] + [player_data["values"][0]]
        metrics = formatted_metrics + [formatted_metrics[0]]
        
        # Definir el color de relleno para cada jugador
        if i == 0:  # Primer jugador - azul
            fillcolor = 'rgba(0, 191, 255, 0.3)'
        elif i == 1:  # Segundo jugador - amarillo
            fillcolor = 'rgba(255, 255, 0, 0.3)'
        else:  # Tercer jugador - verde
            fillcolor = 'rgba(76, 175, 80, 0.3)'
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=player_data["player"],
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=fillcolor
        ))
    
    # Configurar layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([max(p["values"]) for p in radar_data]) * 1.1],
                showticklabels=True,
                gridcolor="rgba(255,255,255,0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.2)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,  # Cambiado de -0.1 a -0.25 para bajar más la leyenda
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        height=450,
        margin=dict(l=80, r=80, t=50, b=120)  # Aumentado el margen inferior de b=100 a b=120
    )
    
    # Añadir análisis comparativo
    analysis = html.Div([
        html.H4(f"Comparación de {player_name} con jugadores similares", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"Se ha comparado el perfil ofensivo de {player_name} con otros jugadores de la posición {player_position}. ",
            f"Los jugadores más similares son ",
            html.Strong(f"{similar_names[0]}", style={"color": colors[1]}),
            " y ",
            html.Strong(f"{similar_names[1]}", style={"color": colors[2]}) if len(similar_names) > 1 else "",
            "."
        ]),
        html.P([
            "El gráfico radar muestra cómo se comparan los jugadores en varias métricas ofensivas clave. ",
            "Cuanto mayor sea el área cubierta, mejor es el rendimiento global del jugador."
        ]),
        html.P([
            f"{player_name} destaca especialmente en ",
            html.Strong(
                formatted_metrics[player_values.index(max(player_values))],
                style={"color": "#FFFF00"}
            ),
            " en comparación con jugadores similares."
        ]),
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    return html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False}), analysis])

# Callback para crear gráfico de tendencia y eficiencia
@callback(
    Output("tendencia-eficiencia-chart", "children"),
    Input("jugador-seleccionado-store", "data"),
    State("filtered-data-store-ofensivo", "data")
)
def create_tendencia_eficiencia(jugador_data, filtered_data):
    if not jugador_data or not filtered_data:
        return html.Div("Selecciona un jugador de la tabla para ver su tendencia y eficiencia.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    player_name = jugador_data.get("player_name")
    if not player_name:
        return html.Div("No se ha seleccionado ningún jugador.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Obtener datos completos
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
    else:
        raw_data = filtered_data
    
    df = pd.DataFrame(raw_data)
    
    # Filtrar para el jugador seleccionado
    player_df = df[df["NOMBRE"] == player_name]
    
    if player_df.empty:
        return html.Div(f"No se encontraron datos para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Buscar métricas clave disponibles
    goles_metric = next((m for m in ["GOLES POR 90", "GOLES DENTRO ÁREA"] if m in player_df.columns), None)
    tiros_metric = next((m for m in ["TIROS A PORTERÍA", "TIROS EXITOSOS"] if m in player_df.columns), None)
    xg_metric = next((m for m in ["XG POR TIROS", "XG TIROS CERCANOS"] if m in player_df.columns), None)
    
    # Si no tenemos métricas suficientes, mostrar alternativa
    if not goles_metric or not tiros_metric:
        return create_alternative_chart(player_df, player_name)
    
    # Extraer valores
    goles_valor = player_df[goles_metric].values[0]
    tiros_valor = player_df[tiros_metric].values[0]
    xg_valor = player_df[xg_metric].values[0] if xg_metric else None
    
    # Calcular métricas de eficiencia
    conversion_rate = (goles_valor / tiros_valor * 100) if tiros_valor > 0 else 0
    xg_efficiency = (goles_valor / xg_valor * 100) if xg_valor and xg_valor > 0 else None
    
    # Obtener métricas de tiros por zona si están disponibles
    tiros_data = []
    if "TIROS CORTA DISTANCIA" in player_df.columns:
        tiros_data.append(("Corta", player_df["TIROS CORTA DISTANCIA"].values[0]))
    if "TIROS MEDIA DISTANCIA" in player_df.columns:
        tiros_data.append(("Media", player_df["TIROS MEDIA DISTANCIA"].values[0]))
    if "TIROS LARGA DISTANCIA" in player_df.columns:
        tiros_data.append(("Larga", player_df["TIROS LARGA DISTANCIA"].values[0]))
    if "REMATES CABEZA" in player_df.columns:
        tiros_data.append(("Cabeza", player_df["REMATES CABEZA"].values[0]))
    
    # Crear gráfico de barras para porcentaje de tiros por zona
    tiros_fig = None
    if tiros_data:
        # Calcular porcentajes
        total_tiros = sum(t[1] for t in tiros_data)
        if total_tiros > 0:
            for i in range(len(tiros_data)):
                tiros_data[i] = (tiros_data[i][0], tiros_data[i][1], tiros_data[i][1]/total_tiros*100)
        
        # Crear figura
        tiros_fig = go.Figure()
        
        # Añadir barras
        tiros_fig.add_trace(go.Bar(
            x=[t[0] for t in tiros_data],
            y=[t[2] for t in tiros_data],
            marker=dict(
                color='#00BFFF',
                line=dict(width=2, color='#FFFF00')
            ),
            text=[f"{t[2]:.1f}%" for t in tiros_data],
            textposition='auto'
        ))
        
        # Configurar layout
        tiros_fig.update_layout(
            title="Distribución de Tiros por Zona",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(text="Zona de Tiro", font=dict(color="white", size=12)),
                gridcolor="rgba(255,255,255,0.1)"
            ),
            yaxis=dict(
                title=dict(text="Porcentaje", font=dict(color="white", size=12)),
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            height=300,
            margin=dict(l=50, r=50, t=50, b=30)
        )
    
    # Crear gauges para eficiencia
    # Gauge de conversión de tiros
    conversion_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conversion_rate,
        title={'text': "Conversión de Tiros (%)", 'font': {'size': 14, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': "#00BFFF"},
            'bordercolor': "#FFFF00",
            'steps': [
                {'range': [0, 33], 'color': "rgba(255,255,255,0.1)"},
                {'range': [33, 66], 'color': "rgba(255,255,255,0.2)"},
                {'range': [66, 100], 'color': "rgba(255,255,255,0.3)"}
            ],
            'threshold': {
                'line': {'color': "#FFFF00", 'width': 4},
                'thickness': 0.75,
                'value': conversion_rate
            }
        },
        number={'suffix': "%", 'font': {'size': 20, 'color': 'white'}}
    ))
    
    conversion_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    # Gauge de eficiencia vs xG si está disponible
    xg_gauge = None
    if xg_efficiency is not None:
        xg_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=xg_efficiency,
            title={'text': "Eficiencia vs xG (%)", 'font': {'size': 14, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 200], 'tickcolor': "white"},
                'bar': {'color': "#FFFF00"},
                'bordercolor': "#00BFFF",
                'steps': [
                    {'range': [0, 100], 'color': "rgba(255,255,255,0.1)"},
                    {'range': [100, 150], 'color': "rgba(255,255,255,0.2)"},
                    {'range': [150, 200], 'color': "rgba(255,255,255,0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#00BFFF", 'width': 4},
                    'thickness': 0.75,
                    'value': 100  # Línea de referencia en 100% (rendimiento esperado)
                }
            },
            number={'suffix': "%", 'font': {'size': 20, 'color': 'white'}}
        ))
        
        xg_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=30, r=30, t=50, b=30)
        )
    
    # Crear análisis de texto
    metric_analysis = html.Div([
        html.H4(f"Análisis de Eficiencia de {player_name}", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"La tasa de conversión de tiros del jugador es del ",
            html.Strong(f"{conversion_rate:.1f}%", style={"color": "#FFFF00"}),
            f", lo que significa que marca {conversion_rate/100:.2f} goles por cada tiro realizado."
        ]),
        html.P([
            "Su eficiencia respecto al xG esperado es del ",
            html.Strong(f"{xg_efficiency:.1f}%" if xg_efficiency is not None else "N/A", 
                      style={"color": "#FFFF00"}),
            ". ",
            html.Span(
                "Está rindiendo por encima de lo esperado según las ocasiones generadas." 
                if xg_efficiency is not None and xg_efficiency > 100 else
                "Está rindiendo por debajo de lo esperado según las ocasiones generadas."
                if xg_efficiency is not None else ""
            )
        ]) if xg_efficiency is not None else None,
        html.P([
            "La mayoría de sus tiros provienen de ",
            html.Strong(
                tiros_data[max(range(len(tiros_data)), key=lambda i: tiros_data[i][1])][0] if tiros_data else "N/A",
                style={"color": "#FFFF00"}
            ),
            " distancia, lo que refleja su estilo de juego y posicionamiento."
        ]) if tiros_data else None,
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    # Organizar los elementos en la página
    if tiros_fig and xg_gauge:
        # Si tenemos todos los gráficos
        return html.Div([
            dbc.Row([
                # Gauges en la parte superior
                dbc.Col([dcc.Graph(figure=conversion_gauge, config={'displayModeBar': False})], width=6),
                dbc.Col([dcc.Graph(figure=xg_gauge, config={'displayModeBar': False})], width=6),
            ]),
            dbc.Row([
                # Gráfico de distribución de tiros
                dbc.Col([dcc.Graph(figure=tiros_fig, config={'displayModeBar': False})], width=12),
            ]),
            # Análisis de texto al final
            metric_analysis
        ])
    elif tiros_fig:
        # Si solo tenemos gráfico de tiros y gauge de conversión
        return html.Div([
            dbc.Row([
                # Gauge de conversión
                dbc.Col([dcc.Graph(figure=conversion_gauge, config={'displayModeBar': False})], width=12),
            ]),
            dbc.Row([
                # Gráfico de distribución de tiros
                dbc.Col([dcc.Graph(figure=tiros_fig, config={'displayModeBar': False})], width=12),
            ]),
            # Análisis de texto
            metric_analysis
        ])
    else:
        # Si solo tenemos los gauges
        gauges_row = dbc.Row([
            dbc.Col([dcc.Graph(figure=conversion_gauge, config={'displayModeBar': False})], 
                   width=12 if xg_gauge is None else 6),
        ])
        
        if xg_gauge:
            gauges_row.children.append(
                dbc.Col([dcc.Graph(figure=xg_gauge, config={'displayModeBar': False})], width=6)
            )
        
        return html.Div([
            gauges_row,
            metric_analysis
        ])

# Función auxiliar para gráficos alternativos cuando faltan métricas de tendencia
def create_alternative_chart(player_df, player_name):
    # Buscar métricas numéricas disponibles
    numeric_cols = player_df.select_dtypes(include=['number']).columns
    
    # Filtrar solo columnas ofensivas
    offensive_keywords = ['GOL', 'TIRO', 'XG', 'REGAT', 'ASIST']
    offensive_metrics = []
    
    for col in numeric_cols:
        if any(keyword in col for keyword in offensive_keywords):
            offensive_metrics.append(col)
    
    # Si no hay métricas ofensivas, mostrar mensaje
    if not offensive_metrics:
        return html.Div("No se encontraron métricas ofensivas para analizar tendencias.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Limitar a 6 métricas para mejor visualización
    if len(offensive_metrics) > 6:
        offensive_metrics = offensive_metrics[:6]
    
    # Crear gráfico de barras con las métricas disponibles
    fig = go.Figure()
    
    # Extraer valores
    values = [player_df[metric].values[0] for metric in offensive_metrics]
    
    # Formatear nombres para mejor visualización
    formatted_metrics = [metric.replace("_", " ").title() for metric in offensive_metrics]
    
    # Crear barras
    fig.add_trace(go.Bar(
        x=formatted_metrics,
        y=values,
        marker=dict(
            color='#00BFFF',
            line=dict(width=2, color='#FFFF00')
        ),
        text=[f"{v:.2f}" for v in values],
        textposition='auto'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=f"Métricas Ofensivas de {player_name}",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="white"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Métrica", font=dict(color="white", size=14)),
            gridcolor="rgba(255,255,255,0.1)",
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text="Valor", font=dict(color="white", size=14)),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        height=400,
        margin=dict(l=50, r=50, t=70, b=120)
    )
    
    # Análisis simple
    analysis = html.Div([
        html.H4(f"Análisis de Métricas de {player_name}", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"El gráfico muestra las principales métricas ofensivas disponibles para {player_name}. ",
            f"Destaca especialmente en ",
            html.Strong(
                formatted_metrics[values.index(max(values))],
                style={"color": "#FFFF00"}
            ),
            f" con un valor de {max(values):.2f}."
        ]),
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        analysis
    ])
    
    # Callback para abrir el modal
@callback(
    Output("ofensivo-pdf-modal", "is_open"),
    [Input("ofensivo-pdf-button", "n_clicks"),
     Input("ofensivo-pdf-modal-close", "n_clicks"),
     Input("ofensivo-pdf-modal-generate", "n_clicks")],
    [State("ofensivo-pdf-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

# Callback para generar y descargar el PDF
@callback(
    Output("ofensivo-pdf-download", "data"),
    Input("ofensivo-pdf-modal-generate", "n_clicks"),
    [State("ofensivo-pdf-title", "value"),
     State("ofensivo-pdf-description", "value"),
     State("filtered-data-store-ofensivo", "data")],
    prevent_initial_call=True
)
def generate_pdf_ofensivo(n_clicks, title, description, filtered_data):
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
    elements.append(Paragraph(title or "Informe de Rendimiento Ofensivo", title_style))
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
        metrics_text = f"Filtros aplicados: Métricas: {', '.join(metrics[:3])}"
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
    
    # Crear tabla de jugadores
    if not df.empty:
        # Ordenar por alguna métrica relevante
        sort_metric = None
        for metric in ["GOLES POR 90", "TIROS A PORTERÍA", "XG POR TIROS"]:
            if metric in df.columns:
                sort_metric = metric
                break
        
        if sort_metric:
            df = df.sort_values(sort_metric, ascending=False)
        
        # Seleccionar columnas de interés
        headers = ["Nombre", "Posición", "Liga"]
        metrics_to_show = []
        
        # Añadir métricas principales - ABREVIAR los nombres
        for metric in metrics[:5]:
            if metric in df.columns:
                # Abreviar los nombres de las métricas para que quepan mejor
                short_name = metric
                short_name = short_name.replace("TIROS A PORTERÍA", "TIROS PORT.")
                short_name = short_name.replace("GOLES POR 90", "GOLES/90")
                short_name = short_name.replace("XG POR TIROS", "XG TIROS")
                short_name = short_name.replace("TIROS LARGA DISTANCIA", "TIROS L.D.")
                short_name = short_name.replace("TIROS MEDIA DISTANCIA", "TIROS M.D.")
                short_name = short_name.replace("TIROS CORTA DISTANCIA", "TIROS C.D.")
                
                # Limitar longitud a 12 caracteres para evitar desbordes
                if len(short_name) > 12:
                    short_name = short_name[:12]
                
                headers.append(short_name)
                metrics_to_show.append(metric)
        
        # Preparar datos para la tabla
        table_data = [headers]
        
        # Añadir filas de datos - Acortar nombres si son muy largos
        for _, row in df.head(10).iterrows():
            nombre = row["NOMBRE"]
            if len(nombre) > 18:  # Limitar largo de nombres
                nombre_parts = nombre.split()
                if len(nombre_parts) > 1:
                    nombre = f"{nombre_parts[0]} {nombre_parts[-1]}"
                else:
                    nombre = nombre[:18]
                    
            row_data = [
                nombre,
                row["POSICIÓN"],
                row.get("LIGA", "")
            ]
            
            # Añadir métricas
            for metric in metrics_to_show:
                value = row.get(metric, "")
                if isinstance(value, (int, float)):
                    row_data.append(f"{value:.2f}")
                else:
                    row_data.append(str(value))
            
            table_data.append(row_data)
        
        # AJUSTAR TAMAÑOS DE COLUMNAS - Más espacio para métricas
        # Calcular anchos proporcionales al espacio disponible (750 puntos disponibles aprox.)
        page_width = 750
        num_cols = len(headers)
        
        # Distribuir el ancho disponible
        col_widths = []
        col_widths.append(110)  # Nombre
        col_widths.append(85)   # Posición
        col_widths.append(65)   # Liga
        
        # Calcular espacio restante para métricas
        metrics_space = page_width - sum(col_widths) - 10
        metrics_width = metrics_space / len(metrics_to_show)
        
        # Añadir anchos para métricas
        for _ in metrics_to_show:
            col_widths.append(metrics_width)
        
        # Crear tabla con los anchos calculados
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
            ('ALIGN', (0, 1), (2, -1), 'LEFT'),      # Alineación izquierda para nombre, posición, liga
            ('ALIGN', (3, 1), (-1, -1), 'CENTER'),   # Alineación central para métricas
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),       # Texto más pequeño para que quepa
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
    elements.append(Paragraph("SOCCER DATA DBC - Análisis de Rendimiento Deportivo", 
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
        "filename": f"informe_ofensivo_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }