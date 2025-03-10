from dash import dcc, html, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from components.data_manager import DataManager
from components.sidebar import create_sidebar
from pdf_export import export_button, create_pdf_report
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir métricas de creación de juego principales
metricas_creacion = [
    "PASES EXITOSOS POR 90", 
    "PASES FALLIDOS", 
    "PASES RASOS EXITOSOS", 
    "PASES DIAGONALES EXITOSOS",
    "PASES ELEVADOS EXITOSOS", 
    "PASES AÉREOS CORTOS EXITOSOS", 
    "CENTROS ALTOS EXITOSOS", 
    "CENTROS RASOS EXITOSOS", 
    "CÓRNERS EXITOSOS", 
    "TIROS LIBRES EXITOSOS", 
    "PASES DESDE PRIMER TERCIO", 
    "PASES DESDE TERCIO MEDIO", 
    "PASES DESDE ÚLTIMO TERCIO",
    "PASES EXITOSOS HACIA ÚLTIMO TERCIO", 
    "PASES EXITOSOS HACIA ÁREA RIVAL", 
    "PASES EXITOSOS A MEDIAPUNTA", 
    "PASES EXITOSOS A MEDIOCENTRO", 
    "PASES EXITOSOS A BANDA DERECHA",
    "PASES EXITOSOS A BANDA IZQUIERDA", 
    "PASES EXITOSOS A PIVOTE", 
    "PASES DESDE CARRIL CENTRAL",
    "PASES HACIA CARRIL CENTRAL", 
    "PASES DESDE BANDA DERECHA", 
    "PASES HACIA BANDA DERECHA",
    "PASES DESDE BANDA IZQUIERDA", 
    "PASES HACIA BANDA IZQUIERDA", 
    "PASES EN FASE POSESIÓN",
    "PASES EN TRANSICIÓN OFENSIVA", 
    "PASES EN JUGADAS BALÓN PARADO", 
    "PASES EXITOSOS HACIA ÁREA",
    "PASES EXITOSOS HACIA PRIMER TERCIO", 
    "PASES EXITOSOS HACIA TERCIO MEDIO", 
    "PASES DESDE LATERAL DERECHO POR 90", 
    "PASES DESDE DEFENSA CENTRAL POR 90", 
    "PASES DESDE LATERAL IZQUIERDO POR 90", 
    "ASISTENCIAS POR 90", 
    "ASISTENCIAS PASE RASO",
    "ASISTENCIAS PASE DIAGONAL", 
    "ASISTENCIAS PASE ELEVADO", 
    "ASISTENCIAS PASE AÉREO CORTO",
    "ASISTENCIAS CENTRO RASO", 
    "ASISTENCIAS CENTRO ALTO", 
    "ASISTENCIAS DESDE MEDIAPUNTA",
    "ASISTENCIAS DESDE MEDIOCENTRO", 
    "ASISTENCIAS DESDE BANDA DERECHA", 
    "ASISTENCIAS DESDE BANDA IZQUIERDA", 
    "ASISTENCIAS DESDE ÚLTIMO TERCIO", 
    "ASISTENCIAS DESDE CARRIL CENTRAL", 
    "ASISTENCIAS DESDE BANDA DERECHA/90", 
    "ASISTENCIAS DESDE BANDA IZQUIERDA/90", 
    "XG CREADA CON PASES", 
    "ASISTENCIAS POR CÓRNER",
    "ASISTENCIAS CONTRAATAQUE", 
    "ASISTENCIAS EN POSESIÓN", 
    "ASISTENCIAS JUGADAS BALÓN PARADO"
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
            html.H1("Análisis de Creación de Juego", 
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
                                id="posicion-filter-creacion",
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
                                id="liga-filter-creacion",
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
                                id="equipo-filter-creacion",
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
                                id="edad-range-slider-creacion",
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
                                id="minutos-slider-creacion",
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
                            html.Div(className="filter-title", children="Métricas de Creación"),
                            dcc.Dropdown(
                                id="metricas-filter-creacion",
                                options=[
                                    {"label": "PASES EXITOSOS POR 90", "value": "PASES EXITOSOS POR 90"},
                                    {"label": "PASES FALLIDOS", "value": "PASES FALLIDOS"},
                                    {"label": "PASES RASOS EXITOSOS", "value": "PASES RASOS EXITOSOS"},
                                    {"label": "PASES DIAGONALES EXITOSOS", "value": "PASES DIAGONALES EXITOSOS"},
                                    {"label": "PASES ELEVADOS EXITOSOS", "value": "PASES ELEVADOS EXITOSOS"},
                                    {"label": "PASES AÉREOS CORTOS EXITOSOS", "value": "PASES AÉREOS CORTOS EXITOSOS"},
                                    {"label": "CENTROS ALTOS EXITOSOS", "value": "CENTROS ALTOS EXITOSOS"},
                                    {"label": "CENTROS RASOS EXITOSOS", "value": "CENTROS RASOS EXITOSOS"},
                                    {"label": "PASES EXITOSOS HACIA ÚLTIMO TERCIO", "value": "PASES EXITOSOS HACIA ÚLTIMO TERCIO"},
                                    {"label": "PASES EXITOSOS HACIA ÁREA RIVAL", "value": "PASES EXITOSOS HACIA ÁREA RIVAL"},
                                    {"label": "PASES EN FASE POSESIÓN", "value": "PASES EN FASE POSESIÓN"},
                                    {"label": "PASES EN TRANSICIÓN OFENSIVA", "value": "PASES EN TRANSICIÓN OFENSIVA"},
                                    {"label": "ASISTENCIAS POR 90", "value": "ASISTENCIAS POR 90"},
                                    {"label": "ASISTENCIAS PASE RASO", "value": "ASISTENCIAS PASE RASO"},
                                    {"label": "ASISTENCIAS CENTRO ALTO", "value": "ASISTENCIAS CENTRO ALTO"},
                                    {"label": "ASISTENCIAS DESDE ÚLTIMO TERCIO", "value": "ASISTENCIAS DESDE ÚLTIMO TERCIO"},
                                    {"label": "XG CREADA CON PASES", "value": "XG CREADA CON PASES"},
                                    {"label": "ASISTENCIAS CONTRAATAQUE", "value": "ASISTENCIAS CONTRAATAQUE"},
                                    {"label": "ASISTENCIAS EN POSESIÓN", "value": "ASISTENCIAS EN POSESIÓN"}
                                ],
                                value=["ASISTENCIAS POR 90", "PASES EXITOSOS POR 90", "XG CREADA CON PASES"],  # Valores por defecto
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
                                id="aplicar-filtros-btn-creacion",
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
                    export_button("creacion-pdf", "Informe de Creación de Juego"),
                ], width={"size": 4, "offset": 4}),
            ], className="mb-4"),
            
            # Contenido de la página
            html.Div(
                id="creation-content",
                children=[
                    # Primera fila: Tabla de los mejores creadores de juego
                    dbc.Row([
                        # Ranking de mejores creadores a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Top 10 Creadores de Juego", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="top-creadores-table"),
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
                                id="jugador-seleccionado-info-creacion",
                                className="graph-container",
                                style={"display": "none"}  # Inicialmente oculto
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Tercera fila: Mapas de calor de pases
                    dbc.Row([
                        # Mapa de calor de pases a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Visualización de Pases", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="pases-visualization")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Cuarta fila: Eficiencia de pases y asistencias por zona
                    dbc.Row([
                        # Perfil comparativo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Perfil Creativo Comparativo", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="radar-comparativo-creation")
                                ]
                            )
                        ], width=6),
                        
                        # Gráfico de asistencias por zona
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Asistencias por Zona", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="asistencias-zonas-chart")
                                ]
                            )
                        ], width=6),
                    ]),
                ]
            ),
            
            # Stores para los datos
            dcc.Store(id="filtered-data-store-creacion"),
            dcc.Store(id="jugador-seleccionado-store-creacion"),
        ]
    )
])

# Callback para cargar opciones de filtros iniciales
@callback(
    [Output("posicion-filter-creacion", "options"),
     Output("liga-filter-creacion", "options")],
    Input("creation-content", "children"),
    prevent_initial_call=False
)
def load_filter_options(dummy):
    df = data_manager.get_data()
    
    # Obtener opciones de posición
    posiciones = []
    if not df.empty and 'POSICIÓN' in df.columns:
        posiciones = [{"label": pos, "value": pos} for pos in sorted(df['POSICIÓN'].unique())]
    
    # Obtener opciones de liga exclusivamente de la columna LIGA
    ligas = []
    if not df.empty and 'LIGA' in df.columns:
        # Asegurarse de eliminar valores nulos o vacíos
        ligas_unicas = [liga for liga in df['LIGA'].dropna().unique() if liga and str(liga).strip()]
        # Añadir la opción "Todas las ligas" al principio
        ligas = [{"label": "Todas las ligas", "value": "todas"}] + [{"label": liga, "value": liga} for liga in sorted(ligas_unicas)]
    
    return posiciones, ligas

# Callback para actualizar opciones de equipos basados en ligas seleccionadas
@callback(
    Output("equipo-filter-creacion", "options"),
    Input("liga-filter-creacion", "value"),
    prevent_initial_call=False
)
def update_team_options(ligas_seleccionadas):
    # Si no hay ligas seleccionadas, devolver opción "Todos los equipos"
    if not ligas_seleccionadas:
        return [{"label": "Todos los equipos", "value": "todos"}]
    
    df = data_manager.get_data()
    
    # Verificar que tenemos la columna EQUIPO
    if 'EQUIPO' not in df.columns:
        # Buscar alternativas como CLUB si EQUIPO no está disponible
        equipo_col = None
        for col in ['CLUB', 'TEAM', 'EQUIPO']:
            if col in df.columns:
                equipo_col = col
                break
        
        if not equipo_col:
            return [{"label": "Datos de equipo no disponibles", "value": "no_data"}]
    else:
        equipo_col = 'EQUIPO'
    
    # Si "todas" está en las ligas seleccionadas, no filtrar por liga
    if "todas" in ligas_seleccionadas:
        equipos = [{"label": "Todos los equipos", "value": "todos"}]
        equipos_unicos = df[equipo_col].dropna().unique()
        equipos.extend([{"label": equipo, "value": equipo} for equipo in sorted(equipos_unicos)])
        return equipos
    
    # Filtrar el DataFrame para incluir solo las ligas seleccionadas
    filtered_df = df[df['LIGA'].isin(ligas_seleccionadas)]
    
    # Obtener equipos únicos de las ligas seleccionadas
    equipos = []
    if not filtered_df.empty:
        # Añadir opción para todos los equipos al principio
        equipos = [{"label": "Todos los equipos", "value": "todos"}]
        
        # Obtener equipos únicos de las ligas seleccionadas
        equipos_unicos = filtered_df[equipo_col].dropna().unique()
        equipos.extend([{"label": equipo, "value": equipo} for equipo in sorted(equipos_unicos)])
    
    return equipos

# Callback para aplicar filtros y actualizar datos
@callback(
    Output("filtered-data-store-creacion", "data"),
    [Input("aplicar-filtros-btn-creacion", "n_clicks")],
    [State("posicion-filter-creacion", "value"),
     State("liga-filter-creacion", "value"),
     State("equipo-filter-creacion", "value"),
     State("metricas-filter-creacion", "value"),
     State("edad-range-slider-creacion", "value"),
     State("minutos-slider-creacion", "value")],
    prevent_initial_call=False
)
def filter_data(n_clicks, posiciones, ligas, equipos, metricas_seleccionadas, rango_edad, min_minutos):
    df = data_manager.get_data()
    
    if df.empty:
        return []
    
    # Si no se ha hecho clic (carga inicial), retornar todos los datos con filtros por defecto
    if not n_clicks:
        # Para la carga inicial, aplicar solo filtros básicos por defecto
        filtered_df = df.copy()
        
        # Aplicar filtro de minutos jugados por defecto
        if 'MINUTOS JUGADOS' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['MINUTOS JUGADOS'] >= 500]  # Valor por defecto
        
        # Seleccionar columnas básicas siempre necesarias
        cols_to_include = ['NOMBRE', 'POSICIÓN']
        
        if 'EDAD' in filtered_df.columns:
            cols_to_include.append('EDAD')
        
        if 'MINUTOS JUGADOS' in filtered_df.columns:
            cols_to_include.append('MINUTOS JUGADOS')
            
        # Incluir columnas de equipo y liga si están disponibles
        if 'LIGA' in filtered_df.columns:
            cols_to_include.append('LIGA')
        
        for col in ['EQUIPO', 'CLUB', 'TEAM']:
            if col in filtered_df.columns:
                cols_to_include.append(col)
                break
        
        # Incluir métricas por defecto
        default_metrics = ["ASISTENCIAS POR 90", "PASES EXITOSOS POR 90", "XG CREADA CON PASES"]
        for metric in default_metrics:
            if metric in filtered_df.columns:
                cols_to_include.append(metric)
        
        # Métricas específicas para creación de juego
        # Asegurarse de incluir métricas importantes de pases y asistencias
        important_metrics = [
            "PASES EXITOSOS HACIA ÚLTIMO TERCIO", "PASES EXITOSOS HACIA ÁREA RIVAL",
            "CENTROS ALTOS EXITOSOS", "CENTROS RASOS EXITOSOS", 
            "ASISTENCIAS DESDE ÚLTIMO TERCIO", "ASISTENCIAS DESDE BANDA DERECHA", 
            "ASISTENCIAS DESDE BANDA IZQUIERDA"
        ]
        
        for metric in important_metrics:
            if metric in filtered_df.columns and metric not in cols_to_include:
                cols_to_include.append(metric)
        
        # Filtrar DataFrame
        filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)
        
        # No hacemos selección de columnas para mantener todos los datos
        # filtered_df = filtered_df[cols_to_include]
        
        return {
            'data': filtered_df.to_dict('records'),
            'selected_metrics': default_metrics
        }
    
    # Si se ha hecho clic, aplicar todos los filtros
    # Hacer una copia de los datos
    filtered_df = df.copy()
    
    # Aplicar filtro de posición
    if posiciones and len(posiciones) > 0:
        filtered_df = filtered_df[filtered_df['POSICIÓN'].isin(posiciones)]
    
    # Aplicar filtro de liga solo en la columna LIGA (excluyendo "todas")
    if ligas and len(ligas) > 0 and 'LIGA' in filtered_df.columns and 'todas' not in ligas:
        filtered_df = filtered_df[filtered_df['LIGA'].isin(ligas)]
    
    # Aplicar filtro de equipos si no está en "todos"
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
        "ASISTENCIAS POR 90", "PASES EXITOSOS POR 90", "XG CREADA CON PASES",
        "PASES EXITOSOS HACIA ÚLTIMO TERCIO", "PASES EXITOSOS HACIA ÁREA RIVAL",
        "CENTROS ALTOS EXITOSOS", "CENTROS RASOS EXITOSOS", 
        "ASISTENCIAS DESDE ÚLTIMO TERCIO", "ASISTENCIAS DESDE BANDA DERECHA", 
        "ASISTENCIAS DESDE BANDA IZQUIERDA"
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
    else:
        # Si no hay métricas seleccionadas, incluir todas las disponibles
        for metric in metricas_creacion:
            if metric in filtered_df.columns and metric not in cols_to_include:
                cols_to_include.append(metric)
    
    # Filtrar solo columnas que existen en el DataFrame
    cols_to_include = [col for col in cols_to_include if col in filtered_df.columns]
    
    # Seleccionar columnas y filtrar filas con valores faltantes
    # No filtramos columnas para mantener todos los datos disponibles
    # filtered_df = filtered_df[cols_to_include]
    filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)  # Permitir algunos valores faltantes
    
    # Convertir a diccionario y devolver
    return {
        'data': filtered_df.to_dict('records'),
        'selected_metrics': metricas_seleccionadas if metricas_seleccionadas else essential_metrics
    }

# Callback para crear tabla interactiva de top creadores
@callback(
    Output("top-creadores-table", "children"),
    Input("filtered-data-store-creacion", "data")
)
def create_top_creadores_table(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_creacion
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Crear un score de creación basado en las métricas relacionadas con pases y asistencias
    metric_weights = {
        'ASISTENCIAS POR 90': 1.0,
        'XG CREADA CON PASES': 1.0,
        'PASES EXITOSOS POR 90': 0.7,
        'PASES EXITOSOS HACIA ÚLTIMO TERCIO': 0.8,
        'PASES EXITOSOS HACIA ÁREA RIVAL': 0.9,
        'CENTROS ALTOS EXITOSOS': 0.6,
        'CENTROS RASOS EXITOSOS': 0.6,
        'ASISTENCIAS EN POSESIÓN': 0.7,
        'ASISTENCIAS CONTRAATAQUE': 0.7
    }
    
    # Inicializar score de creación
    creation_score = pd.Series(0, index=df.index)
    metrics_used = []
    
    # Buscar cualquier métrica de asistencias disponible
    asistencias_metric = None
    for metric in ['ASISTENCIAS POR 90', 'ASISTENCIAS PASE RASO', 'ASISTENCIAS CENTRO ALTO', 
                  'ASISTENCIAS DESDE ÚLTIMO TERCIO', 'XG CREADA CON PASES']:
        if metric in df.columns:
            asistencias_metric = metric
            break
    
    # Si no se encuentra ninguna métrica de asistencias, usar la primera métrica de pases disponible
    if not asistencias_metric:
        for metric in ['PASES EXITOSOS POR 90', 'PASES EXITOSOS HACIA ÚLTIMO TERCIO', 
                      'PASES EXITOSOS HACIA ÁREA RIVAL']:
            if metric in df.columns:
                asistencias_metric = metric
                break
    
    if not asistencias_metric:
        return html.Div("No se encontraron métricas de creación en los datos disponibles.", 
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Calcular el score de creación
    for metric, weight in metric_weights.items():
        if metric in df.columns:
            metrics_used.append(metric)
            # Normalizar la métrica (0-1) y añadirla al score con su peso
            if df[metric].max() > df[metric].min():
                normalized = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                creation_score += normalized * weight
    
    # Si no hay métricas con pesos, usar el asistencias_metric
    if not metrics_used:
        metrics_used.append(asistencias_metric)
        if df[asistencias_metric].max() > df[asistencias_metric].min():
            normalized = (df[asistencias_metric] - df[asistencias_metric].min()) / (df[asistencias_metric].max() - df[asistencias_metric].min())
            creation_score += normalized
    
    df['Creation_Score'] = creation_score
    
    # Ordenar por score de creación y tomar los 10 mejores
    top_10 = df.sort_values('Creation_Score', ascending=False).head(10)
    
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
    
    # Añadir la métrica principal de asistencias
    display_cols.append(asistencias_metric)
    data_cols.append(asistencias_metric)
    
    # Añadir otras métricas seleccionadas relacionadas con creación
    creation_key_metrics = ['PASES EXITOSOS POR 90', 'PASES EXITOSOS HACIA ÚLTIMO TERCIO', 'XG CREADA CON PASES']
    for metric in creation_key_metrics:
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
                if 'ASISTENCIAS' in col or 'PASES' in col or 'XG' in col:
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
            id={"type": "jugador-row-creacion", "index": player_name},
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
        id="tabla-creadores"
    )
    
    return table

# Callback para manejar clics en filas de jugadores
@callback(
    [Output("jugador-seleccionado-store-creacion", "data"),
     Output("jugador-seleccionado-info-creacion", "children"),
     Output("jugador-seleccionado-info-creacion", "style")],
    [Input({"type": "jugador-row-creacion", "index": ALL}, "n_clicks")],
    [State({"type": "jugador-row-creacion", "index": ALL}, "id"),
     State("filtered-data-store-creacion", "data")]
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
    pases_cols = [col for col in player_data.columns if 'PASES' in col]
    asistencias_cols = [col for col in player_data.columns if 'ASISTENCIAS' in col]
    xg_cols = [col for col in player_data.columns if 'XG' in col]
    
    # Seleccionar las métricas más relevantes para cada categoría
    pases_top = []
    for col in ['PASES EXITOSOS POR 90', 'PASES EXITOSOS HACIA ÚLTIMO TERCIO', 'PASES EXITOSOS HACIA ÁREA RIVAL']:
        if col in pases_cols:
            pases_top.append(col)
    
    asistencias_top = []
    for col in ['ASISTENCIAS POR 90', 'ASISTENCIAS DESDE ÚLTIMO TERCIO', 'XG CREADA CON PASES']:
        if col in player_data.columns:
            asistencias_top.append(col)
    
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
            
            # Columna de métricas de pases
            html.Div([
                html.H4("Pases", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('PASES', '').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in pases_top[:5]]  # Limitar a 5 métricas
            ], style={"width": "25%", "padding": "0 15px"}),
            
            # Columna de métricas de asistencias
            html.Div([
                html.H4("Asistencias y xG", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('ASISTENCIAS', '').replace('XG', 'xG').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in asistencias_top[:5]]  # Limitar a 5 métricas
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

# Callback para crear visualización de pases
@callback(
    Output("pases-visualization", "children"),
    Input("jugador-seleccionado-store-creacion", "data"),
    State("filtered-data-store-creacion", "data")
)
def create_pases_visualization(jugador_data, filtered_data):
    if not jugador_data:
        return html.Div("Selecciona un jugador de la tabla para ver su mapa de pases.", 
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
    player_df = df[df["NOMBRE"] == player_name].copy()
    
    if player_df.empty:
        return html.Div(f"No se encontraron datos para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Definir categorías de pases
    pases_categorias = [
        # Pases por origen
        ('PASES DESDE PRIMER TERCIO', 'Primer Tercio'),
        ('PASES DESDE TERCIO MEDIO', 'Tercio Medio'),
        ('PASES DESDE ÚLTIMO TERCIO', 'Último Tercio'),
        # Pases por destino
        ('PASES EXITOSOS HACIA PRIMER TERCIO', 'Hacia Primer Tercio'),
        ('PASES EXITOSOS HACIA TERCIO MEDIO', 'Hacia Tercio Medio'),
        ('PASES EXITOSOS HACIA ÚLTIMO TERCIO', 'Hacia Último Tercio'),
        ('PASES EXITOSOS HACIA ÁREA RIVAL', 'Hacia Área'),
        # Pases por zona del campo
        ('PASES DESDE CARRIL CENTRAL', 'Carril Central'),
        ('PASES HACIA CARRIL CENTRAL', 'Hacia Carril Central'),
        ('PASES DESDE BANDA DERECHA', 'Banda Derecha'),
        ('PASES HACIA BANDA DERECHA', 'Hacia Banda Derecha'),
        ('PASES DESDE BANDA IZQUIERDA', 'Banda Izquierda'),
        ('PASES HACIA BANDA IZQUIERDA', 'Hacia Banda Izquierda'),
        # Tipos de pase
        ('PASES RASOS EXITOSOS', 'Rasos'),
        ('PASES DIAGONALES EXITOSOS', 'Diagonales'),
        ('PASES ELEVADOS EXITOSOS', 'Elevados'),
        ('PASES AÉREOS CORTOS EXITOSOS', 'Aéreos Cortos'),
        ('CENTROS ALTOS EXITOSOS', 'Centros Altos'),
        ('CENTROS RASOS EXITOSOS', 'Centros Rasos')
    ]
    
    # Extraer datos disponibles
    pases_data = []
    for col, label in pases_categorias:
        if col in player_df.columns and not pd.isna(player_df[col].values[0]):
            value = player_df[col].values[0]
            pases_data.append({"Categoría": label, "Valor": value})
    
    if not pases_data:
        return html.Div(f"No hay datos de pases disponibles para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Convertir a DataFrame y ordenar
    pases_df = pd.DataFrame(pases_data)
    pases_df = pases_df.sort_values("Valor", ascending=False)
    
    # Crear gráfico de barras horizontal
    fig = px.bar(
        pases_df,
        y="Categoría",
        x="Valor",
        title=f"Distribución de Pases de {player_name}",
        orientation='h',
        color="Valor",
        color_continuous_scale="Viridis"
    )
    
    # Configurar layout
    fig.update_layout(
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Pases por 90 minutos", font=dict(color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            title=dict(text="", font=dict(color="white")),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        coloraxis_showscale=False,
        height=600,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Añadir línea vertical para el promedio de la posición
    player_position = player_df["POSICIÓN"].values[0]
    position_df = df[df["POSICIÓN"] == player_position]
    
    # Añadir análisis textual
    top_categories = pases_df.head(3)["Categoría"].tolist()
    
    analysis = html.Div([
        html.H4(f"Análisis de Distribución de Pases de {player_name}", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"El gráfico muestra los diferentes tipos y zonas de pases de {player_name}. ",
            "Se destaca especialmente en: ",
            html.Strong(", ".join(top_categories), style={"color": "#FFFF00"})
        ]),
        html.P([
            "Estos datos representan valores promedio por 90 minutos jugados, ",
            "lo que permite comparar jugadores independientemente de su tiempo en campo."
        ]),
        html.P([
            "Un jugador con alta capacidad de creación generalmente muestra valores elevados en ",
            "pases hacia el último tercio y hacia el área, así como en centros efectivos y pases progresivos."
        ])
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        analysis
    ])

# Callback para crear radar comparativo para creación
@callback(
    Output("radar-comparativo-creation", "children"),
    Input("jugador-seleccionado-store-creacion", "data"),
    State("filtered-data-store-creacion", "data")
)
def create_radar_comparativo_creation(jugador_data, filtered_data):
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
    
    # Identificar métricas de creación relevantes disponibles
    creation_metrics = []
    for category, keyword in [
        ("Asistencias", "ASISTENCIAS POR 90"),
        ("xG Creada", "XG CREADA CON PASES"),
        ("Pases Último Tercio", "PASES EXITOSOS HACIA ÚLTIMO TERCIO"),
        ("Pases Área", "PASES EXITOSOS HACIA ÁREA"),
        ("Centros", "CENTROS ALTOS EXITOSOS")
    ]:
        # Verificar si la métrica está disponible
        if keyword in df.columns:
            creation_metrics.append(keyword)
    
    # Si no hay suficientes métricas, mostrar mensaje
    if len(creation_metrics) < 3:
        # Usar métricas comunes si no hay suficientes
        alternative_metrics = [
            "PASES EXITOSOS POR 90", 
            "PASES RASOS EXITOSOS", 
            "PASES DIAGONALES EXITOSOS",
            "PASES DESDE ÚLTIMO TERCIO"
        ]
        
        for metric in alternative_metrics:
            if metric in df.columns and metric not in creation_metrics:
                creation_metrics.append(metric)
        
        if len(creation_metrics) < 3:
            return html.Div("No se encontraron suficientes métricas de creación para crear el radar comparativo.", 
                        style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Limitar a 5 métricas para mejor visualización
    creation_metrics = creation_metrics[:5]
    
    # Calcular similitud para encontrar jugadores comparables
    # Método simple: distancia euclidiana normalizada
    
    # Normalizar métricas
    for metric in creation_metrics:
        if metric in similar_players.columns:
            metric_min = df[metric].min()
            metric_max = df[metric].max()
            if metric_max > metric_min:
                similar_players[f"{metric}_norm"] = (similar_players[metric] - metric_min) / (metric_max - metric_min)
            else:
                similar_players[f"{metric}_norm"] = 0.5
    
    # Normalizar métricas para el jugador seleccionado
    player_normalized = {}
    for metric in creation_metrics:
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
        for metric in creation_metrics:
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
    for metric in creation_metrics:
        if metric in player_df.columns:
            player_values.append(player_df[metric].values[0])
        else:
            player_values.append(0)
    
    radar_data.append({
        "player": player_name,
        "values": player_values,
        "metrics": creation_metrics
    })
    
    # Añadir jugadores similares
    for similar_name in similar_names:
        similar_df = df[df["NOMBRE"] == similar_name]
        if not similar_df.empty:
            similar_values = []
            for metric in creation_metrics:
                if metric in similar_df.columns:
                    similar_values.append(similar_df[metric].values[0])
                else:
                    similar_values.append(0)
            
            radar_data.append({
                "player": similar_name,
                "values": similar_values,
                "metrics": creation_metrics
            })
    
    # Crear gráfico de radar
    fig = go.Figure()
    
    # Formatear nombres de métricas para mejor visualización
    formatted_metrics = []
    for metric in creation_metrics:
        # Simplificar los nombres de las métricas
        formatted = metric.replace("PASES EXITOSOS", "Pases")
        formatted = formatted.replace("HACIA", "→")
        formatted = formatted.replace("ÚLTIMO", "Últ.")
        formatted = formatted.replace("ASISTENCIAS", "Asis.")
        formatted = formatted.replace("CENTROS ALTOS", "Centros")
        formatted = formatted.replace("XG CREADA CON PASES", "xG Creada")
        formatted = formatted.replace("POR 90", "/90")
        formatted_metrics.append(formatted)
    
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
            y=-0.25,  # Posicionado más abajo para evitar superposición
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        height=450,
        margin=dict(l=80, r=80, t=50, b=120)
    )
    
    # Añadir análisis comparativo
    analysis = html.Div([
        html.H4(f"Comparación de {player_name} con jugadores similares", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"Se ha comparado el perfil creativo de {player_name} con otros jugadores de la posición {player_position}. ",
            f"Los jugadores más similares son ",
            html.Strong(f"{similar_names[0]}", style={"color": colors[1]}),
            " y ",
            html.Strong(f"{similar_names[1]}", style={"color": colors[2]}) if len(similar_names) > 1 else "",
            "."
        ]),
        html.P([
            "El gráfico radar muestra cómo se comparan los jugadores en varias métricas creativas clave. ",
            "Cuanto mayor sea el área cubierta, mejor es la capacidad creativa general del jugador."
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

# Callback para crear gráfico de asistencias por zona
@callback(
    Output("asistencias-zonas-chart", "children"),
    Input("jugador-seleccionado-store-creacion", "data"),
    State("filtered-data-store-creacion", "data")
)
def create_asistencias_zonas_chart(jugador_data, filtered_data):
    if not jugador_data:
        return html.Div("Selecciona un jugador de la tabla para ver sus asistencias por zona.", 
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
    player_df = df[df["NOMBRE"] == player_name].copy()
    
    if player_df.empty:
        return html.Div(f"No se encontraron datos para {player_name}.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Definir zonas de asistencias
    asistencias_zonas = [
        ('ASISTENCIAS DESDE MEDIAPUNTA', 'Mediapunta'),
        ('ASISTENCIAS DESDE MEDIOCENTRO', 'Mediocentro'),
        ('ASISTENCIAS DESDE BANDA DERECHA', 'Banda Derecha'),
        ('ASISTENCIAS DESDE BANDA IZQUIERDA', 'Banda Izquierda'),
        ('ASISTENCIAS DESDE ÚLTIMO TERCIO', 'Último Tercio'),
        ('ASISTENCIAS DESDE CARRIL CENTRAL', 'Carril Central'),
        ('ASISTENCIAS POR CÓRNER', 'Córner'),
        ('ASISTENCIAS CONTRAATAQUE', 'Contraataque'),
        ('ASISTENCIAS EN POSESIÓN', 'En Posesión'),
        ('ASISTENCIAS JUGADAS BALÓN PARADO', 'Balón Parado')
    ]
    
    # Extraer datos disponibles
    asistencias_data = []
    for col, label in asistencias_zonas:
        if col in player_df.columns and not pd.isna(player_df[col].values[0]):
            value = player_df[col].values[0]
            asistencias_data.append({"Zona": label, "Valor": value})
    
    if not asistencias_data:
        # Si no hay datos específicos de asistencias por zona, mostrar alternativas
        if "ASISTENCIAS POR 90" in player_df.columns:
            asistencias_por_90 = player_df["ASISTENCIAS POR 90"].values[0]
            return html.Div([
                html.H4(f"Asistencias de {player_name}", style={"color": "#00BFFF", "margin-bottom": "15px"}),
                html.P(f"Asistencias por 90 minutos: {asistencias_por_90:.2f}"),
                html.P("No hay datos detallados sobre la distribución de asistencias por zona para este jugador.")
            ], style={"padding": "20px", "background-color": "rgba(0,0,0,0.2)", "border-radius": "5px"})
        else:
            return html.Div(f"No hay datos de asistencias disponibles para {player_name}.", 
                           style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Ordenar por valor
    asistencias_data = sorted(asistencias_data, key=lambda x: x["Valor"], reverse=True)
    
    # Convertir a formato para gráfico de pie
    labels = [item["Zona"] for item in asistencias_data]
    values = [item["Valor"] for item in asistencias_data]
    
    # Crear gráfico de pie
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(
                colors=px.colors.sequential.Viridis,
                line=dict(color='rgba(0,0,0,0)', width=2)
            ),
            textfont=dict(color="white", size=12),
            hoverinfo='label+value+percent',
            hovertemplate='%{label}: %{value:.2f} asistencias<br>%{percent}<extra></extra>'
        )
    ])
    
    # Configurar layout
    fig.update_layout(
        title=f"Distribución de Asistencias por Zona - {player_name}",
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )
    
    # Añadir análisis textual
    top_zones = [asistencias_data[i]["Zona"] for i in range(min(3, len(asistencias_data)))]
    total_asistencias = sum(values)
    
    analysis = html.Div([
        html.H4(f"Análisis de Asistencias de {player_name}", style={"color": "#00BFFF", "margin-top": "20px"}),
        html.P([
            f"{player_name} registra un total de {total_asistencias:.2f} asistencias por 90 minutos, ",
            "distribuidas principalmente desde: ",
            html.Strong(", ".join(top_zones), style={"color": "#FFFF00"})
        ]),
        html.P([
            "Esta distribución refleja su estilo de juego y posicionamiento habitual en el campo, ",
            "mostrando desde qué zonas es más efectivo creando oportunidades para sus compañeros."
        ])
    ], style={"margin-top": "20px", "background-color": "rgba(0,0,0,0.2)", "padding": "15px", "border-radius": "5px"})
    
    return html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        analysis
    ])
    
    # Callback para abrir el modal
@callback(
    Output("creacion-pdf-modal", "is_open"),
    [Input("creacion-pdf-button", "n_clicks"),
     Input("creacion-pdf-modal-close", "n_clicks"),
     Input("creacion-pdf-modal-generate", "n_clicks")],
    [State("creacion-pdf-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

# Callback para generar y descargar el PDF
@callback(
    Output("creacion-pdf-download", "data"),
    Input("creacion-pdf-modal-generate", "n_clicks"),
    [State("creacion-pdf-title", "value"),
     State("creacion-pdf-description", "value"),
     State("filtered-data-store-creacion", "data")],
    prevent_initial_call=True
)
def generate_pdf_creacion(n_clicks, title, description, filtered_data):
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
    elements.append(Paragraph(title or "Informe de Creación de Juego", title_style))
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
        for metric in ["ASISTENCIAS POR 90", "PASES EXITOSOS POR 90", "XG CREADA CON PASES"]:
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
                short_name = short_name.replace("ASISTENCIAS POR 90", "ASIS./90")
                short_name = short_name.replace("PASES EXITOSOS POR 90", "PASES/90")
                short_name = short_name.replace("XG CREADA CON PASES", "XG CREADA")
                short_name = short_name.replace("PASES EXITOSOS HACIA ÚLTIMO TERCIO", "PASES Ú.T.")
                short_name = short_name.replace("PASES EXITOSOS HACIA ÁREA RIVAL", "PASES ÁREA")
                
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
        "filename": f"informe_creacion_juego_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }