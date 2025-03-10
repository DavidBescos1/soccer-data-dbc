from dash import dcc, html, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from pdf_export import export_button, create_pdf_report
from components.data_manager import DataManager
from components.sidebar import create_sidebar
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir métricas de superación principales
metricas_superacion = [
    "OPONENTES SUPERADOS POR 90", 
    "OPONENTES SUPERADOS/90", 
    "DEFENSORES SUPERADOS POR 90",
    "OPONENTES SUPERADOS POR REGATE", 
    "REGATES EXITOSOS", 
    "OPONENTES SUPERADOS AL RECIBIR",
    "NÚMERO OPONENTES SUPERADOS AL RECIBIR", 
    "TOQUES EN ÚLTIMO TERCIO", 
    "TOQUES OFENSIVOS ÁREA RIVAL",
    "TOQUES EN POSESIÓN", 
    "TOQUES TRANSICIÓN OFENSIVA", 
    "DISPONIBILIDAD EN ÁREA",
    "TOQUES PROTEGIENDO BALÓN", 
    "OPONENTES SUPERADOS EN POSESIÓN POR 90", 
    "OPONENTES SUPERADOS CONTRAATAQUE POR 90", 
    "DISTANCIA REGATE HACIA PORTERÍA",
    "DISTANCIA CONDUCCIÓN HACIA PORTERÍA", 
    "TOTAL OPONENTES SUPERADOS", 
    "OPONENTES SUPERADOS PASE BAJO", 
    "OPONENTES SUPERADOS PASE DIAGONAL",
    "OPONENTES SUPERADOS PASE ELEVADO", 
    "OPONENTES SUPERADOS PASE AÉREO CORTO",
    "OPONENTES SUPERADOS DESDE TERCIO MEDIO", 
    "OPONENTES SUPERADOS HACIA ÚLTIMO TERCIO",
    "CENTROCAMPISTAS SUPERADOS POR 90", 
    "DEFENSAS SUPERADOS PASE RASO",
    "DEFENSAS SUPERADOS PASE DIAGONAL", 
    "DEFENSAS SUPERADOS PASE ELEVADO", 
    "DEFENSAS SUPERADOS REGATE", 
    "OPONENTES SUPERADOS EN POSESIÓN", 
    "OPONENTES SUPERADOS EN CONTRAATAQUE", 
    "NÚMERO JUGADAS NEUTRAS", 
    "NÚMERO JUGADAS REVERSIBLES", 
    "NÚMERO PÉRDIDAS", 
    "PÉRDIDAS CRÍTICAS POR 90",
    "OPONENTES GANADOS RIVAL TRAS PÉRDIDA", 
    "COMPAÑEROS DESPOSICIONADOS TRAS PÉRDIDA",
    "COMPAÑEROS BENEFICIADOS TRAS RECUPERACIÓN", 
    "OPONENTES ANULADOS TRAS RECUPERACIÓN",
    "OPONENTES SUPERADOS JUGADAS BALÓN PARADO", 
    "OPONENTES SUPERADOS DESDE MEDIAPUNTA",
    "OPONENTES SUPERADOS DESDE MEDIOCENTRO", 
    "OPONENTES SUPERADOS DESDE BANDA DERECHA",
    "OPONENTES SUPERADOS DESDE BANDA IZQUIERDA", 
    "OPONENTES SUPERADOS HACIA MEDIAPUNTA"
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
            html.H1("Análisis de Superación", 
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
                                id="posicion-filter-superacion",
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
                                id="liga-filter-superacion",
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
                                id="equipo-filter-superacion",
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
                                id="edad-range-slider-superacion",
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
                                id="minutos-slider-superacion",
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
                            html.Div(className="filter-title", children="Métricas de Superación"),
                            dcc.Dropdown(
                                id="metricas-filter-superacion",
                                options=[
                                    {"label": "OPONENTES SUPERADOS POR 90", "value": "OPONENTES SUPERADOS POR 90"},
                                    {"label": "REGATES EXITOSOS", "value": "REGATES EXITOSOS"},
                                    {"label": "OPONENTES SUPERADOS AL RECIBIR", "value": "OPONENTES SUPERADOS AL RECIBIR"},
                                    {"label": "TOQUES EN ÚLTIMO TERCIO", "value": "TOQUES EN ÚLTIMO TERCIO"},
                                    {"label": "TOQUES EN POSESIÓN", "value": "TOQUES EN POSESIÓN"},
                                    {"label": "TOQUES TRANSICIÓN OFENSIVA", "value": "TOQUES TRANSICIÓN OFENSIVA"},
                                    {"label": "OPONENTES SUPERADOS EN POSESIÓN", "value": "OPONENTES SUPERADOS EN POSESIÓN"},
                                    {"label": "OPONENTES SUPERADOS EN CONTRAATAQUE", "value": "OPONENTES SUPERADOS EN CONTRAATAQUE"},
                                    {"label": "NÚMERO PÉRDIDAS", "value": "NÚMERO PÉRDIDAS"},
                                    {"label": "PÉRDIDAS CRÍTICAS POR 90", "value": "PÉRDIDAS CRÍTICAS POR 90"},
                                    {"label": "TOTAL OPONENTES SUPERADOS", "value": "TOTAL OPONENTES SUPERADOS"},
                                    {"label": "DEFENSORES SUPERADOS POR 90", "value": "DEFENSORES SUPERADOS POR 90"},
                                    {"label": "CENTROCAMPISTAS SUPERADOS POR 90", "value": "CENTROCAMPISTAS SUPERADOS POR 90"},
                                    {"label": "DISTANCIA REGATE HACIA PORTERÍA", "value": "DISTANCIA REGATE HACIA PORTERÍA"},
                                    {"label": "DISTANCIA CONDUCCIÓN HACIA PORTERÍA", "value": "DISTANCIA CONDUCCIÓN HACIA PORTERÍA"}
                                ],
                                value=["REGATES EXITOSOS", "OPONENTES SUPERADOS POR 90", "NÚMERO PÉRDIDAS"],  # Valores por defecto
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
                                id="aplicar-filtros-btn-superacion",
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
                    export_button("superacion-pdf", "Informe de Análisis de Superación"),
                ], width={"size": 4, "offset": 4}),
            ], className="mb-4"),
            
            # Contenido de la página - SIMPLIFICADO con los 3 gráficos solicitados
            html.Div(
                id="superacion-content",
                children=[
                    # Primera fila: Tabla de los mejores regateadores
                    dbc.Row([
                        # Ranking de mejores regateadores a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Top 10 Regateadores", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="top-regateadores-table"),
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
                                id="jugador-seleccionado-info-superacion",
                                className="graph-container",
                                style={"display": "none"}  # Inicialmente oculto
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Tercera fila: Gráfico de eficiencia en posesión vs contraataque
                    dbc.Row([
                        # Gráfico de eficiencia en posesión a ancho completo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Eficiencia en Posesión vs. Contraataque", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="eficiencia-posesion-chart")
                                ]
                            )
                        ], width=12),
                    ], className="mb-4"),
                    
                    # Cuarta fila: Gráficos de superación por zona y perfil comparativo
                    dbc.Row([
                        # Superación por zonas
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Superación por Zonas", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="superacion-zonas-chart")
                                ]
                            )
                        ], width=6),
                        
                        # Perfil de superación comparativo
                        dbc.Col([
                            html.Div(
                                className="graph-container",
                                children=[
                                    html.H3("Perfil de Superación Comparativo", 
                                           style={"color": "#00BFFF", "margin-bottom": "15px", "text-align": "center"}),
                                    html.Div(id="radar-comparativo-superacion")
                                ]
                            )
                        ], width=6),
                    ], className="mb-4"),
                ]
            ),
            
            # Stores para los datos
            dcc.Store(id="filtered-data-store-superacion"),
            dcc.Store(id="jugador-seleccionado-store-superacion"),
        ]
    )
])

# Callback para cargar opciones de filtros iniciales
@callback(
    [Output("posicion-filter-superacion", "options"),
     Output("liga-filter-superacion", "options")],
    Input("posicion-filter-superacion", "id")  # Trigger dummy
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
    Output("equipo-filter-superacion", "options"),
    Input("liga-filter-superacion", "value")
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
    Output("filtered-data-store-superacion", "data"),
    Input("aplicar-filtros-btn-superacion", "n_clicks"),
    [State("posicion-filter-superacion", "value"),
     State("liga-filter-superacion", "value"),
     State("equipo-filter-superacion", "value"),
     State("metricas-filter-superacion", "value"),
     State("edad-range-slider-superacion", "value"),
     State("minutos-slider-superacion", "value")],
    prevent_initial_call=False  # Permitir ejecución inicial automática
)
def filter_data(n_clicks, posiciones, ligas, equipos, metricas_seleccionadas, rango_edad, min_minutos):
    df = data_manager.get_data()
    
    if df.empty:
        return []
    
    # Si no se ha hecho clic y es la carga inicial, aplicar filtros por defecto
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
        
        # Métricas por defecto
        default_metrics = ["REGATES EXITOSOS", "OPONENTES SUPERADOS POR 90", "NÚMERO PÉRDIDAS"]
        
        # No filtramos columnas para mantener todos los datos disponibles
        filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)  # Permitir algunos valores faltantes
        
        return {
            'data': filtered_df.to_dict('records'),
            'selected_metrics': default_metrics
        }
    
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
    elif 'EDAD' in filtered_df.columns:  # Alternativa si ya existe la columna EDAD
        min_edad, max_edad = rango_edad
        filtered_df = filtered_df[(filtered_df['EDAD'] >= min_edad) & (filtered_df['EDAD'] <= max_edad)]
    
    # Aplicar filtro de minutos jugados
    if 'MINUTOS JUGADOS' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['MINUTOS JUGADOS'] >= min_minutos]
    
    # Seleccionar columnas básicas siempre necesarias
    cols_to_include = ['NOMBRE', 'POSICIÓN', 'EDAD', 'MINUTOS JUGADOS']
    
    # Incluir columnas de equipo y liga si están disponibles
    if 'LIGA' in filtered_df.columns:
        cols_to_include.append('LIGA')
    
    for col in ['EQUIPO', 'CLUB', 'TEAM']:
        if col in filtered_df.columns:
            cols_to_include.append(col)
            break
    
    # Incluir solo las métricas de superación seleccionadas
    if metricas_seleccionadas and len(metricas_seleccionadas) > 0:
        # Añadir solo las métricas seleccionadas que existen en el DataFrame
        for col in metricas_seleccionadas:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    else:
        # Si no hay métricas seleccionadas, incluir todas las disponibles
        for col in metricas_superacion:
            if col in filtered_df.columns:
                cols_to_include.append(col)
    
    # Seleccionar solo columnas disponibles
    cols_to_include = [col for col in cols_to_include if col in filtered_df.columns]
    
    # NO filtramos columnas para mantener todos los datos disponibles
    # filtered_df = filtered_df[cols_to_include]
    filtered_df = filtered_df.dropna(thresh=len(cols_to_include) - 3)  # Permitir algunos valores faltantes
    
    # Guardar métricas seleccionadas en los datos (para usar en otros callbacks)
    selected_metrics_data = {
        'data': filtered_df.to_dict('records'),
        'selected_metrics': metricas_seleccionadas if metricas_seleccionadas else metricas_superacion
    }
    
    return selected_metrics_data

# Callback para crear tabla de top regateadores
@callback(
    Output("top-regateadores-table", "children"),
    Input("filtered-data-store-superacion", "data")
)
def create_top_regateadores_table(filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Extraer datos y métricas seleccionadas
    if isinstance(filtered_data, dict) and 'data' in filtered_data:
        raw_data = filtered_data['data']
        selected_metrics = filtered_data.get('selected_metrics', [])
    else:
        raw_data = filtered_data
        selected_metrics = metricas_superacion
    
    # Convertir a DataFrame
    df = pd.DataFrame(raw_data)
    
    if df.empty:
        return html.Div("No hay datos que cumplan con los criterios de filtrado.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Lista de posiciones ofensivas a mostrar por defecto
    posiciones_ofensivas = [
        'EXTREMO DERECHO', 'EXTREMO IZQUIERDO', 
        'DELANTERO', 'DELANTERO CENTRO', 'SEGUNDO DELANTERO',
        'MEDIAPUNTA', 'MEDIOCENTRO OFENSIVO',
        'INTERIOR DERECHO', 'INTERIOR IZQUIERDO'
    ]
    
    # Filtrar el DataFrame para incluir solo posiciones ofensivas
    # Primero verificamos si hay posiciones ofensivas en los datos
    posiciones_disponibles = set(df['POSICIÓN'].unique())
    posiciones_a_incluir = [pos for pos in posiciones_ofensivas if pos in posiciones_disponibles]
    
    # Si no hay ninguna posición ofensiva, usamos todas las posiciones
    if not posiciones_a_incluir:
        df_filtrado = df.copy()
    else:
        df_filtrado = df[df['POSICIÓN'].isin(posiciones_a_incluir)].copy()
        # Si el filtrado resulta en un DataFrame vacío, usamos el original
        if df_filtrado.empty:
            df_filtrado = df.copy()
    
    # Crear un score de regates basado en las métricas relacionadas con superación
    metric_weights = {
        'REGATES EXITOSOS': 1.0,
        'OPONENTES SUPERADOS POR 90': 1.0,
        'OPONENTES SUPERADOS AL RECIBIR': 0.8,
        'TOTAL OPONENTES SUPERADOS': 0.9,
        'DEFENSORES SUPERADOS POR 90': 0.8,
        'OPONENTES SUPERADOS EN POSESIÓN': 0.7,
        'OPONENTES SUPERADOS EN CONTRAATAQUE': 0.7,
        'DISTANCIA REGATE HACIA PORTERÍA': 0.6,
        'DISTANCIA CONDUCCIÓN HACIA PORTERÍA': 0.6
    }
    
    # Definir métricas de pérdidas (valores negativos en el score)
    negative_metrics = {
        'NÚMERO PÉRDIDAS': -0.4,
        'PÉRDIDAS CRÍTICAS POR 90': -0.7
    }
    
    # Inicializar score de regates
    dribble_score = pd.Series(0, index=df_filtrado.index)
    metrics_used = []
    
    # Buscar cualquier métrica de regates disponible
    regates_metric = None
    for metric in ['REGATES EXITOSOS', 'OPONENTES SUPERADOS POR 90', 'TOTAL OPONENTES SUPERADOS']:
        if metric in df_filtrado.columns:
            regates_metric = metric
            break
    
    if not regates_metric:
        return html.Div("No se encontraron métricas de regates en los datos disponibles.", 
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Calcular el score de regates con métricas positivas
    for metric, weight in metric_weights.items():
        if metric in df_filtrado.columns:
            metrics_used.append(metric)
            # Normalizar la métrica (0-1) y añadirla al score con su peso
            if df_filtrado[metric].max() > df_filtrado[metric].min():
                normalized = (df_filtrado[metric] - df_filtrado[metric].min()) / (df_filtrado[metric].max() - df_filtrado[metric].min())
                dribble_score += normalized * weight
    
    # Restar métricas negativas (pérdidas)
    for metric, weight in negative_metrics.items():
        if metric in df_filtrado.columns:
            metrics_used.append(metric)
            if df_filtrado[metric].max() > df_filtrado[metric].min():
                normalized = (df_filtrado[metric] - df_filtrado[metric].min()) / (df_filtrado[metric].max() - df_filtrado[metric].min())
                dribble_score += normalized * weight  # weight es negativo
    
    # Si no hay métricas con pesos, usar el regates_metric
    if not metrics_used:
        metrics_used.append(regates_metric)
        if df_filtrado[regates_metric].max() > df_filtrado[regates_metric].min():
            normalized = (df_filtrado[regates_metric] - df_filtrado[regates_metric].min()) / (df_filtrado[regates_metric].max() - df_filtrado[regates_metric].min())
            dribble_score += normalized
    
    df_filtrado['Dribble_Score'] = dribble_score
    
    # Ordenar por score de regates y tomar los 10 mejores
    top_10 = df_filtrado.sort_values('Dribble_Score', ascending=False).head(10)
    
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
    
    # Añadir la métrica principal de regates
    display_cols.append(regates_metric)
    data_cols.append(regates_metric)
    
    # Añadir otras métricas seleccionadas relacionadas con superación y pérdidas
    key_metrics = [
        'OPONENTES SUPERADOS POR 90', 
        'NÚMERO PÉRDIDAS', 
        'DEFENSORES SUPERADOS POR 90',
        'OPONENTES SUPERADOS EN CONTRAATAQUE'
    ]
    
    for metric in key_metrics:
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
                if 'OPONENTES' in col or 'REGATES' in col or 'PÉRDIDAS' in col or 'DISTANCIA' in col:
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
            id={"type": "jugador-row-superacion", "index": player_name},
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
        id="tabla-regateadores"
    )
    
    return table

# Callback para manejar clics en filas de jugadores
@callback(
    [Output("jugador-seleccionado-store-superacion", "data"),
     Output("jugador-seleccionado-info-superacion", "children"),
     Output("jugador-seleccionado-info-superacion", "style")],
    [Input({"type": "jugador-row-superacion", "index": ALL}, "n_clicks")],
    [State({"type": "jugador-row-superacion", "index": ALL}, "id"),
     State("filtered-data-store-superacion", "data")]
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
    regates_cols = [col for col in player_data.columns if 'REGATES' in col or 'SUPERADOS' in col]
    perdidas_cols = [col for col in player_data.columns if 'PÉRDIDA' in col or 'PERDIDA' in col]
    toques_cols = [col for col in player_data.columns if 'TOQUE' in col]
    
    # Seleccionar las métricas más relevantes para cada categoría
    regates_top = []
    for col in ['REGATES EXITOSOS', 'OPONENTES SUPERADOS POR 90', 'DEFENSORES SUPERADOS POR 90']:
        if col in regates_cols:
            regates_top.append(col)
    
    perdidas_top = []
    for col in ['NÚMERO PÉRDIDAS', 'PÉRDIDAS CRÍTICAS POR 90']:
        if col in player_data.columns:
            perdidas_top.append(col)
    
    toques_top = []
    for col in ['TOQUES EN ÚLTIMO TERCIO', 'TOQUES OFENSIVOS ÁREA RIVAL', 'TOQUES EN POSESIÓN']:
        if col in player_data.columns:
            toques_top.append(col)
    
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
                ], style={"margin-bottom": "8px"}) if ("EQUIPO" in player_info or "CLUB" in player_info) else None,
                
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
            
            # Columna de métricas de regates
            html.Div([
                html.H4("Regates y Superación", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('OPONENTES', '').replace('SUPERADOS', 'Super.').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in regates_top[:5]]  # Limitar a 5 métricas
            ], style={"width": "25%", "padding": "0 15px"}),
            
            # Columna de métricas de pérdidas y toques
            html.Div([
                html.H4("Pérdidas y Toques", style={"color": "#00BFFF", "border-bottom": "1px solid #00BFFF", "padding-bottom": "5px"}),
                *[html.Div([
                    html.Strong(f"{col.replace('NÚMERO', 'Num.').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in perdidas_top[:2]],  # Limitar a 2 métricas de pérdidas
                
                *[html.Div([
                    html.Strong(f"{col.replace('TOQUES', 'Toq.').strip()}: "),
                    html.Span(f"{player_info[col]:.2f}" if isinstance(player_info[col], (int, float)) else player_info[col])
                ], style={"margin-bottom": "5px"}) for col in toques_top[:3]]  # Limitar a 3 métricas de toques
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

# Callback para crear gráfico de eficiencia en posesión vs contraataque
@callback(
    Output("eficiencia-posesion-chart", "children"),
    Input("filtered-data-store-superacion", "data")
)
def create_eficiencia_posesion_chart(filtered_data):
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
    
    # Buscar métricas de posesión y contraataque
    posesion_metric = None
    contraataque_metric = None
    perdidas_metric = None
    
    # Buscar métricas de superación en posesión
    for m in ['OPONENTES SUPERADOS EN POSESIÓN', 'OPONENTES SUPERADOS EN POSESIÓN POR 90', 'TOQUES EN POSESIÓN']:
        if m in df.columns:
            posesion_metric = m
            break
    
    # Buscar métricas de superación en contraataque
    for m in ['OPONENTES SUPERADOS EN CONTRAATAQUE', 'OPONENTES SUPERADOS CONTRAATAQUE POR 90', 'TOQUES TRANSICIÓN OFENSIVA']:
        if m in df.columns:
            contraataque_metric = m
            break
    
    # Buscar métricas de pérdidas
    for m in ['NÚMERO PÉRDIDAS', 'PÉRDIDAS CRÍTICAS POR 90']:
        if m in df.columns:
            perdidas_metric = m
            break
    
    if not posesion_metric or not contraataque_metric:
        return html.Div("No se encontraron suficientes métricas para crear el gráfico de eficiencia en posesión vs contraataque.",
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Filtrar para incluir solo jugadores con datos de posesión y contraataque
    # IMPORTANTE: Crear una copia completa del DataFrame antes de modificarlo para evitar SettingWithCopyWarning
    mask = (df[posesion_metric] > 0) | (df[contraataque_metric] > 0)
    filtered_players = df[mask].copy(deep=True)
    
    if filtered_players.empty:
        return html.Div("No hay suficientes jugadores con datos para analizar la eficiencia.",
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Calcular ratio de eficiencia (posesión/contraataque)
    # Usar .loc para asignar nuevas columnas correctamente y evitar warnings
    filtered_players.loc[:, 'eficiencia_ratio'] = filtered_players[posesion_metric] / filtered_players[contraataque_metric]
    filtered_players.loc[:, 'eficiencia_ratio'] = filtered_players['eficiencia_ratio'].replace([np.inf, -np.inf], np.nan)
    filtered_players = filtered_players.dropna(subset=['eficiencia_ratio'])
    
    # Crear un índice de balance entre posesión y contraataque
    filtered_players.loc[:, 'balance_index'] = (filtered_players[posesion_metric] - filtered_players[contraataque_metric]) / (filtered_players[posesion_metric] + filtered_players[contraataque_metric])
    filtered_players.loc[:, 'balance_index'] = filtered_players['balance_index'].replace([np.inf, -np.inf, np.nan], 0)
    
    # Añadir columna de pérdidas si está disponible
    if perdidas_metric:
        min_val = filtered_players[perdidas_metric].min()
        max_val = filtered_players[perdidas_metric].max()
        if max_val > min_val:
            filtered_players.loc[:, 'perdidas_normalizadas'] = (filtered_players[perdidas_metric] - min_val) / (max_val - min_val)
    
    # Seleccionar los mejores jugadores para mostrar en el gráfico (top 25 por suma de métricas)
    filtered_players.loc[:, 'total_valor'] = filtered_players[posesion_metric] + filtered_players[contraataque_metric]
    top_players = filtered_players.nlargest(25, 'total_valor')
    
    # Crear gráfico de dispersión
    fig = px.scatter(
        top_players,
        x=posesion_metric,
        y=contraataque_metric,
        color="POSICIÓN",
        size=abs(top_players["balance_index"]) * 10 + 5,  # Escalar para mejor visualización
        size_max=20,
        hover_name="NOMBRE",
        text="NOMBRE",
        # Quitar la asignación de colores por posición para usar un color consistente
        template="plotly_dark",
        opacity=0.8
    )
    
    # Actualizar texto y trazas para establecer color azul con borde amarillo
    fig.update_traces(
        textposition='top center',
        textfont=dict(color='white', size=10),
        marker=dict(
            color='#00BFFF',  # Azul claro
            line=dict(width=2, color='#FFFF00')  # Borde amarillo
        )
    )
    
    # Añadir una línea diagonal que representa balance perfecto
    max_val = max(top_players[posesion_metric].max(), top_players[contraataque_metric].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='rgba(255, 255, 0, 0.5)', dash='dash', width=2),
        name='Balance Perfecto',
        showlegend=True
    ))
    
    # Configurar layout - altura aumentada para que los puntos estén más dispersos
    fig.update_layout(
        title={
            'text': "Eficiencia en Posesión vs Contraataque",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'white'}
        },
        font=dict(family="Arial, sans-serif", color="white", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text=posesion_metric.replace('_', ' '), font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            title=dict(text=contraataque_metric.replace('_', ' '), font=dict(size=14, color="white")),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        height=700,  # Aumentado significativamente para más espacio vertical
        margin=dict(l=50, r=50, t=70, b=150)
    )
    
    # Añadir cuadrantes con anotaciones
    # Cuadrante 1: Alto en posesión y contraataque
    fig.add_shape(
        type="rect",
        x0=top_players[posesion_metric].median(),
        y0=top_players[contraataque_metric].median(),
        x1=max_val,
        y1=max_val,
        line=dict(color="rgba(0,191,255,0.2)", width=1),
        fillcolor="rgba(0,191,255,0.1)",
        layer="below"
    )
    
    # Añadir anotaciones con fondo para evitar superposición
    fig.add_annotation(
        x=top_players[posesion_metric].median() + (max_val - top_players[posesion_metric].median())*0.75,
        y=top_players[contraataque_metric].median() + (max_val - top_players[contraataque_metric].median())*0.75,
        text="Alto Balanceado",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.7)",  # Fondo oscuro para evitar superposición
        borderpad=4,  # Padding para el texto
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1
    )
    
    # Cuadrante 2: Alto en posesión, bajo en contraataque
    fig.add_shape(
        type="rect",
        x0=top_players[posesion_metric].median(),
        y0=0,
        x1=max_val,
        y1=top_players[contraataque_metric].median(),
        line=dict(color="rgba(255,255,0,0.2)", width=1),
        fillcolor="rgba(255,255,0,0.1)",
        layer="below"
    )
    
    fig.add_annotation(
        x=top_players[posesion_metric].median() + (max_val - top_players[posesion_metric].median())*0.75,
        y=top_players[contraataque_metric].median()*0.5,
        text="Dominante en Posesión",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.7)",  # Fondo oscuro para evitar superposición
        borderpad=4,  # Padding para el texto
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1
    )
    
    # Cuadrante 3: Bajo en posesión, alto en contraataque
    fig.add_shape(
        type="rect",
        x0=0,
        y0=top_players[contraataque_metric].median(),
        x1=top_players[posesion_metric].median(),
        y1=max_val,
        line=dict(color="rgba(50,205,50,0.2)", width=1),
        fillcolor="rgba(50,205,50,0.1)",
        layer="below"
    )
    
    fig.add_annotation(
        x=top_players[posesion_metric].median()*0.5,
        y=top_players[contraataque_metric].median() + (max_val - top_players[contraataque_metric].median())*0.75,
        text="Especialista en Contraataque",
        showarrow=False,
        font=dict(color="white", size=12),
        bgcolor="rgba(0,0,0,0.7)",  # Fondo oscuro para evitar superposición
        borderpad=4,  # Padding para el texto
        bordercolor="rgba(255,255,255,0.3)",
        borderwidth=1
    )
    
    # Análisis simple para no ocupar mucho espacio
    analysis = html.Div([
        html.P([
            "Los jugadores en el cuadrante superior derecho destacan en ambas fases, siendo muy valiosos en cualquier situación de juego."
        ]),
        html.P([
            "Los entrenadores pueden utilizar esta visualización para identificar perfiles específicos que se adapten mejor al estilo de juego del equipo."
        ])
    ], style={"margin-top": "20px", "padding": "15px"})
    
    return html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False}), analysis])

# Callback para crear gráfico de superación por zonas
@callback(
    Output("superacion-zonas-chart", "children"),
    Input("filtered-data-store-superacion", "data")
)
def create_superacion_zonas_chart(filtered_data):
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
    
    # Definir zonas de superación
    zonas_superacion = [
        ('OPONENTES SUPERADOS DESDE TERCIO MEDIO', 'Tercio Medio'),
        ('OPONENTES SUPERADOS HACIA ÚLTIMO TERCIO', 'Últ. Tercio'),
        ('OPONENTES SUPERADOS DESDE MEDIAPUNTA', 'Mediapunta'),
        ('OPONENTES SUPERADOS DESDE MEDIOCENTRO', 'Mediocentro'),
        ('OPONENTES SUPERADOS DESDE BANDA DERECHA', 'B. Derecha'),
        ('OPONENTES SUPERADOS DESDE BANDA IZQUIERDA', 'B. Izquierda')
    ]
    
    # Filtrar las zonas disponibles en los datos
    available_zones = []
    for col, label in zonas_superacion:
        if col in df.columns:
            available_zones.append((col, label))
    
    if not available_zones:
        return html.Div("No se encontraron métricas de superación por zona en los datos disponibles.", 
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Calcular promedio por posición para cada zona
    position_zones = []
    positions = df['POSICIÓN'].unique()
    
    # Mapeo de posiciones a abreviaturas
    position_abbr = {
        'PORTERO': 'PO',
        'DEFENSA CENTRAL': 'DC',
        'LATERAL DERECHO': 'LD',
        'LATERAL IZQUIERDO': 'LI',
        'MEDIOCENTRO': 'MC',
        'MEDIOCENTRO DEFENSIVO': 'MCD',
        'MEDIOCENTRO OFENSIVO': 'MCO',
        'EXTREMO DERECHO': 'ED',
        'EXTREMO IZQUIERDO': 'EI',
        'MEDIAPUNTA': 'MP',
        'DELANTERO': 'DL',
        'DELANTERO CENTRO': 'DC',
        'SEGUNDO DELANTERO': 'SD',
        'CENTROCAMPISTA': 'CC',
        'INTERIOR DERECHO': 'ID',
        'INTERIOR IZQUIERDO': 'II',
        'CARRILERO DERECHO': 'CLD',
        'CARRILERO IZQUIERDO': 'CLI'
    }
    
    for position in positions:
        position_data = df[df['POSICIÓN'] == position]
        if position_data.empty:
            continue
        
        # Obtener abreviatura para la posición
        abbr = position_abbr.get(position, position[:2])  # Default: primeras 2 letras
        
        for col, label in available_zones:
            avg_value = position_data[col].mean()
            if not pd.isna(avg_value):
                position_zones.append({
                    'Posición': abbr,  # Usar abreviatura
                    'PosiciónCompleta': position,  # Guardar nombre completo para hover
                    'Zona': label,
                    'Promedio': avg_value
                })
    
    if not position_zones:
        return html.Div("No hay suficientes datos para mostrar la superación por zonas.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Convertir a DataFrame
    position_zones_df = pd.DataFrame(position_zones)
    
    # Crear gráfico de barras agrupadas
    fig = px.bar(
        position_zones_df,
        x='Posición',
        y='Promedio',
        color='Zona',
        barmode='group',
        title="Superación por Zonas según Posición",
        template="plotly_dark",
        color_discrete_sequence=['#00BFFF', '#FFFF00', '#4CAF50', '#FF8C00', '#FF69B4', '#9370DB'],  # Colores distintos
        hover_data=['PosiciónCompleta']  # Incluir nombre completo en hover
    )
    
    # Mejorar estética del gráfico
    fig.update_traces(
        marker_line_color='rgb(255, 255, 255)',
        marker_line_width=0.5, 
        opacity=0.85
    )
    
    # Actualizar el formato de hover para mostrar el nombre completo
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                     'Zona: %{color}<br>' +
                     'Promedio: %{y:.2f}<extra></extra>'
    )
    
    # Configurar layout con leyenda a la derecha
    fig.update_layout(
        font=dict(color="white", family="Arial, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={
            'text': "Superación por Zonas según Posición",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis=dict(
            title=dict(text="Posición", font=dict(color="white", size=14)),
            gridcolor="rgba(255,255,255,0.1)",
            categoryorder='total descending',
            tickangle=0,  # Texto horizontal
            tickfont=dict(size=14)  # Texto más grande para que se vea mejor
        ),
        yaxis=dict(
            title=dict(text="Promedio de Oponentes Superados", font=dict(color="white", size=14)),
            gridcolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        # Leyenda a la derecha
        legend=dict(
            title=dict(text="Zona", font=dict(color="white")),
            orientation="v",  # Vertical
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05,  # Colocar a la derecha del gráfico
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        height=450,
        margin=dict(l=50, r=120, t=70, b=50)  # Margen derecho mayor para la leyenda
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Callback para crear radar comparativo de superación
@callback(
    Output("radar-comparativo-superacion", "children"),
    [Input("jugador-seleccionado-store-superacion", "data"),
     Input("filtered-data-store-superacion", "data")]
)
def create_radar_comparativo_superacion(jugador_data, filtered_data):
    if not filtered_data:
        return html.Div("No hay datos disponibles. Aplica filtros para visualizar resultados.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Si hay un jugador seleccionado, usarlo; si no, tomar el mejor jugador
    selected_player = None
    player_df = None
    
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
    
    # Determinar si hay un jugador seleccionado
    if jugador_data and 'player_name' in jugador_data:
        selected_player = jugador_data['player_name']
        player_df = df[df["NOMBRE"] == selected_player]
        if player_df.empty:
            selected_player = None
    
    # Si no hay jugador seleccionado, tomar el mejor en regates exitosos
    if not selected_player:
        regates_metric = None
        for m in ['REGATES EXITOSOS', 'OPONENTES SUPERADOS POR 90', 'TOTAL OPONENTES SUPERADOS']:
            if m in df.columns:
                regates_metric = m
                break
        
        if not regates_metric:
            return html.Div("No se encontraron métricas de regates para el análisis comparativo.", 
                           style={"text-align": "center", "padding": "20px", "color": "orange"})
        
        # Tomar el mejor jugador en la métrica seleccionada
        best_player_idx = df[regates_metric].idxmax()
        selected_player = df.loc[best_player_idx, "NOMBRE"]
        player_df = df[df["NOMBRE"] == selected_player]
    
    if player_df.empty:
        return html.Div("No se encontraron datos para el jugador seleccionado.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Obtener posición del jugador para comparar con similares
    player_position = player_df['POSICIÓN'].iloc[0]
    
    # Filtrar jugadores de la misma posición
    position_players = df[df['POSICIÓN'] == player_position].copy(deep=True)
    
    # Eliminar al jugador seleccionado
    others_df = position_players[position_players['NOMBRE'] != selected_player].copy(deep=True)
    
    if others_df.empty or len(others_df) < 2:
        return html.Div(f"No hay suficientes jugadores en la posición {player_position} para comparar.", 
                       style={"text-align": "center", "padding": "20px", "font-style": "italic"})
    
    # Seleccionar métricas para el radar
    radar_metrics = []
    for metric in [
        'REGATES EXITOSOS', 'OPONENTES SUPERADOS POR 90', 'DEFENSORES SUPERADOS POR 90',
        'OPONENTES SUPERADOS EN POSESIÓN', 'OPONENTES SUPERADOS EN CONTRAATAQUE',
        'NÚMERO PÉRDIDAS', 'TOQUES EN ÚLTIMO TERCIO'
    ]:
        if metric in df.columns:
            radar_metrics.append(metric)
    
    # Si no hay suficientes métricas, buscar cualquier métrica relacionada con superación
    if len(radar_metrics) < 4:
        for col in df.columns:
            if ('SUPERADOS' in col or 'REGAT' in col) and col not in radar_metrics:
                radar_metrics.append(col)
            if len(radar_metrics) >= 5:
                break
    
    if len(radar_metrics) < 3:
        return html.Div("No hay suficientes métricas de superación para crear el radar comparativo.", 
                       style={"text-align": "center", "padding": "20px", "color": "orange"})
    
    # Limitar a máximo 6 métricas para mejor visualización
    radar_metrics = radar_metrics[:6]
    
    # Identificar jugadores similares por distancia euclidiana en las métricas seleccionadas
    # Normalizar métricas
    for metric in radar_metrics:
        metric_min = position_players[metric].min()
        metric_max = position_players[metric].max()
        if metric_max > metric_min:
            others_df.loc[:, f"{metric}_norm"] = (others_df[metric] - metric_min) / (metric_max - metric_min)
        else:
            others_df.loc[:, f"{metric}_norm"] = 0.5
    
    # Normalizar métricas para el jugador seleccionado
    player_normalized = {}
    for metric in radar_metrics:
        metric_min = position_players[metric].min()
        metric_max = position_players[metric].max()
        if metric_max > metric_min:
            player_normalized[f"{metric}_norm"] = (player_df[metric].iloc[0] - metric_min) / (metric_max - metric_min)
        else:
            player_normalized[f"{metric}_norm"] = 0.5
    
    # Calcular distancia euclidiana
    distances = []
    for idx, row in others_df.iterrows():
        distance = 0
        for metric in radar_metrics:
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
    for metric in radar_metrics:
        if metric in player_df.columns:
            player_values.append(player_df[metric].iloc[0])
        else:
            player_values.append(0)
    
    radar_data.append({
        "player": selected_player,
        "values": player_values,
        "metrics": radar_metrics
    })
    
    # Añadir jugadores similares
    for similar_name in similar_names:
        similar_df = df[df["NOMBRE"] == similar_name]
        if not similar_df.empty:
            similar_values = []
            for metric in radar_metrics:
                if metric in similar_df.columns:
                    similar_values.append(similar_df[metric].iloc[0])
                else:
                    similar_values.append(0)
            
            radar_data.append({
                "player": similar_name,
                "values": similar_values,
                "metrics": radar_metrics
            })
    
    # Formatear métricas para mejor visualización
    formatted_metrics = []
    for metric in radar_metrics:
        # Simplificar nombres
        formatted = metric.replace("OPONENTES SUPERADOS", "OP. SUPER.")
        formatted = formatted.replace("DEFENSORES", "DEF.")
        formatted = formatted.replace("NÚMERO PÉRDIDAS", "PÉRDIDAS")
        formatted = formatted.replace("REGATES EXITOSOS", "REGATES")
        formatted = formatted.replace("TOQUES EN ÚLTIMO TERCIO", "TOQUES ÚLT.")
        formatted = formatted.replace("EN POSESIÓN", "POS.")
        formatted = formatted.replace("EN CONTRAATAQUE", "CONT.")
        formatted = formatted.replace("POR 90", "/90")
        
        formatted_metrics.append(formatted)
    
    # Crear gráfico de radar
    fig = go.Figure()
    
    # Colores para cada jugador - MEJORA: Primer jugador siempre azul claro con borde amarillo
    colors = ['#00BFFF', '#FFFF00', '#4CAF50']
    
    # Añadir un trace para cada jugador
    for i, player_data in enumerate(radar_data):
        # Cerrar el polígono repitiendo el primer valor
        values = player_data["values"] + [player_data["values"][0]]
        metrics = formatted_metrics + [formatted_metrics[0]]
        
        # Definir el color y el relleno para cada jugador
        if i == 0:  # Primer jugador - azul con borde amarillo
            line_color = '#00BFFF'
            line_width = 3
            fillcolor = 'rgba(0, 191, 255, 0.3)'
            border_color = '#FFFF00'
        elif i == 1:  # Segundo jugador - amarillo
            line_color = colors[1]
            line_width = 2
            fillcolor = 'rgba(255, 255, 0, 0.3)'
            border_color = line_color
        else:  # Tercer jugador - verde
            line_color = colors[2]
            line_width = 2
            fillcolor = 'rgba(76, 175, 80, 0.3)'
            border_color = line_color
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=player_data["player"],
            line=dict(color=line_color, width=line_width),
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
            y=-0.25,  # Posición más baja para evitar superposición
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        title={
            'text': f"Perfil Comparativo de {selected_player}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': 'white'}
        },
        height=450,
        margin=dict(l=40, r=40, t=80, b=120)  # Márgenes ajustados para evitar superposición
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

# Callback para abrir el modal
@callback(
    Output("superacion-pdf-modal", "is_open"),
    [Input("superacion-pdf-button", "n_clicks"),
     Input("superacion-pdf-modal-close", "n_clicks"),
     Input("superacion-pdf-modal-generate", "n_clicks")],
    [State("superacion-pdf-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n_open, n_close, n_generate, is_open):
    if n_open or n_close or n_generate:
        return not is_open
    return is_open

# Callback para generar y descargar el PDF
@callback(
    Output("superacion-pdf-download", "data"),
    Input("superacion-pdf-modal-generate", "n_clicks"),
    [State("superacion-pdf-title", "value"),
     State("superacion-pdf-description", "value"),
     State("filtered-data-store-superacion", "data")],
    prevent_initial_call=True
)
def generate_pdf_superacion(n_clicks, title, description, filtered_data):
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
    elements.append(Paragraph(title or "Informe de Análisis de Superación", title_style))
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
        for metric in ["REGATES EXITOSOS", "OPONENTES SUPERADOS POR 90", "TOTAL OPONENTES SUPERADOS"]:
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
                short_name = short_name.replace("OPONENTES SUPERADOS POR 90", "OP.SUPER/90")
                short_name = short_name.replace("REGATES EXITOSOS", "REG.EXIT.")
                short_name = short_name.replace("OPONENTES SUPERADOS AL RECIBIR", "OP.SUP.RECEP")
                short_name = short_name.replace("TOQUES EN ÚLTIMO TERCIO", "TOQ.ULT.TERC")
                short_name = short_name.replace("NÚMERO PÉRDIDAS", "NUM.PERD")
                short_name = short_name.replace("PÉRDIDAS CRÍTICAS POR 90", "PERD.CRIT/90")
                
                # Limitar longitud a 12 caracteres para evitar desbordes
                if len(short_name) > 12:
                    short_name = short_name[:12]
                
                headers.append(short_name)
                metrics_to_show.append(metric)
        
        # Preparar datos para la tabla
        table_data = [headers]
        
        # Añadir filas de datos - Acortar nombres si son muy largos
        for _, row in df.head(15).iterrows():
            nombre = row["NOMBRE"]
            if len(nombre) > 18:  # Limitar largo de nombres
                nombre_parts = nombre.split()
                if len(nombre_parts) > 1:
                    nombre = f"{nombre_parts[0]} {nombre_parts[-1]}"
                else:
                    nombre = nombre[:18]
                    
            posicion = row["POSICIÓN"] 
            liga = row.get("LIGA", "")
            
            row_data = [nombre, posicion, liga]
            
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
        metrics_width = metrics_space / len(metrics_to_show) if metrics_to_show else 1
        
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
        "filename": f"informe_superacion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        "type": "application/pdf",
        "base64": True
    }

