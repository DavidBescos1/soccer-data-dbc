from dash import dcc, html, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from components.data_manager import DataManager
from components.sidebar import create_sidebar

# Inicializar el gestor de datos
data_manager = DataManager()

# Definir categorías de métricas para agrupar los filtros
categorias_metricas = {
    "Métricas Defensivas": [
        "TOQUES DEFENSIVOS ÁREA PROPIA", "DUELOS TERRESTRES GANADOS", "DUELOS TERRESTRES PERDIDOS", 
        "DUELOS AÉREOS GANADOS", "DUELOS AÉREOS PERDIDOS", "INTERCEPCIONES", "BLOQUEOS", 
        "RECUPERACIONES BALONES SUELTOS", "DESPEJES CABEZA", "TOQUES DEFENSIVOS PRIMER TERCIO", 
        "RECUPERACIONES PRIMER TERCIO", "NÚMERO PRESIONES", "TOQUES DEFENSIVOS SIN POSESIÓN", 
        "TOQUES TRANSICIÓN DEFENSIVA", "RECUPERACIONES SIN POSESIÓN", "RECUPERACIONES TRANSICIÓN DEFENSIVA", 
        "RECUPERACIONES EN ÚLTIMO TERCIO", "VALOR AÑADIDO BLOQUEOS", "RECUPERACIONES COMO CENTRAL", 
        "RECUPERACIONES COMO LATERAL DERECHO", "RECUPERACIONES COMO LATERAL IZQUIERDO", 
        "VALOR INTERCEPCIONES EQUIPO", "PASES DESDE PRIMER TERCIO", "TOQUES DEFENSIVOS EN DUELOS", 
        "DUELOS DEFENSIVOS GANADOS", "TOQUES DEFENSIVOS CARRIL CENTRAL"
    ],
    "Métricas Ofensivas": [
        "OPONENTES SUPERADOS POR 90", "OPONENTES SUPERADOS/90", "DEFENSORES SUPERADOS POR 90",
        "OPONENTES SUPERADOS POR REGATE", "REGATES EXITOSOS", "OPONENTES SUPERADOS AL RECIBIR",
        "NÚMERO OPONENTES SUPERADOS AL RECIBIR", "TOQUES EN ÚLTIMO TERCIO", "TOQUES OFENSIVOS ÁREA RIVAL",
        "TOQUES EN POSESIÓN", "TOQUES TRANSICIÓN OFENSIVA", "DISPONIBILIDAD EN ÁREA",
        "TOQUES PROTEGIENDO BALÓN", "OPONENTES SUPERADOS EN POSESIÓN POR 90", 
        "OPONENTES SUPERADOS CONTRAATAQUE POR 90", "DISTANCIA REGATE HACIA PORTERÍA",
        "DISTANCIA CONDUCCIÓN HACIA PORTERÍA", "TOTAL OPONENTES SUPERADOS", 
        "OPONENTES SUPERADOS PASE BAJO", "OPONENTES SUPERADOS PASE DIAGONAL",
        "OPONENTES SUPERADOS PASE ELEVADO", "OPONENTES SUPERADOS PASE AÉREO CORTO",
        "OPONENTES SUPERADOS DESDE TERCIO MEDIO", "OPONENTES SUPERADOS HACIA ÚLTIMO TERCIO",
        "CENTROS ALTOS EXITOSOS", "CENTROS RASOS EXITOSOS", "CÓRNERS EXITOSOS", "TIROS LIBRES EXITOSOS"
    ],
    "Métricas de Creación": [
        "PASES EXITOSOS POR 90", "PASES FALLIDOS", "PASES RASOS EXITOSOS", "PASES DIAGONALES EXITOSOS",
        "PASES ELEVADOS EXITOSOS", "PASES AÉREOS CORTOS EXITOSOS", "CENTROS ALTOS EXITOSOS", 
        "CENTROS RASOS EXITOSOS", "CÓRNERS EXITOSOS", "TIROS LIBRES EXITOSOS", 
        "PASES DESDE PRIMER TERCIO", "PASES DESDE TERCIO MEDIO", "PASES DESDE ÚLTIMO TERCIO",
        "PASES EXITOSOS HACIA ÚLTIMO TERCIO", "PASES EXITOSOS HACIA ÁREA RIVAL", 
        "PASES EXITOSOS A MEDIAPUNTA", "PASES EXITOSOS A MEDIOCENTRO", "PASES EXITOSOS A BANDA DERECHA",
        "PASES EXITOSOS A BANDA IZQUIERDA", "PASES EXITOSOS A PIVOTE", "PASES DESDE CARRIL CENTRAL",
        "PASES HACIA CARRIL CENTRAL", "PASES DESDE BANDA DERECHA", "PASES HACIA BANDA DERECHA",
        "PASES DESDE BANDA IZQUIERDA", "PASES HACIA BANDA IZQUIERDA", "PASES EN FASE POSESIÓN",
        "PASES EN TRANSICIÓN OFENSIVA", "PASES EN JUGADAS BALÓN PARADO", "PASES EXITOSOS HACIA ÁREA",
        "PASES EXITOSOS HACIA PRIMER TERCIO", "PASES EXITOSOS HACIA TERCIO MEDIO", 
        "PASES DESDE LATERAL DERECHO POR 90", "PASES DESDE DEFENSA CENTRAL POR 90", 
        "PASES DESDE LATERAL IZQUIERDO POR 90", "ASISTENCIAS POR 90", "ASISTENCIAS PASE RASO",
        "ASISTENCIAS PASE DIAGONAL", "ASISTENCIAS PASE ELEVADO", "ASISTENCIAS PASE AÉREO CORTO",
        "ASISTENCIAS CENTRO RASO", "ASISTENCIAS CENTRO ALTO", "ASISTENCIAS DESDE MEDIAPUNTA",
        "ASISTENCIAS DESDE MEDIOCENTRO", "ASISTENCIAS DESDE BANDA DERECHA", 
        "ASISTENCIAS DESDE BANDA IZQUIERDA", "ASISTENCIAS DESDE ÚLTIMO TERCIO", 
        "ASISTENCIAS DESDE CARRIL CENTRAL", "ASISTENCIAS DESDE BANDA DERECHA/90", 
        "ASISTENCIAS DESDE BANDA IZQUIERDA/90", "XG CREADA CON PASES", "ASISTENCIAS POR CÓRNER",
        "ASISTENCIAS CONTRAATAQUE", "ASISTENCIAS EN POSESIÓN", "ASISTENCIAS JUGADAS BALÓN PARADO"
    ],
    "Métricas de Tiro y Gol": [
        "TIROS A PORTERÍA", "TIROS FUERA", "TIROS BLOQUEADOS", "TIROS EXITOSOS", 
        "TIROS LARGA DISTANCIA", "TIROS MEDIA DISTANCIA", "TIROS CORTA DISTANCIA", 
        "REMATES CABEZA", "TIROS UNO CONTRA UNO", "PENALTIS TIRADOS", "TIROS LIBRES DIRECTOS",
        "TIROS DENTRO ÁREA", "TIROS DESDE ÚLTIMO TERCIO", "XG POR TIROS", "XG TIROS LEJANOS",
        "XG TIROS MEDIA DISTANCIA", "XG TIROS CERCANOS", "XG REMATES CABEZA", "XG PENALTIS",
        "GOLES POR 90", "GOLES TIRO LARGA DISTANCIA", "GOLES TIRO MEDIA DISTANCIA", 
        "GOLES TIRO CORTA DISTANCIA", "GOLES CABEZA", "GOLES UNO CONTRA UNO", "GOLES PENALTI",
        "GOLES TIRO LIBRE", "GOLES PUERTA VACÍA", "GOLES DENTRO ÁREA", "GOLES DESDE ÚLTIMO TERCIO",
        "GOLES DESDE CARRIL CENTRAL", "GOLES EN POSESIÓN POR 90", "GOLES CONTRAATAQUE POR 90",
        "GOLES JUGADA PARADA POR 90", "XG POST TIRO", "XG DENTRO ÁREA", "XG DESDE ÚLTIMO TERCIO",
        "XG DESDE CARRIL CENTRAL", "XG EN POSESIÓN", "XG EN CONTRAATAQUE", "XG EN JUGADAS BALÓN PARADO"
    ],
    "Métricas de Portero": [
        "BALONES ATRAPADOS", "PARADAS REALIZADAS", "TIROS SALVADOS", 
        "OPONENTES SUPERADOS POR PARADA", "TOTAL TIROS ENFRENTADOS", "VALOR AÑADIDO PARADAS",
        "SAQUES PUERTA EXITOSOS", "OPONENTES SUPERADOS POR SAQUE", "OPORTUNIDADES NEUTRALIZADAS"
    ],
    "Métricas de Pérdida y Recuperación": [
        "CENTROCAMPISTAS SUPERADOS POR 90", "DEFENSAS SUPERADOS PASE RASO",
        "DEFENSAS SUPERADOS PASE DIAGONAL", "DEFENSAS SUPERADOS PASE ELEVADO", 
        "DEFENSAS SUPERADOS REGATE", "OPONENTES SUPERADOS EN POSESIÓN", 
        "OPONENTES SUPERADOS EN CONTRAATAQUE", "NÚMERO JUGADAS NEUTRAS", 
        "NÚMERO JUGADAS REVERSIBLES", "NÚMERO PÉRDIDAS", "PÉRDIDAS CRÍTICAS POR 90",
        "OPONENTES GANADOS RIVAL TRAS PÉRDIDA", "COMPAÑEROS DESPOSICIONADOS TRAS PÉRDIDA",
        "COMPAÑEROS BENEFICIADOS TRAS RECUPERACIÓN", "OPONENTES ANULADOS TRAS RECUPERACIÓN",
        "OPONENTES SUPERADOS JUGADAS BALÓN PARADO", "OPONENTES SUPERADOS DESDE MEDIAPUNTA",
        "OPONENTES SUPERADOS DESDE MEDIOCENTRO", "OPONENTES SUPERADOS DESDE BANDA DERECHA",
        "OPONENTES SUPERADOS DESDE BANDA IZQUIERDA", "OPONENTES SUPERADOS HACIA MEDIAPUNTA"
    ]
}

# Layout de la página
layout = html.Div([
    # Sidebar
    create_sidebar(),
    
    # Contenido principal
    html.Div(
        className="content-container fade-in",
        children=[
            # Título de la página
            html.H1("Búsqueda de Jugadores Similares", 
                   style={"color": "#00BFFF", "margin-bottom": "20px"}),
            
            # CAMBIO 1: Información del jugador base ahora va antes de los filtros
            html.Div(
                id="jugador-base-info",
                className="dashboard-card mb-4",
                style={"min-height": "120px", "display": "none"}  # Inicialmente oculto
            ),
            
            # Contenedor de filtros
            html.Div(
                className="filter-container",
                children=[
                    dbc.Row([
                        # Columna de búsqueda de jugador
                        dbc.Col([
                            html.Div(className="filter-title", children="Buscar Jugador Base"),
                            dcc.Dropdown(
                                id="jugador-base-dropdown",
                                options=[],  # Se llenará dinámicamente
                                placeholder="Selecciona un jugador",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Columna para seleccionar posición del jugador
                        dbc.Col([
                            html.Div(className="filter-title", children="Posición del Jugador"),
                            dcc.Dropdown(
                                id="posicion-jugador-dropdown",
                                options=[],  # Se llenará con las posiciones del jugador seleccionado
                                placeholder="Selecciona posición",
                                className="filter-dropdown",
                                style={"color": "black"}
                            ),
                        ], width=4),
                        
                        # Columna de filtros de edad
                        dbc.Col([
                            html.Div(className="filter-title", children="Rango de Edad"),
                            dcc.RangeSlider(
                                id="edad-range-slider",
                                min=16, max=45, step=1,
                                marks={i: str(i) for i in range(16, 46, 5)},
                                value=[18, 35],
                                className="filter-slider"
                            ),
                        ], width=4),
                    ]),
                    
                    # NUEVO: Fila para filtro de minutos jugados
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="filter-title", children="Minutos Jugados (mínimo)"),
                            dcc.Slider(
                                id="minutos-jugados-slider",
                                min=0,
                                max=3000,
                                step=100,
                                value=0,
                                marks={i: str(i) for i in range(0, 3001, 500)},
                                className="filter-slider"
                            ),
                        ], width=12),
                    ], className="mb-3"),
                    
                    # Fila para filtro de posición
                    dbc.Row([
                        dbc.Col([
                            html.Div(className="filter-title", children="Filtrar por Posición"),
                            dcc.Checklist(
                                id="posicion-match-checklist",
                                options=[
                                    {"label": "Misma posición", "value": "same_position"},
                                    {"label": "Posiciones similares", "value": "similar_position"}
                                ],
                                value=["same_position"],
                                inputStyle={"margin-right": "10px"},
                                labelStyle={"color": "white", "margin-right": "20px"},
                                style={"display": "flex"}
                            ),
                        ], width=12),
                    ]),
                    
                    # MODIFICACIÓN: Sección de métricas por grupo
                    html.Div([
                        html.H4("Selección de Métricas por Categoría", 
                               style={"color": "#00BFFF", "margin-top": "20px", "margin-bottom": "15px"}),
                        html.P("Selecciona métricas de cada categoría para comparar jugadores:", 
                              style={"color": "white", "margin-bottom": "20px"}),
                        
                        # Crear un desplegable para cada categoría
                        html.Div(
                            id="metricas-categorias-container",
                            children=[
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(categoria, className="filter-title"),
                                        dcc.Dropdown(
                                            id=f"metricas-dropdown-{i}",
                                            options=[{"label": metrica, "value": metrica} for metrica in metricas],
                                            value=[],
                                            multi=True,
                                            placeholder=f"Seleccionar métricas {categoria}",
                                            className="filter-dropdown mb-3",
                                            style={"color": "black"}
                                        )
                                    ], width=12)
                                ]) for i, (categoria, metricas) in enumerate(categorias_metricas.items())
                            ]
                        ),
                    ]),
                    
                    # Ponderaciones de métricas
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="ponderaciones-container", className="mt-3")
                        ], width=12)
                    ]),
                    
                    # Botón de búsqueda
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "BUSCAR JUGADORES SIMILARES",
                                id="buscar-similares-button",
                                className="login-button mt-4"
                            ),
                        ], width={"size": 4, "offset": 4}),
                    ]),
                ]
            ),
            
            # CAMBIO 2: Gráfico de radar a ancho completo
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id="jugadores-similares-chart",
                        className="graph-container mt-4",
                        style={"min-height": "400px", "width": "100%"}
                    )
                ], width=12),
            ]),
            
            # Tabla de jugadores similares
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id="jugadores-similares-table",
                        className="graph-container mt-4",
                        style={"min-height": "300px"}
                    )
                ], width=12),
            ]),
            
            # Store para datos calculados
            dcc.Store(id="jugadores-similares-data"),
            dcc.Store(id="metricas-seleccionadas-store"),
        ]
    )
])

# Callback para cargar las opciones de jugadores
@callback(
    Output("jugador-base-dropdown", "options"),
    Input("jugador-base-dropdown", "search_value")
)
def update_jugadores_opciones(search_value):
    df = data_manager.get_data()
    
    if df.empty:
        return []
    
    jugadores = df["NOMBRE"].unique()
    jugadores_ordenados = sorted(jugadores)
    
    options = [{"label": jugador, "value": jugador} for jugador in jugadores_ordenados]
    
    # Filtrar por búsqueda si hay un valor
    if search_value:
        options = [opt for opt in options if search_value.upper() in opt["label"].upper()]
    
    return options

# Callback para actualizar las opciones de posición según el jugador seleccionado
@callback(
    Output("posicion-jugador-dropdown", "options"),
    Output("posicion-jugador-dropdown", "value"),
    Input("jugador-base-dropdown", "value")
)
def update_posiciones_jugador(jugador_seleccionado):
    if not jugador_seleccionado:
        return [], None
    
    # Obtener datos del jugador
    df = data_manager.get_data()
    jugador_data = df[df["NOMBRE"] == jugador_seleccionado]
    
    if jugador_data.empty:
        return [], None
    
    # Obtener posiciones únicas para este jugador
    posiciones = jugador_data["POSICIÓN"].unique()
    
    # Crear opciones para el dropdown
    options = [{"label": pos, "value": pos} for pos in posiciones]
    
    # Establecer la primera posición como valor predeterminado
    default_value = posiciones[0] if len(posiciones) > 0 else None
    
    return options, default_value

# Callback para recopilar todas las métricas seleccionadas
@callback(
    Output("metricas-seleccionadas-store", "data"),
    [Input(f"metricas-dropdown-{i}", "value") for i in range(len(categorias_metricas))]
)
def collect_selected_metrics(*all_metrics_values):
    # Combinar todas las métricas seleccionadas en una lista única
    all_selected_metrics = []
    for metrics in all_metrics_values:
        if metrics:
            all_selected_metrics.extend(metrics)
    
    return all_selected_metrics

# Callback para generar controles de ponderación
@callback(
    Output("ponderaciones-container", "children"),
    Input("metricas-seleccionadas-store", "data")
)
def create_ponderaciones_controls(metricas_seleccionadas):
    if not metricas_seleccionadas:
        return html.Div("Selecciona métricas para ajustar su importancia")
    
    # Crear sliders para cada métrica seleccionada
    sliders = []
    for metrica in metricas_seleccionadas:
        slider_row = dbc.Row([
            dbc.Col([
                html.Div(metrica, style={"color": "white", "margin-bottom": "5px"})
            ], width=4),
            dbc.Col([
                dcc.Slider(
                    id=f"ponderacion-slider-{metrica}",
                    min=1,
                    max=10,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(1, 11)},
                )
            ], width=8),
        ], className="mb-2")
        
        sliders.append(slider_row)
    
    return html.Div([
        html.Div("Ajusta la importancia de cada métrica (1-10):", 
                 className="filter-title mb-3"),
        html.Div(sliders)
    ])

# CAMBIO 3: Modificar el callback para agregar una salida para el estilo
@callback(
    Output("jugador-base-info", "children"),
    Output("jugador-base-info", "style"),
    [Input("jugador-base-dropdown", "value"),
     Input("posicion-jugador-dropdown", "value")]
)
def update_jugador_base_info(jugador_seleccionado, posicion_seleccionada):
    if not jugador_seleccionado:
        return html.Div("Selecciona un jugador base para ver su información."), {"display": "none"}
    
    # Obtener datos del jugador
    df = data_manager.get_data()
    jugador_data = df[df["NOMBRE"] == jugador_seleccionado]
    
    if jugador_data.empty:
        return html.Div("No se encontraron datos para este jugador."), {"display": "none"}
    
    # Filtrar por posición si está seleccionada
    if posicion_seleccionada:
        jugador_data = jugador_data[jugador_data["POSICIÓN"] == posicion_seleccionada]
        if jugador_data.empty:
            return html.Div("No se encontraron datos para esta posición."), {"display": "none"}
    
    # Obtener la primera fila (en caso de que haya múltiples)
    jugador_info = jugador_data.iloc[0]
    
    # Obtener la URL de la imagen del jugador
    imagen_url = data_manager.get_player_image_url(jugador_seleccionado)
    
    # Crear un diseño con dos áreas: información a la izquierda, imagen a la derecha
    return html.Div([
        # Encabezado
        html.H3(jugador_seleccionado, style={"color": "#00BFFF", "margin-bottom": "15px"}),
        
        # Contenedor principal con 2 columnas
        html.Div([
            # Columna izquierda: todos los datos
            html.Div([
                html.Div([
                    html.Strong("Posición: "),
                    html.Span(jugador_info.get("POSICIÓN", "No disponible"))
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Fecha de nacimiento: "),
                    html.Span(str(jugador_info.get("FECHA NACIMIENTO", "No disponible")))
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Pierna: "),
                    html.Span(jugador_info.get("PIERNA", "No disponible"))
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Lugar de nacimiento: "),
                    html.Span(jugador_info.get("LUGAR NACIMIENTO", "No disponible"))
                ], style={"margin-bottom": "8px"}),
                
                html.Div([
                    html.Strong("Minutos jugados: "),
                    html.Span(str(jugador_info.get("MINUTOS JUGADOS", "No disponible")))
                ], style={"margin-bottom": "8px"}),
            ], style={"width": "50%", "float": "left", "padding-top": "25px"}),
            
            # Columna derecha: imagen centrada
            html.Div([
                html.Img(
                    src=imagen_url if imagen_url else "/assets/players/placeholder.png",
                    style={
                        "width": "180px",
                        "height": "180px",
                        "border-radius": "50%",
                        "object-fit": "cover",
                        "border": "3px solid #00BFFF",
                        "display": "block",
                        "margin": "0 auto"
                    }
                )
            ], style={
                "width": "50%", 
                "float": "left", 
                "display": "flex",
                "align-items": "center",
                "justify-content": "center"
            }),
            
            # Clear para el float
            html.Div(style={"clear": "both"})
        ], style={"display": "flex", "align-items": "center", "min-height": "180px"})
    ]), {"display": "block", "padding": "15px", "min-height": "230px"}

# Función para encontrar jugadores similares
def find_similar_players(jugador_base, metricas, ponderaciones, filtros, posicion_jugador=None):
    """
    Encuentra jugadores similares al jugador base según las métricas y ponderaciones.
    
    Args:
        jugador_base (str): Nombre del jugador base
        metricas (list): Lista de métricas a considerar
        ponderaciones (dict): Diccionario de ponderaciones para cada métrica
        filtros (dict): Filtros adicionales (edad, posición)
        posicion_jugador (str, optional): Posición específica del jugador base
        
    Returns:
        DataFrame: Jugadores similares ordenados por similitud
    """
    df = data_manager.get_data()
    
    if df.empty or jugador_base not in df["NOMBRE"].unique():
        return pd.DataFrame()
    
    # Datos del jugador base, filtrados por posición si es necesario
    jugador_base_data = df[df["NOMBRE"] == jugador_base]
    if posicion_jugador:
        jugador_base_data = jugador_base_data[jugador_base_data["POSICIÓN"] == posicion_jugador]
        if jugador_base_data.empty:
            return pd.DataFrame()
    
    # Aplicar filtros
    filtered_df = df.copy()
    
    # Filtro de posición
    if "same_position" in filtros.get("posicion", []):
        posicion_base = jugador_base_data["POSICIÓN"].iloc[0]
        filtered_df = filtered_df[filtered_df["POSICIÓN"] == posicion_base]
    elif "similar_position" in filtros.get("posicion", []):
        # Aquí se implementaría lógica para posiciones similares
        # Por ahora, usamos una regla simple (primera letra de la posición)
        posicion_base = jugador_base_data["POSICIÓN"].iloc[0]
        if posicion_base:
            filtered_df = filtered_df[filtered_df["POSICIÓN"].str[0] == posicion_base[0]]
    
    # Filtro de edad (si existe la columna FECHA NACIMIENTO)
    if "FECHA NACIMIENTO" in df.columns and "edad" in filtros:
        min_edad, max_edad = filtros["edad"]
        # Manejar fechas de forma segura
        try:
            # Usar errors='coerce' para manejar valores problemáticos
            fecha_nacimiento = pd.to_datetime(filtered_df["FECHA NACIMIENTO"], errors='coerce')
            filtered_df["EDAD"] = pd.to_datetime('now').year - fecha_nacimiento.dt.year
            filtered_df = filtered_df[(filtered_df["EDAD"] >= min_edad) & (filtered_df["EDAD"] <= max_edad)]
        except Exception as e:
            print(f"Error al calcular edades: {e}")
    
    # NUEVO: Filtro de minutos jugados
    if "min_minutos" in filtros and filtros["min_minutos"] > 0:
        min_minutos = filtros["min_minutos"]
        if "MINUTOS JUGADOS" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["MINUTOS JUGADOS"] >= min_minutos]
    
    # Excluir al jugador base de la búsqueda
    filtered_df = filtered_df[filtered_df["NOMBRE"] != jugador_base]
    
    # Verificar que tenemos todas las métricas necesarias
    available_metrics = [m for m in metricas if m in filtered_df.columns]
    
    if not available_metrics:
        return pd.DataFrame()
    
    # Extraer las métricas relevantes para el cálculo de similitud
    metrics_df = filtered_df[["NOMBRE"] + available_metrics].copy()
    jugador_base_metrics = jugador_base_data[available_metrics].iloc[0]
    
    # Eliminar filas con valores NaN en las métricas
    metrics_df = metrics_df.dropna(subset=available_metrics)
    
    if metrics_df.empty:
        return pd.DataFrame()
    
    # Normalizar las métricas
    scaler = MinMaxScaler()
    metrics_only = metrics_df[available_metrics]
    normalized_metrics = pd.DataFrame(
        scaler.fit_transform(metrics_only),
        columns=available_metrics,
        index=metrics_df.index
    )
    
    # Normalizar las métricas del jugador base
    jugador_base_normalized = pd.DataFrame(
        scaler.transform([jugador_base_metrics.values]),
        columns=available_metrics
    )
    
    # Aplicar ponderaciones
    weighted_metrics = normalized_metrics.copy()
    weighted_base = jugador_base_normalized.copy()
    
    for metric in available_metrics:
        weight = ponderaciones.get(metric, 5) / 5.0  # Normalizar peso a un factor (1-2)
        weighted_metrics[metric] = weighted_metrics[metric] * weight
        weighted_base[metric] = weighted_base[metric] * weight
    
    # Calcular similitud coseno
    similarity_scores = []
    for idx, row in weighted_metrics.iterrows():
        sim = cosine_similarity([row.values], [weighted_base.iloc[0].values])[0][0]
        similarity_scores.append((metrics_df.loc[idx, "NOMBRE"], sim))
    
    # Crear DataFrame de resultados
    similarity_df = pd.DataFrame(similarity_scores, columns=["NOMBRE", "similitud"])
    
    # Eliminar duplicados basados en el nombre antes de ordenar
    similarity_df = similarity_df.sort_values("similitud", ascending=False)
    similarity_df = similarity_df.drop_duplicates(subset=["NOMBRE"])
    
    # Unir con datos originales para mostrar todas las métricas
    result_df = pd.merge(similarity_df, filtered_df, on="NOMBRE")
    
    # Ordenar por similitud y limitar a los 10 más similares
    result_df = result_df.sort_values("similitud", ascending=False).head(10)
    
    # Convertir similitud a porcentaje
    result_df["similitud"] = (result_df["similitud"] * 100).round(1)
    
    return result_df

# Callback para buscar jugadores similares y actualizar visualizaciones
@callback(
    Output("jugadores-similares-data", "data"),
    Output("jugadores-similares-chart", "children"),
    Output("jugadores-similares-table", "children"),
    Input("buscar-similares-button", "n_clicks"),
    State("jugador-base-dropdown", "value"),
    State("posicion-jugador-dropdown", "value"),
    State("metricas-seleccionadas-store", "data"),
    State("posicion-match-checklist", "value"),
    State("edad-range-slider", "value"),
    State("minutos-jugados-slider", "value")  # NUEVO: Añadir el estado del slider de minutos
)
def update_jugadores_similares(n_clicks, jugador_base, posicion_jugador, metricas, posicion_match, edad_range, min_minutos):
    if not n_clicks or not jugador_base or not metricas:
        return None, "Selecciona un jugador base y métricas para buscar similitudes", ""
    
    # Recopilar las ponderaciones directamente por sus IDs
    ponderaciones = {}
    ctx = dash.callback_context
    
    # Asignar ponderaciones a cada métrica buscando los valores directamente
    for metrica in metricas:
        slider_id = f"ponderacion-slider-{metrica}"
        try:
            # Intentar obtener el valor desde los inputs del callback
            slider_value = ctx.inputs.get(f"{slider_id}.value")
            if slider_value is not None:
                ponderaciones[metrica] = slider_value
            else:
                # Si no está disponible, usar valor predeterminado
                ponderaciones[metrica] = 5
        except:
            # En caso de error, usar valor predeterminado
            ponderaciones[metrica] = 5
    
    print(f"Métricas seleccionadas: {metricas}")        
    print(f"Ponderaciones finales: {ponderaciones}")
    
    # Configurar filtros
    filtros = {
        "posicion": posicion_match,
        "edad": edad_range,
        "min_minutos": min_minutos  # NUEVO: Incluir filtro de minutos jugados
    }
    
    # Buscar jugadores similares
    similares_df = find_similar_players(jugador_base, metricas, ponderaciones, filtros, posicion_jugador)
    
    if similares_df.empty:
        return None, "No se encontraron jugadores similares con los criterios seleccionados", ""
    
    # Crear gráfico radar para comparación
    def create_radar_chart(jugador_base, similar_players, metrics):
        df = data_manager.get_data()
        
        # Filtrar los datos del jugador base por la posición seleccionada
        jugador_base_data = df[df["NOMBRE"] == jugador_base]
        if posicion_jugador:
            jugador_base_data = jugador_base_data[jugador_base_data["POSICIÓN"] == posicion_jugador]
        
        base_data = jugador_base_data[metrics].iloc[0]
        
        # Seleccionar los 3 jugadores más similares
        top_similares = similar_players.head(3)
        
        # Normalizar los datos para el radar
        all_players_data = pd.concat([
            base_data.to_frame().T,
            top_similares[metrics]
        ])
        
        # Aplicar MinMaxScaler para normalizar valores entre 0 y 1
        scaler = MinMaxScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(all_players_data),
            columns=metrics
        )
        
        # Preparar datos para el gráfico
        fig = go.Figure()
        
        # Añadir jugador base
        fig.add_trace(go.Scatterpolar(
            r=normalized_data.iloc[0].values.tolist() + [normalized_data.iloc[0].values.tolist()[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f"{jugador_base} ({posicion_jugador})" if posicion_jugador else jugador_base,
            line=dict(color='#FFFF00', width=3),
            fillcolor='rgba(255, 255, 0, 0.2)'
        ))
        
        # Añadir jugadores similares
        colors = ['#00BFFF', '#00FF00', '#FF00FF']
        # Definir los colores de relleno directamente
        fill_colors = ['rgba(0, 191, 255, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(255, 0, 255, 0.1)']
        
        for i, idx in enumerate(top_similares.index[:3]):
            if i < len(normalized_data) - 1:
                player_name = top_similares.loc[idx, "NOMBRE"]
                player_pos = top_similares.loc[idx, "POSICIÓN"]
                player_idx = i + 1  # Índice en normalized_data
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_data.iloc[player_idx].values.tolist() + [normalized_data.iloc[player_idx].values.tolist()[0]],
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=f"{player_name} - {player_pos} ({top_similares.loc[idx, 'similitud']}%)",
                    line=dict(color=colors[i % len(colors)]),
                    fillcolor=fill_colors[i % len(fill_colors)]
                ))
        
        # CAMBIO 5: Mejorar la configuración del gráfico de radar
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False,
                    linecolor='rgba(255,255,255,0.2)',
                ),
                angularaxis=dict(
                    tickfont=dict(color='white', size=10),
                    linecolor='rgba(255,255,255,0.2)',
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title=dict(
                text='Comparación de Métricas',
                font=dict(color='#00BFFF', size=20)
            ),
            legend=dict(
                font=dict(color='white'),
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(l=20, r=20, t=60, b=100),
            height=600,  # Mayor altura para mejor visualización
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    
    # Crear tabla de resultados
    def create_similarity_table(similar_players, metrics):
        # Seleccionar columnas para mostrar (incluir EQUIPO después de NOMBRE)
        # Añadimos una columna visual al principio para las imágenes
        display_cols = ['FOTO', 'NOMBRE']

        # Añadir EQUIPO si está disponible
        if "EQUIPO" in similar_players.columns:
            display_cols.append('EQUIPO')

        # Añadir el resto de columnas
        display_cols += ['similitud', 'POSICIÓN', 'EDAD'] + metrics
        # Eliminar 'FOTO' de la lista para el procesamiento de datos
        data_cols = [col for col in display_cols if col != 'FOTO' and col in similar_players.columns]

        # Crear encabezados de tabla
        header = html.Tr([
            # La primera columna es para fotos (sin texto)
            html.Th("", style={"padding": "12px", "width": "60px"}) if 'FOTO' in display_cols else None,
        ] + [
            html.Th(col, style={"padding": "12px"}) for col in data_cols
        ])
        
        # Crear filas de datos
        rows = []
        for i, (idx, row) in enumerate(similar_players.iterrows()):
            # Alternar colores de fondo para las filas
            bg_color = "rgba(0, 191, 255, 0.1)" if i % 2 == 0 else "rgba(0, 0, 0, 0)"
            
            # Crear celdas para cada columna
            cells = []
            
            # Añadir foto en miniatura como primera celda
            if 'FOTO' in display_cols:
                # Obtener la URL de la imagen del jugador
                player_name = row['NOMBRE']
                imagen_url = data_manager.get_player_image_url(player_name)
                
                # Celda con imagen
                cells.append(html.Td(
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
                ))
            
            # Añadir el resto de celdas con datos
            for col in data_cols:
                value = row[col]
                
                # Formatear el valor según el tipo de columna
                if col == 'similitud':
                    formatted_value = f"{value}%"
                elif col == 'EDAD' and pd.notna(value):
                    formatted_value = f"{int(value)}"
                elif isinstance(value, (int, float)) and pd.notna(value):
                    formatted_value = f"{value:.2f}" if value % 1 != 0 else f"{int(value)}"
                else:
                    formatted_value = str(value) if pd.notna(value) else "-"
                
                cells.append(html.Td(formatted_value, style={"padding": "10px"}))
            
            # Añadir fila completa a la lista
            rows.append(html.Tr(cells, style={"background-color": bg_color}))
        
        # Construir tabla completa
        table = html.Table(
            [header] + rows,
            style={
                "width": "100%", 
                "border-collapse": "collapse",
                "color": "white",
                "margin-top": "20px"
            }
        )
        
        return html.Div([
            html.H3("Jugadores Más Similares", style={"color": "#00BFFF", "margin-bottom": "15px"}),
            table
        ])
    
    # Crear visualizaciones
    radar_chart = create_radar_chart(jugador_base, similares_df, metricas)
    similarity_table = create_similarity_table(similares_df, metricas)
    
    # Convertir DataFrame a diccionario para almacenar en dcc.Store
    similares_data = similares_df.to_dict('records')
    
    return similares_data, radar_chart, similarity_table

