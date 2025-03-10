import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from flask_login import login_user, current_user
from flask import redirect
from app import authenticate_user, server

# Layout para la página de login
layout = html.Div(
    className="login-container",
    children=[
        html.Div(
            className="login-card",
            children=[
                html.Div(
                    className="login-logo",
                    children=[
                        html.H1("SOCCER DATA DBC", className="login-title"),
                    ]
                ),
                # Campo de usuario con autoFocus para que el cursor aparezca aquí al cargar
                dbc.Input(
                    id="login-username",
                    type="text",
                    placeholder="Usuario",
                    className="mb-3",
                    autoFocus=True,
                    style={
                        "backgroundColor": "#2a2a2a", 
                        "color": "white", 
                        "borderColor": "#00BFFF", 
                        "padding": "12px"
                    }
                ),
                # Campo de contraseña
                dbc.Input(
                    id="login-password",
                    type="password",
                    placeholder="Contraseña",
                    className="mb-3",
                    style={
                        "backgroundColor": "#2a2a2a", 
                        "color": "white", 
                        "borderColor": "#00BFFF", 
                        "padding": "12px"
                    }
                ),
                # Botón de inicio de sesión
                dbc.Button(
                    "INICIAR SESIÓN",
                    id="login-button",
                    color="primary",
                    className="mt-3",
                    style={
                        "backgroundColor": "#00BFFF", 
                        "width": "100%", 
                        "padding": "12px"
                    }
                ),
                # Div para mostrar mensajes de error
                html.Div(
                    id="login-error", 
                    style={"color": "#FF6B6B", "marginTop": "15px"}
                ),
                # Location para redireccionar
                dcc.Location(id='login-redirect', refresh=True),
            ],
        ),
    ],
)

# Callback para validar las credenciales con Flask-Login
@callback(
    [Output('login-error', 'children'),
     Output('login-redirect', 'pathname')],
    [Input('login-button', 'n_clicks'),
     Input('login-password', 'n_submit')],  # Añadido para permitir presionar Enter
    [State('login-username', 'value'),
     State('login-password', 'value')],
    prevent_initial_call=True
)
def validate_login(n_clicks, n_submit, username, password):
    # No hacer nada si no hay clic o Enter presionado
    if not n_clicks and not n_submit:
        raise PreventUpdate
    
    # Verificar si el usuario ya está autenticado
    if current_user.is_authenticated:
        return "", "/dashboard"
    
    # Comprobar que se ingresaron credenciales
    if not username or not password:
        return "Por favor ingresa usuario y contraseña", dash.no_update
    
    # Imprimir para depuración
    print(f"Login intentado - Usuario: '{username}', Contraseña: '{password}'")
    
    # Autenticar al usuario con Flask-Login
    user = authenticate_user(username, password)
    if user:
        login_user(user)
        print("Login exitoso - redirigiendo al dashboard")
        return "", "/dashboard"
    else:
        print(f"Login fallido - usuario: '{username}', pass: '{password}'")
        return "Usuario o contraseña incorrectos", dash.no_update