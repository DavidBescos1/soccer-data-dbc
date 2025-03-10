import dash
import dash_bootstrap_components as dbc
from flask_login import LoginManager, UserMixin

# Definición de clase User para Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

    @staticmethod
    def get(user_id):
        if user_id == 'admin':
            return User(user_id)
        return None

# Inicializar la aplicación Dash con tema personalizado
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)

# Configuración del título de la página
app.title = "Soccer Data DBC"
server = app.server
server.config['SECRET_KEY'] = 'una_clave_secreta_muy_segura'

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Función de autenticación
def authenticate_user(username, password):
    if username == 'admin' and password == 'admin':
        return User(username)
    return None

# Exportar server para poder importarlo en index.py
# Exportar también User y otras funciones que necesitamos