import pandas as pd
import os

class DataManager:
    """
    Clase para gestionar la carga y procesamiento de datos.
    Implementa el patrón Singleton para cargar los datos una sola vez.
    """
    _instance = None
    _data = None
    _nutrition_data = None  # Nueva variable para datos de nutrición

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._load_data()
            cls._instance._load_nutrition_data()  # Cargar datos de nutrición
        return cls._instance

    def _load_data(self):
        """
        Carga los datos únicamente de los archivos CSV.
        """
        try:
            # Ruta a los archivos de datos
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            file1_path = os.path.join(data_dir, 'estadisticas_limpio.csv')
            file2_path = os.path.join(data_dir, 'estadisticas_final.csv')
            
            dataframes = []
            
            # Cargar el primer archivo
            if os.path.exists(file1_path):
                print(f"Cargando datos desde: {file1_path}")
                df1 = pd.read_csv(file1_path)
                dataframes.append(df1)
            else:
                print(f"Archivo CSV no encontrado: {file1_path}")
            
            # Cargar el segundo archivo
            if os.path.exists(file2_path):
                print(f"Cargando datos desde: {file2_path}")
                df2 = pd.read_csv(file2_path)
                dataframes.append(df2)
            else:
                print(f"Archivo CSV no encontrado: {file2_path}")
            
            # Si no se encontró ningún archivo, intentar buscar cualquier archivo CSV
            if not dataframes:
                print("Buscando archivos CSV alternativos en la carpeta data...")
                if os.path.exists(data_dir):
                    for file in os.listdir(data_dir):
                        if file.endswith('.csv') and file != 'nutricion_jugadores.csv':  # Excluir el archivo de nutrición
                            alt_path = os.path.join(data_dir, file)
                            print(f"Usando archivo CSV alternativo: {alt_path}")
                            df_alt = pd.read_csv(alt_path)
                            dataframes.append(df_alt)
                            break
            
            # Combinar los dataframes si hay más de uno
            if len(dataframes) > 1:
                self._data = pd.concat(dataframes, ignore_index=True)
                print(f"Datos combinados: {len(self._data)} registros")
            elif len(dataframes) == 1:
                self._data = dataframes[0]
                print(f"Datos cargados: {len(self._data)} registros")
            else:
                raise Exception("No se encontraron archivos CSV para cargar")
            
            # Limpiar espacios en los nombres de las columnas
            self._data.columns = [col.strip() if isinstance(col, str) else col for col in self._data.columns]
            
            # Mostrar las columnas para depuración
            print("Columnas en el archivo (después de limpiar espacios):")
            for col in self._data.columns:
                print(f"  - '{col}'")
            
            # Verificar si tenemos las columnas clave
            required_columns = ['NOMBRE', 'POSICIÓN']
            missing_columns = [col for col in required_columns if col not in self._data.columns]
            
            if missing_columns:
                print(f"Columnas faltantes después de limpiar: {missing_columns}")
                
                # Buscar columnas con nombres similares (ignorando espacios y mayúsculas/minúsculas)
                clean_columns = {col.upper().strip(): col for col in self._data.columns}
                
                for missing in missing_columns:
                    missing_upper = missing.upper().strip()
                    if missing_upper in clean_columns:
                        original_col = clean_columns[missing_upper]
                        print(f"Renombrando columna '{original_col}' a '{missing}'")
                        self._data.rename(columns={original_col: missing}, inplace=True)
            
            # Realizar limpieza básica de datos
            self._clean_data()
            
            # Verificar columnas después de la limpieza
            print(f"Columnas disponibles después de la limpieza: {self._data.columns.tolist()}")
            
            print(f"Datos cargados exitosamente: {len(self._data)} registros.")
            
        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            # Datos de ejemplo si hay error
            self._data = pd.DataFrame({
                'NOMBRE': ['Jugador_1', 'Jugador_2', 'Jugador_3'],
                'POSICIÓN': ['Delantero', 'Centrocampista', 'Defensa'],
                'FECHA NACIMIENTO': ['1990-01-01', '1992-05-15', '1988-11-30'],
                'PIERNA': ['Derecha', 'Izquierda', 'Derecha'],
                'LUGAR NACIMIENTO': ['España', 'Argentina', 'Brasil'],
                'MINUTOS JUGADOS': [1800, 1650, 1920],
                'PARTIDOS JUGADOS': [20, 18, 21],
                'DUELOS TERRESTRES GANADOS': [45, 38, 62],
                'DUELOS AÉREOS GANADOS': [28, 15, 40],
                'INTERCEPCIONES': [12, 25, 35]
            })
            print("Se han cargado datos de ejemplo debido al error.")

    def _load_nutrition_data(self):
        """
        Carga los datos de nutrición del archivo CSV.
        """
        try:
            # Ruta al archivo de nutrición
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            nutrition_path = os.path.join(data_dir, 'nutricion_jugadores.csv')
            
            if os.path.exists(nutrition_path):
                print(f"Cargando datos de nutrición desde: {nutrition_path}")
                self._nutrition_data = pd.read_csv(nutrition_path)
                
                # Limpiar espacios en los nombres de las columnas
                self._nutrition_data.columns = [col.strip() if isinstance(col, str) else col for col in self._nutrition_data.columns]
                
                # Mostrar las columnas para depuración
                print("Columnas en el archivo de nutrición (después de limpiar espacios):")
                for col in self._nutrition_data.columns:
                    print(f"  - '{col}'")
                
                # Realizar limpieza básica de datos
                self._clean_nutrition_data()
                
                print(f"Datos de nutrición cargados exitosamente: {len(self._nutrition_data)} registros.")
            else:
                print(f"Archivo de nutrición no encontrado: {nutrition_path}")
                self._nutrition_data = pd.DataFrame()
                
        except Exception as e:
            print(f"Error al cargar los datos de nutrición: {e}")
            self._nutrition_data = pd.DataFrame()
            print("No se han podido cargar los datos de nutrición.")

    def _clean_data(self):
        """
        Realiza la limpieza y preparación de los datos.
        """
        if self._data.empty:
            return
        
        # Eliminar filas con todos los valores NaN
        self._data = self._data.dropna(how='all')
        
        # Asegurar que las métricas numéricas sean de tipo float
        numeric_cols = self._data.select_dtypes(include=['int', 'float']).columns
        for col in numeric_cols:
            self._data[col] = pd.to_numeric(self._data[col], errors='coerce')

    def _clean_nutrition_data(self):
        """
        Realiza la limpieza y preparación de los datos de nutrición.
        """
        if self._nutrition_data is None or self._nutrition_data.empty:
            return
        
        # Eliminar filas con todos los valores NaN
        self._nutrition_data = self._nutrition_data.dropna(how='all')
        
        # Asegurar que las métricas numéricas sean de tipo float
        numeric_cols = self._nutrition_data.select_dtypes(include=['int', 'float']).columns
        for col in numeric_cols:
            self._nutrition_data[col] = pd.to_numeric(self._nutrition_data[col], errors='coerce')

    def get_data(self):
        """
        Retorna el DataFrame completo.
        """
        return self._data.copy() if self._data is not None else pd.DataFrame()
    
    def get_nutrition_data(self):
        """
        Retorna el DataFrame de nutrición completo.
        """
        return self._nutrition_data.copy() if self._nutrition_data is not None else pd.DataFrame()

    def get_unique_values(self, column, nutrition=False):
        """
        Retorna los valores únicos de una columna específica.
        
        Args:
            column (str): Nombre de la columna
            nutrition (bool): Si es True, busca la columna en los datos de nutrición
        """
        df = self._nutrition_data if nutrition else self._data
        
        if df is None or df.empty or column not in df.columns:
            print(f"Columna '{column}' no encontrada en get_unique_values")
            # Intentar buscar columna con espacios
            column_strip = column.strip()
            for col in df.columns if df is not None and not df.empty else []:
                if col.strip() == column_strip:
                    print(f"Usando columna '{col}' en lugar de '{column}'")
                    return sorted(df[col].dropna().unique().tolist())
            return []
        return sorted(df[column].dropna().unique().tolist())
    
    def get_player_data(self, player_name, nutrition=False):
        """
        Retorna los datos de un jugador específico.
        
        Args:
            player_name (str): Nombre del jugador
            nutrition (bool): Si es True, busca en los datos de nutrición
        """
        df = self._nutrition_data if nutrition else self._data
        
        if df is None or df.empty or not player_name:
            return pd.DataFrame()
        
        if 'NOMBRE' not in df.columns:
            print("Error crítico: columna 'NOMBRE' no disponible en get_player_data")
            return pd.DataFrame()
            
        return df[df['NOMBRE'] == player_name]
    
    def get_filtered_data(self, filters=None, nutrition=False):
        """
        Retorna los datos filtrados según los criterios especificados.
        
        Args:
            filters (dict): Diccionario con los filtros a aplicar.
                            Formato: {columna: valor} o {columna: [valor1, valor2, ...]}
            nutrition (bool): Si es True, aplica los filtros a los datos de nutrición
        """
        df = self._nutrition_data if nutrition else self._data
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        if not filters:
            return df.copy()
        
        filtered_data = df.copy()
        
        for column, value in filters.items():
            if column in filtered_data.columns:
                if isinstance(value, list):
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                else:
                    filtered_data = filtered_data[filtered_data[column] == value]
            else:
                print(f"Columna '{column}' no encontrada en get_filtered_data")
                # Intentar buscar columna con espacios
                column_strip = column.strip()
                for col in filtered_data.columns:
                    if col.strip() == column_strip:
                        print(f"Usando columna '{col}' en lugar de '{column}'")
                        if isinstance(value, list):
                            filtered_data = filtered_data[filtered_data[col].isin(value)]
                        else:
                            filtered_data = filtered_data[filtered_data[col] == value]
                        break
        
        return filtered_data
    
    def get_player_image_url(self, player_name):
        """
        Obtiene la ruta local de la imagen del jugador.

        Args:
            player_name (str): Nombre del jugador
    
        Returns:
            str: Ruta relativa a la imagen o None si no se encuentra
        """
        if not player_name:
            return None
    
        # Convertir el nombre a mayúsculas para buscar el archivo
        player_name_upper = player_name.upper()
    
        # Construir ruta relativa para Dash con el nombre en mayúsculas
        image_path = f"/assets/players/{player_name_upper}.png"

        # Verificar si el archivo existe físicamente (también en mayúsculas)
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'players')
        full_path = os.path.join(assets_folder, f"{player_name_upper}.png")

        if os.path.exists(full_path):
            return image_path

        return None