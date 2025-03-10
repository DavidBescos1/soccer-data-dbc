import pandas as pd

print("Iniciando conversión de archivos Excel a CSV...")

# Convertir el archivo grande
print("Convirtiendo estadisticas_limpio.xlsx...")
df_limpio = pd.read_excel('data/estadisticas_limpio.xlsx')
df_limpio.to_csv('data/estadisticas_limpio.csv', index=False)
print(f"Archivo convertido. Tamaño original: {df_limpio.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Convertir el archivo más pequeño
print("Convirtiendo estadisticas_final.xlsx...")
df_final = pd.read_excel('data/estadisticas_final.xlsx')
df_final.to_csv('data/estadisticas_final.csv', index=False)
print(f"Archivo convertido. Tamaño original: {df_final.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("Conversión completada exitosamente!")