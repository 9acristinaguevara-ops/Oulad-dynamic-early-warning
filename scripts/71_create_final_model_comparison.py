
import pandas as pd


# 1. Cargar resultados de los dos modelos
ruta = "outputs/tables/global_fallback_test_results_v1.csv"

df = pd.read_csv(ruta)

print("COMPARACION FINAL DE MODELOS")
print("Registros analizados:", len(df))


# 2. Separar los resultados de cada modelo
actividad = df[
    df["modelo"] == "Solo actividad"
].copy()

evaluaciones = df[
    df["modelo"] == "Actividad + evaluaciones"
].copy()


# 3. Seleccionar las columnas importantes
columnas = [
    "periodo",
    "modulo",
    "umbral",
    "precision",
    "recall",
    "f1",
    "auc",
    "alertas",
    "retiradas"
]

actividad = actividad[columnas]
evaluaciones = evaluaciones[columnas]


# 4. Cambiar nombres para distinguir los modelos
actividad = actividad.rename(columns={
    "umbral": "umbral_actividad",
    "precision": "precision_actividad",
    "recall": "recall_actividad",
    "f1": "f1_actividad",
    "auc": "auc_actividad",
    "alertas": "alertas_actividad",
    "retiradas": "retiradas_actividad"
})

evaluaciones = evaluaciones.rename(columns={
    "umbral": "umbral_evaluaciones",
    "precision": "precision_evaluaciones",
    "recall": "recall_evaluaciones",
    "f1": "f1_evaluaciones",
    "auc": "auc_evaluaciones",
    "alertas": "alertas_evaluaciones",
    "retiradas": "retiradas_evaluaciones"
})


# 5. Unir los resultados
comparacion = pd.merge(
    actividad,
    evaluaciones,
    on=["periodo", "modulo"],
    how="outer"
)


# 6. Calcular diferencias entre modelos
comparacion["diferencia_f1"] = (
    comparacion["f1_evaluaciones"] -
    comparacion["f1_actividad"]
)

comparacion["diferencia_auc"] = (
    comparacion["auc_evaluaciones"] -
    comparacion["auc_actividad"]
)

comparacion["mejor_f1"] = comparacion.apply(
    lambda fila:
        "Actividad + evaluaciones"
        if fila["diferencia_f1"] > 0
        else (
            "Solo actividad"
            if fila["diferencia_f1"] < 0
            else "Empate"
        ),
    axis=1
)


# 7. Ordenar los resultados
comparacion = comparacion.sort_values(
    ["periodo", "modulo"]
)


# 8. Guardar tabla completa
salida = (
    "outputs/tables/"
    "final_model_comparison_v1.csv"
)

comparacion.to_csv(salida, index=False)


# 9. Mostrar resumen
print("\nRESULTADOS POR PERIODO Y MODULO")

for _, fila in comparacion.iterrows():

    print(
        "Periodo:", fila["periodo"],
        "| Modulo:", fila["modulo"],
        "| F1 actividad:",
        round(fila["f1_actividad"], 3),
        "| F1 evaluaciones:",
        round(fila["f1_evaluaciones"], 3),
        "| Diferencia:",
        round(fila["diferencia_f1"], 3),
        "| Mejor F1:", fila["mejor_f1"]
    )


# 10. Resumen general
print("\nRESUMEN GENERAL")

print(
    "F1 promedio - Solo actividad:",
    round(comparacion["f1_actividad"].mean(), 3)
)

print(
    "F1 promedio - Actividad + evaluaciones:",
    round(comparacion["f1_evaluaciones"].mean(), 3)
)

print(
    "AUC promedio - Solo actividad:",
    round(comparacion["auc_actividad"].mean(), 3)
)

print(
    "AUC promedio - Actividad + evaluaciones:",
    round(comparacion["auc_evaluaciones"].mean(), 3)
)

print(
    "Modulos-periodos donde mejora el F1:",
    int((comparacion["diferencia_f1"] > 0).sum())
)

print(
    "Modulos-periodos donde empeora el F1:",
    int((comparacion["diferencia_f1"] < 0).sum())
)

print("\nArchivo guardado en:", salida)
print("SCRIPT FINALIZADO")
