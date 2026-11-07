# Plan de investigación (borrador inicial)

## Pregunta
¿Los cambios temporales en la actividad del entorno virtual contienen señales tempranas compatibles con una transición hacia el abandono académico, antes de que el resultado final sea registrado?

## Hipótesis que se deben contrastar
- H1: las trayectorias semanales de actividad difieren entre estudiantes que posteriormente aparecen como `Withdrawn` y otros resultados.
- H2: algunos indicadores dinámicos (por ejemplo, disminución persistente de actividad o aumento de intervalos sin interacción) pueden anticipar el resultado con información disponible hasta ese momento.
- H3: los modelos dinámicos aportan valor frente a baselines simples, bajo validación temporal y evitando fuga de información.

Estas son hipótesis, no resultados.

## Precauciones
1. OULAD no mide directamente fatiga psicológica.
2. `final_result == Withdrawn` es una etiqueta observada al final; no se debe convertir en una señal disponible desde el comienzo.
3. La unidad analítica inicial será estudiante-módulo-presentación, identificada por `id_student`, `code_module` y `code_presentation`.
4. Los registros de `studentVle.csv` tienen `date` en días relativos al inicio de la presentación y `sum_click`; revisar valores especiales antes de agregarlos.
5. Evitar que las observaciones posteriores al momento de predicción entren en predictores.
6. Separar evaluación por estudiante y/o presentación cuando sea necesario para reducir contaminación entre conjuntos.
7. Definir el origen temporal, el horizonte de predicción y la censura antes de comparar modelos.

## Candidato de dinámica (no fijado todavía)
Podrían formularse estados latentes como engagement E(t) y fatiga operacional F(t), por ejemplo:
dE/dt = recuperación(E, contexto) - agotamiento(F, carga)
dF/dt = acumulación(carga, irregularidad) - recuperación(F)

No se deben ajustar ecuaciones ni asignar significado psicológico a los estados hasta definir variables observables, restricciones, identificabilidad y pruebas de sensibilidad. Una alternativa será comparar modelos de estado con baselines predictivos y métodos de detección de cambios.
