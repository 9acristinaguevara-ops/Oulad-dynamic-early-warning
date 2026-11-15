# Auditoría de tablas y cifras del manuscrito OULAD

**Documento auditado:** *Early Warning of Student Withdrawal in Virtual Learning Environments Through Dynamic Engagement Signals: A Temporal Machine Learning and Longitudinal Analysis Using OULAD* (23 páginas, versión localizada en la biblioteca).  
**Fecha de revisión:** 14 de noviembre de 2026.  
**Estado:** revisión de contenido; el DOCX original no fue modificado.

## 1. Dictamen general

Las tablas predictivas 1–6 describen una evaluación anterior, basada en particiones por presentación y con solapamiento de estudiantes entre cohortes. Ya no representan la evaluación estricta por estudiante de los experimentos 144–151. No deben trasladarse sin cambios a la versión final.

La tabla 7 corresponde a un análisis distinto de persistencia global entre transiciones semanales. No debe confundirse con el análisis de diferencias de persistencia por retiro futuro de los experimentos 140–141. Su vigencia numérica requiere contrastar el CSV exacto de origen antes de conservarla.

## 2. Tabla por tabla

### Tabla 1 — Composición de la cohorte

El manuscrito presenta:

- 2013B: 4.086 registros, 757 retiros.
- 2013J: 7.620 registros, 1.155 retiros.
- 2014B: 6.420 registros, 1.234 retiros.
- 2014J: 9.009 registros, 1.579 retiros.

Estos tamaños pertenecen al diseño predictivo anterior y no son las cohortes independientes del experimento 144. La tabla debe rehacerse con cinco particiones: entrenamiento 2013B (3.975 matrículas), calibración 2013J (3.809), selección de umbral 2013J (3.811), prueba 2014B (5.778) y prueba 2014J (7.783). La tabla debe indicar matrículas y estudiantes únicos, porque no son equivalentes.

No sumar las cinco cohortes y llamar al total “cohorte analítica general”: los 27.135 registros de la auditoría 142 son otra población, construida para el análisis general.

### Tabla 2 — Evaluación predictiva principal

La tabla usa 6.420 y 9.009 registros y AUC cercanas a 0,6216 y 0,6444. Esas cifras corresponden al modelo predictivo anterior y no a la evaluación estricta por estudiante ni a la comparación estandarizada de bloques.

Para la comparación principal actual, el experimento 150 informa para el modelo combinado sin identidad de módulo:

- 2014B: AUC 0,59662; Brier 0,14474; 5.778 matrículas y 5.503 estudiantes.
- 2014J: AUC 0,63570; Brier 0,13181; 7.783 matrículas y 7.479 estudiantes.

No reemplazar automáticamente toda la tabla 2 con estas cuatro métricas: se debe definir primero qué modelo se presentará como modelo principal y añadir las métricas de umbral solo con un protocolo de selección comparable y cohortes estrictas. La versión del manuscrito también deja vacías las celdas de alertas y tasa de alertas para 2014J.

### Tabla 3 — Exclusión de estudiantes compartidos con entrenamiento

La tabla elimina estudiantes que comparten 2013B con pruebas posteriores, pero no garantiza independencia entre todas las cohortes ni resuelve el solapamiento de estudiantes entre las particiones de 2013J. Los experimentos 143–144 proporcionan una separación más rigurosa. La tabla 3 debe reemplazarse por una descripción de la validación estricta y su resultado, no conservarse como prueba principal de independencia.

### Tabla 4 — Capacidad de alertas

La tabla usa rankings top 10 %, 20 % y 30 % del modelo anterior y tamaños de prueba anteriores. No es intercambiable con el análisis 150, que selecciona umbrales en una cohorte de selección independiente y evalúa la tasa resultante en pruebas nuevas.

En el experimento 150, el modelo combinado, con umbrales seleccionados para capacidades objetivo de 20 %, 30 % y 40 %, produjo tasas de alerta en prueba de:

| Capacidad objetivo | 2014B | 2014J |
|---|---:|---:|
| 20 % | 31,05 % | 27,87 % |
| 30 % | 39,69 % | 38,24 % |
| 40 % | 50,36 % | 45,75 % |

Estas diferencias muestran que la capacidad observada en pruebas no coincide necesariamente con la capacidad objetivo en selección. El modelo contextual solo produjo una tasa de alertas de aproximadamente 87,38 % en la selección para los tres objetivos, por lo que no alcanzó capacidades 20/30/40 %.

### Tabla 5 — Indicadores de valores faltantes

La tabla compara modelos del procedimiento anterior (AUC 0,6216/0,6444, etc.). El análisis de sensibilidad de indicadores de ausencia se debe describir con los resultados y las particiones del protocolo final si se conserva. No mezclar resultados de muestras y particiones distintas.

### Tabla 6 — Modelos calibrados y no calibrados

La tabla compara métricas del modelo anterior. La calibración y sus umbrales no deben compararse con resultados de validación estricta sin reconstruir la tabla sobre las cinco cohortes independientes y asegurar un protocolo comparable. AUC y Brier son métricas distintas: calibración puede mejorar Brier sin cambiar el ordenamiento (AUC).

### Tabla 7 — Comparaciones de persistencia entre transiciones

La tabla informa diferencias entre coeficientes de persistencia global de las transiciones 0–1, 1–2, 2–3 y 3–4, con Holm. Es un análisis distinto de la diferencia entre grupos de retiro futuro en la transición 2–3, evaluada en los experimentos 140–141. Antes de conservarla, verificar el CSV que generó la tabla, el modelo exacto, el tamaño muestral, la unidad de observación y la versión del script. No sustituir los resultados de la tabla 7 por los de 141, ni viceversa.

## 3. Inconsistencias editoriales adicionales

1. **Resumen y sección 3.3:** AUC 0,6216/0,6444 (en otros lugares 0,6217/0,6443) son resultados del protocolo predictivo anterior. Actualizar el resumen y el cuerpo solo después de decidir la tabla predictiva final.
2. **Sección 3.1:** la distribución general de 27.135 registros (4.725 positivos y 22.410 negativos) sí corresponde a la auditoría 142, pero la tabla 1 inmediatamente posterior describe particiones antiguas. Se mezclan dos poblaciones y diseños distintos.
3. **Secciones 3.4–3.7 y discusión:** las cifras de pruebas antiguas no deben presentarse como resultados de la validación estricta por estudiante.
4. **Sección 3.11:** pseudo-R² = 0,0085 para SD contradice el experimento 124, que informa 0,0456. El CV informa 0,0466. Verificar el CSV antes de editar.
5. **Sección 3.12:** documentar el criterio de alta actividad como media de al menos 54 clics. La interacción CV × número de semanas no mantuvo significación en ese subconjunto.
6. **Secciones de persistencia:** distinguir las comparaciones globales entre coeficientes de transición de la diferencia entre grupos de retiro futuro. Para esta última, el hallazgo robusto de semanas 2–3 procede del experimento 141 con Holm.
7. **Figura 2:** el manuscrito contiene el pie de figura, pero la extracción textual no muestra una figura incorporada en ese lugar. Inspeccionar visualmente el DOCX y comprobar que el archivo de imagen esté realmente incrustado.
8. **Referencias:** las páginas finales terminan en conclusiones; no se observa una sección bibliográfica final. Las citas en el texto necesitan una lista de referencias completa y verificada.
9. **Términos:** sustituir “elapsed course weeks” por “number of observed weekly measurements” cuando se refiera al análisis CV; aclarar que las unidades predictivas son matrículas estudiante-curso.

## 4. Acciones necesarias antes de editar el manuscrito

- Obtener/leer los CSV definitivos locales de los experimentos 144, 150, 151, 124, 127–129 y 141.
- Crear una tabla maestra con una fila por resultado y columnas: afirmación, estimación, intervalo/SE, muestra, unidad, script, CSV y estado de verificación.
- Decidir qué configuración predictiva será el modelo principal. La comparación estandarizada del experimento 150 favorece al combinado sin identidad de módulo en AUC, pero no convierte automáticamente sus métricas de umbral en una política de alerta validada.
- Reconstruir las tablas 1–6 con cohortes estrictas y separar claramente cohorte analítica general de cohortes predictivas.
- Verificar la procedencia exacta de la tabla 7 y la figura 2.
- Añadir referencias bibliográficas completas y comprobar todas las citas.

**Protección:** esta auditoría es un documento nuevo. No modifica el DOCX original, los datos ni los CSV experimentales.
