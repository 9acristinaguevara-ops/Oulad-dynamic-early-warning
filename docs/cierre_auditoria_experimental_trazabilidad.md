# Cierre de auditoría experimental y trazabilidad científica

**Proyecto:** OULAD — señales longitudinales de participación y predicción temprana del retiro  
**Fecha de corte:** 14 de noviembre de 2026  
**Rama:** `audit-142-complementaria`  
**Alcance:** documentación de resultados ya ejecutados. Este informe no modifica datos, scripts analíticos ni el borrador del artículo.

## 1. Decisión sobre el enfoque científico

Se recomienda formular el estudio como una investigación observacional longitudinal que integra:

1. caracterización de intensidad, persistencia y variabilidad de la participación semanal;
2. análisis de asociaciones entre esos patrones y el retiro posterior;
3. evaluación temporal de modelos de alerta temprana.

No se debe afirmar que se implementó o estimó un sistema de ecuaciones diferenciales. Los scripts auditados implementan agregación semanal, regresiones logísticas y lineales, análisis de transiciones, medidas de variabilidad y modelos predictivos. Tampoco se debe afirmar que se demostró una transición crítica universal, causalidad, agotamiento psicológico ni un mecanismo dinámico diferencial.

**Título de trabajo recomendado:**

*Dynamic Engagement Patterns and Early Prediction of Student Withdrawal in Online Learning Environments: A Longitudinal Study Using OULAD*

Este título es provisional; no constituye una edición del manuscrito.

## 2. Integridad de los datos y definición del objetivo

### 2.1 Reconstrucción e integridad

La auditoría 142 reconstruyó de forma independiente la actividad semanal desde `studentVle.csv`, el objetivo desde los registros de matrícula y el panel de semanas 0–4. Los resultados informados fueron:

- 110 controles `OK` y un control que requiere revisión de definición;
- 627.031 filas de actividad semanal;
- 27.135 registros elegibles en los conjuntos corregido, enriquecido y combinado;
- cero diferencias en las reconstrucciones comparadas de actividad, objetivo, variables de evaluación y panel;
- cero claves duplicadas en los conjuntos auditados;
- 24.887 registros en la muestra analítica de CV y 15.317 trayectorias completas, según las comprobaciones de muestras.

La única observación especial fue un registro `Fail` con retiro administrativo posterior al día 34. No es una discrepancia de reconstrucción: revela que el objetivo representa retiro administrativo posterior al límite temporal, no exclusivamente la categoría final `Withdrawn`.

### 2.2 Definición que debe utilizarse

La cohorte predictiva aplica el límite de observación del día 34. Se excluyen retiros ocurridos en o antes del día 34, matrículas posteriores al límite y casos que no pueden clasificarse con los criterios establecidos. El objetivo positivo representa retiro administrativo posterior al día 34; el negativo representa resultados Pass, Fail o Distinction sin fecha de retiro.

Por tanto, el manuscrito no debe definir el positivo simplemente como “recibir un resultado final Withdrawn”. La fecha administrativa de retiro es el criterio operativo del objetivo. Debe explicarse también el caso excepcional Fail con retiro posterior.

### 2.3 Tamaños que no deben confundirse

- Cohorte analítica general auditada: **27.135 registros de estudiante-curso**.
- Cohortes predictivas estrictamente independientes: **25.156 matrículas** en total.
- La unidad de análisis predictivo es la **matrícula estudiante-curso**, no la persona agregada. Un mismo estudiante puede tener matrículas en cursos distintos; las particiones se mantienen independientes por identificador de estudiante.

## 3. Validación predictiva y sensibilidad (experimentos 107–151)

### 3.1 Separación estricta por estudiante — experimento 144

Se construyeron cinco cohortes sin estudiantes compartidos:

| Cohorte | Matrículas | Estudiantes únicos | Retiros |
|---|---:|---:|---:|
| Entrenamiento 2013B | 3.975 | 3.973 | 650 |
| Calibración 2013J | 3.809 | 3.808 | 577 |
| Selección de umbral 2013J | 3.811 | 3.808 | 578 |
| Prueba 2014B | 5.778 | 5.503 | 1.043 |
| Prueba 2014J | 7.783 | 7.479 | 1.268 |

Se comprobaron cero intersecciones de identificadores entre los diez pares de cohortes. La evaluación estricta produjo AUC de 0,5960 para 2014B y 0,6355 para 2014J en la versión de referencia anterior a la comparación estandarizada de bloques.

### 3.2 Matrículas múltiples — experimentos 145–147

La ponderación para que cada estudiante tuviera peso total uno dentro de su cohorte produjo cambios mínimos en las métricas. El bootstrap pareado por estudiante tampoco mostró una mejora sustantiva del modelo ponderado frente al modelo por matrícula. La unidad de resultado sigue siendo la matrícula, y la incertidumbre debe considerar el agrupamiento de matrículas por estudiante.

Los intervalos bootstrap de los modelos fijos reflejan incertidumbre de la muestra de prueba; no incorporan la variabilidad que resultaría de volver a entrenar y calibrar el modelo.

### 3.3 Comparación de bloques sin identidad de módulo — experimentos 148–151

El experimento 149 mostró que los módulos presentes en evaluación no estaban todos representados en entrenamiento. La codificación de categorías no vistas podía convertir identidades de módulo desconocidas en vectores de ceros. Por ello, la comparación con identidad de módulo no es la base recomendada para la afirmación principal de generalización.

El experimento 150 compara bloques sin identidad de módulo:

- **Estudiante:** actividad y participación individual.
- **Contexto:** número de evaluaciones programadas hasta el día 34.
- **Combinado:** ambos bloques.

La versión más reciente del experimento 150 usa `StandardScaler` dentro de un pipeline, ajustado con los datos de entrenamiento. Los resultados actualizados son:

| Modelo | AUC 2014B | AUC 2014J | Brier 2014B | Brier 2014J |
|---|---:|---:|---:|---:|
| Solo estudiante | 0,5113 | 0,5822 | 0,1489 | 0,1357 |
| Solo contexto | 0,5858 | 0,5515 | 0,1468 | 0,1356 |
| Combinado | **0,5966** | **0,6357** | **0,1447** | **0,1318** |

El modelo combinado ofrece la mayor AUC en ambos periodos. Sin embargo, en 2014B su diferencia frente al modelo contextual no es concluyente en el bootstrap pareado.

### 3.4 Bootstrap pareado actualizado — experimento 151

Se ejecutaron 1.000 réplicas de bootstrap por estudiante, preservando las matrículas de cada estudiante. Diferencia = modelo combinado menos comparador.

| Periodo | Comparación | Diferencia AUC | IC 95 % |
|---|---|---:|---:|
| 2014B | Combinado − estudiante | 0,08529 | [0,06952; 0,10078] |
| 2014B | Combinado − contexto | 0,01085 | [−0,00709; 0,02954] |
| 2014J | Combinado − estudiante | 0,05355 | [0,03818; 0,06968] |
| 2014J | Combinado − contexto | 0,08423 | [0,06901; 0,10122] |

Conclusión: la ventaja del combinado frente al modelo solo estudiante es consistente en ambos periodos. Frente al modelo contextual, no se demuestra una ventaja concluyente en 2014B, pero sí en 2014J. Las fracciones empíricas de réplicas con diferencia positiva no deben describirse como valores p convencionales.

### 3.5 Capacidad operativa de alertas

El modelo contextual produce probabilidades muy discretas y no permite alcanzar de forma fiable capacidades deseadas de alerta del 20 %, 30 % o 40 % mediante un único umbral. En la selección de umbral terminó alertando aproximadamente al 87,38 % para los tres objetivos de capacidad. No se deben presentar esos resultados como si hubiera logrado las capacidades solicitadas.

El modelo combinado tampoco reproduce exactamente en las pruebas la proporción de alertas de la selección. Con umbrales derivados para 20 %, 30 % y 40 %, las tasas de alerta en las pruebas fueron aproximadamente:

| Capacidad objetivo | 2014B | 2014J |
|---|---:|---:|
| 20 % | 31,05 % | 27,87 % |
| 30 % | 39,69 % | 38,24 % |
| 40 % | 50,36 % | 45,75 % |

Esto debe describirse como una limitación de transportabilidad del umbral y de capacidad operativa. Las alertas no están listas para despliegue sin evaluación prospectiva y una decisión institucional sobre la capacidad de intervención.

## 4. Análisis longitudinal (experimentos 124–141)

### 4.1 Variabilidad — experimentos 124–129

- En el experimento 124, pseudo-R² para el modelo con desviación estándar: **0,0456**.
- Pseudo-R² para el modelo con coeficiente de variación: **0,0466**.
- En el modelo conjunto del experimento 125, el término de desviación estándar no fue significativo (β = −0,0195; p = 0,704).
- El CV mantuvo asociación en varias especificaciones y análisis de sensibilidad; esto no demuestra causalidad ni mejora predictiva independiente.
- El análisis de alta actividad utilizó media de al menos **54 clics**; la interacción CV correspondiente no mantuvo significación en ese subconjunto.
- La inferencia con errores agrupados se basó en 22 grupos módulo-presentación, un número reducido que exige cautela.

La cifra 0,0085 que aparece en el borrador para pseudo-R² de desviación estándar contradice el resultado del experimento 124 y debe verificarse/corregirse a 0,0456 si esa sección describe exactamente dicho modelo.

Las métricas longitudinales deben definirse con precisión: semanas 0–4, valores observados disponibles, mínimo de dos semanas observadas, media positiva, desviación estándar muestral (ddof=1) y CV = SD/media. “Número de semanas observadas” no debe sustituirse por “semanas transcurridas”.

### 4.2 Persistencia y transiciones — experimentos 131–141

El resultado más consistente se encuentra en la transición entre las semanas 2 y 3. El experimento 141 aplicó corrección de Holm y agrupó por estudiante:

| Muestra | Diferencia de persistencia (retiro futuro − no retiro) | p ajustado Holm |
|---|---:|---:|
| Al menos 2 transiciones | −0,1037 | 0,00000241 |
| Al menos 3 transiciones | −0,0894 | 0,000225 |
| 4 transiciones completas | −0,1024 | 0,0000459 |

La diferencia es negativa en las tres muestras y permanece significativa tras el ajuste. Las otras transiciones, en general, no conservan significación después del ajuste. Debe presentarse como una asociación localizada en una transición concreta, no como evidencia de una transición crítica universal.

La disponibilidad de observaciones semanales difiere entre grupos. La ausencia de una fila semanal no equivale automáticamente a cero clics; esta limitación debe mantenerse explícita.

### 4.3 Valor incremental de la dinámica — experimento 132

Pseudo-R² informado:

| Especificación | Pseudo-R² |
|---|---:|
| Inicial | 0,03488 |
| Con variabilidad | 0,03633 |
| Con persistencia y recuperación | 0,03637 |

La mejora incremental es pequeña y los términos de persistencia/recuperación no fueron significativos en esa especificación. Por tanto, la evidencia de asociación longitudinal no debe confundirse con una demostración de mejora predictiva independiente.

## 5. Matriz de trazabilidad para el manuscrito

| Afirmación o sección | Evidencia principal | Acción antes de redactar |
|---|---|---|
| Descripción de cohorte general | Auditoría 142 | Mantener 27.135 como cohorte analítica general, no como tamaño de la prueba predictiva estricta |
| Definición del objetivo (métodos 2.4) | Script 23 y auditoría 142 | Corregir: retiro administrativo posterior al día 34; no solo resultado final Withdrawn |
| Diseño de validación predictiva | Scripts 143–144 | Describir cinco cohortes independientes por estudiante |
| Resultados predictivos | Scripts 144, 150 y 151 | Dar prioridad a resultados estrictos y a la comparación estandarizada de bloques |
| Ponderación de matrículas | Scripts 145–147 | Resumir como análisis de sensibilidad; no colapsar matrículas de cursos distintos |
| Comparación de variables | Scripts 148–151 | Explicar la exclusión de identidad de módulo y limitación de categorías no vistas |
| Capacidad de alertas | Scripts 150–151 | No afirmar que el contexto logra capacidad 20/30/40%; informar su tasa real cercana a 87% |
| Variabilidad | Scripts 124–129 | Corregir pseudo-R² SD; definir CV y número de semanas observadas |
| Persistencia | Scripts 131–141 | Informar diferencia semanas 2–3 con Holm; evitar lenguaje causal/universal |
| Valor incremental | Script 132 | Reconocer mejora pequeña y ausencia de significación de persistencia/recuperación |
| Ecuaciones diferenciales | Ningún modelo diferencial identificado en scripts auditados | No afirmar implementación ni estimación |
| Referencias y figuras | Manuscrito | Auditoría editorial posterior: comprobar bibliografía final, referencias cruzadas y presencia real de figuras |

## 6. Limitaciones que deben declararse

1. OULAD procede de una institución y de un conjunto limitado de cursos y presentaciones; la generalización institucional requiere datos externos.
2. El objetivo es retiro administrativo por matrícula, no abandono global de la persona en la educación superior.
3. Hay matrículas múltiples por estudiante y dependencia potencial entre observaciones; la validación evita solapamiento de estudiantes entre cohortes y el bootstrap agrupa por estudiante, pero no convierte la unidad de resultado en persona.
4. La ausencia de una fila semanal no equivale necesariamente a inactividad; la disponibilidad de observaciones puede ser informativa.
5. El modelo combinado presenta discriminación moderada y proporciones de alerta elevadas; no es un sistema listo para despliegue.
6. El bootstrap descrito mantiene fijas las predicciones del modelo; no cuantifica toda la incertidumbre de entrenamiento y calibración.
7. Los errores agrupados por módulo-presentación se basan en 22 grupos.
8. Los análisis son observacionales; no identifican mecanismos causales ni estados psicológicos.
9. No se implementó un sistema de ecuaciones diferenciales ni un modelo matemático de transición crítica.

## 7. Estado y próximos pasos

**Cerrado a nivel de evidencia disponible:** reconstrucción de datos, definición operativa del objetivo, validación estricta por estudiante, sensibilidad a matrículas múltiples, comparación de bloques y bootstrap, análisis de variabilidad y análisis de transiciones.

**Pendiente antes de iniciar la redacción del artículo:**

- comprobar en los CSV locales que los resultados y tablas citados en este informe corresponden a las últimas ejecuciones;
- decidir las cifras exactas que se trasladarán a cada tabla del manuscrito;
- realizar una revisión editorial completa del documento vigente, incluida la bibliografía y las figuras;
- mantener explícito que el análisis es observacional y que no contiene ecuaciones diferenciales.

**Protección de archivos:** este informe se añade a `docs/` en la rama de auditoría. No modifica `main`, los datos, los resultados CSV locales ni el archivo DOCX del manuscrito.
