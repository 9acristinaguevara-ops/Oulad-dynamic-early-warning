# OULAD Early Warning — proyecto reproducible

Proyecto de investigación: **Señales dinámicas tempranas de abandono estudiantil en entornos virtuales**.

Este ZIP es un esqueleto inicial, comentado y organizado para ejecutarse desde Anaconda Prompt. Se trabajará archivo por archivo; no contiene resultados científicos inventados ni afirma que la hipótesis de transición crítica esté demostrada.

## 1. Requisitos
- Windows + Anaconda/Miniconda.
- Conexión a internet para intentar descargar OULAD.
- Espacio libre suficiente: `studentVle.csv` es grande (más de 400 MB según el catálogo UCI); conviene reservar varios GB para extracción y archivos derivados.
- Python 3.11 (definido en `environment.yml`).

## 2. Crear el entorno
Abre Anaconda Prompt, entra en la carpeta donde descomprimiste el proyecto y ejecuta:

```bat
conda env create -f environment.yml
conda activate oulad-dynamics
```

Si el entorno ya existe y más adelante se modifica `environment.yml`, se puede actualizar con:
```bat
conda env update -f environment.yml --prune
```

## 3. Orden de ejecución
Ejecuta los scripts desde la carpeta raíz del proyecto, no desde `scripts`.

```bat
python scripts\01_download_oulad.py
python scripts\02_validate_data.py
python scripts\03_build_weekly_activity.py
```

El tercer script genera una tabla semanal preliminar de actividad y todavía no estima fatiga ni ajusta ecuaciones diferenciales. Esos análisis se añadirán después de inspeccionar los datos y fijar las definiciones científicas.

## 4. Descarga de OULAD
El primer script intenta localizar un enlace descargable desde la página oficial:
https://analyse.kmi.open.ac.uk/open_dataset

La página puede cambiar y el servidor puede impedir la descarga automática. Si el script no encuentra un enlace válido:
1. Abre la página oficial en el navegador.
2. Descarga el archivo comprimido del conjunto OULAD.
3. Copia el ZIP descargado a `data/raw/downloads/`.
4. Vuelve a ejecutar `python scripts\01_download_oulad.py --local-zip data/raw/downloads/NOMBRE_DEL_ARCHIVO.zip`

El script extraerá los CSV a `data/raw/csv/`. No renombres las columnas. Se esperan estos archivos:
- `assessments.csv`
- `courses.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`

## 5. Carpetas
- `data/raw/downloads/`: ZIP original descargado.
- `data/raw/csv/`: CSV originales extraídos, no editarlos manualmente.
- `data/interim/`: tablas intermedias.
- `data/processed/`: datos analíticos derivados.
- `outputs/figures/`: figuras.
- `outputs/tables/`: tablas.
- `outputs/logs/`: registros de ejecución.
- `docs/`: decisiones metodológicas y notas de investigación.
- `scripts/`: scripts numerados en orden de ejecución.

## 6. Nota metodológica esencial
OULAD contiene actividad observada en el entorno virtual y resultados académicos; no mide directamente la fatiga psicológica. Cualquier variable denominada `fatigue_proxy` deberá definirse como indicador operacional o estado latente, validarse y reportarse con cautela. No se debe describir como diagnóstico de fatiga.

El resultado final (`final_result`) no equivale exactamente a un retiro formal en todos los casos. Antes de definir la variable objetivo, se documentará cómo se tratarán `Withdrawn`, `Fail`, `Pass` y `Distinction`, y cómo se evita usar información futura al predecir.

## 7. Referencia principal del conjunto de datos
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171

Fuente oficial: https://analyse.kmi.open.ac.uk/open_dataset
El artículo debe citar también la licencia y las condiciones vigentes indicadas por el proveedor.

## 8. Qué hace cada script inicial
1. `01_download_oulad.py`: intenta descargar el ZIP desde la web oficial o acepta un ZIP descargado manualmente.
2. `02_validate_data.py`: comprueba que estén los siete CSV esperados, informa tamaños y columnas, y registra el resultado.
3. `03_build_weekly_activity.py`: agrega los clics de `studentVle.csv` por estudiante, módulo, presentación y semana; guarda una tabla intermedia.

Estos scripts son una base de trabajo, no un análisis completo ni una garantía de publicación Q1. La calidad científica dependerá de la pregunta, diseño, comparaciones, robustez, validación temporal y contribución frente a la literatura existente.
