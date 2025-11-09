# IMDb Rating Category Classifier

**Clasificación automática de categoría de rating en películas IMDb**

Proyecto MLOps - Grupo 21

## 📋 Descripción

Este repositorio contiene el pipeline completo de datos y entrenamiento de modelos para clasificar películas de IMDb en categorías de rating (Poor, Average, Good, Excellent) basándose en sus características.

## 🎯 Objetivo

Predecir automáticamente la categoría de rating de una película basándose en sus características:
- **Poor**: Rating < 4
- **Average**: Rating 4-6
- **Good**: Rating 6-8  
- **Excellent**: Rating > 8

## 🏗️ Arquitectura del Proyecto

```
.
├── data/
│   ├── raw/              # Datos crudos de IMDb
│   ├── processed/        # Datos preprocesados
│   └── reviews/          # Reseñas (si aplica)
├── models/               # Modelos entrenados (.pkl)
├── mlruns/               # Experimentos MLflow
├── src/
│   ├── data/            # Scripts de descarga y preprocesamiento
│   ├── features/        # Feature engineering
│   └── models/          # Scripts de entrenamiento
├── dvc.yaml             # Pipeline DVC
├── params.yaml          # Parámetros de configuración
└── requirements.txt     # Dependencias Python
```

## 📊 Dataset

El proyecto utiliza dos fuentes de datos de IMDb:

### 1. Metadatos de IMDb (para features)

- **title.akas.tsv.gz**: Alternative titles for media
  - titleId (string) - a tconst, an alphanumeric unique identifier of the title
  - ordering (integer) – a number to uniquely identify rows for a given titleId
  - title (string) – the localized title
  - region (string) - the region for this version of the title
  - language (string) - the language of the title
  - types (array) - Enumerated set of attributes for this alternative title
  - attributes (array) - Additional terms to describe this alternative title
  - isOriginalTitle (boolean) – 0: not original title; 1: original title

- **title.basics.tsv.gz**: Basic information about titles
  - tconst (string) - alphanumeric unique identifier of the title
  - titleType (string) – the type/format of the title
  - primaryTitle (string) – the more popular title
  - originalTitle (string) - original title, in the original language
  - isAdult (boolean) - 0: non-adult title; 1: adult title
  - startYear (YYYY) – represents the release year of a title
  - endYear (YYYY) – TV Series end year
  - runtimeMinutes – primary runtime of the title, in minutes
  - genres (string array) – includes up to three genres associated with the title

- **title.crew.tsv.gz**: Directors and writers for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - directors (array of nconsts) - director(s) of the given title
  - writers (array of nconsts) – writer(s) of the given title

- **title.episode.tsv.gz**: TV episode information
  - tconst (string) - alphanumeric identifier of episode
  - parentTconst (string) - alphanumeric identifier of the parent TV Series
  - seasonNumber (integer) – season number the episode belongs to
  - episodeNumber (integer) – episode number of the tconst in the TV series

- **title.principals.tsv.gz**: Principal cast/crew for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - ordering (integer) – a number to uniquely identify rows for a given titleId
  - nconst (string) - alphanumeric unique identifier of the name/person
  - category (string) - the category of job that person was in
  - job (string) - the specific job title if applicable, else '\N'
  - characters (string) - the name of the character played if applicable, else '\N'

- **title.ratings.tsv.gz**: User ratings for titles
  - tconst (string) - alphanumeric unique identifier of the title
  - averageRating – weighted average of all the individual user ratings
  - numVotes - number of votes the title has received

- **name.basics.tsv.gz**: Information about individuals
  - nconst (string) - alphanumeric unique identifier of the name/person
  - primaryName (string)– name by which the person is most often credited
  - birthYear – in YYYY format
  - deathYear – in YYYY format if applicable, else '\N'
  - primaryProfession (array of strings)– the top-3 professions of the person
  - knownForTitles (array of tconsts) – titles the person is known for

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.12+
- Git
- DVC (Data Version Control)

### 1. Clonar el repositorio
```bash
git clone https://github.com/mbayonal/sentiment_classification_model.git
cd sentiment_classification_model
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline DVC
```bash
# Ejecutar todo el pipeline
dvc repro

# O ejecutar etapas específicas
dvc repro download_data      # Descargar datos de IMDb
dvc repro preprocess_data    # Preprocesar datos
dvc repro build_features     # Generar features
dvc repro train_rating_classifier  # Entrenar modelos
```

## 🎓 Modelos Entrenados

### Algoritmos Implementados
- **Logistic Regression** (multiclase): Mejor desempeño
- **Linear SVM** (multiclase)

### Resultados del Mejor Modelo
- **Modelo**: Logistic Regression
- **Accuracy**: 99.99%
- **F1 Score (weighted)**: 0.9999
- **Features utilizadas**: 
  - startYear
  - runtimeMinutes
  - numVotes
  - averageRating
  - runtime_category
  - popularity

### Artefactos Generados
- `models/best_model.pkl` - Modelo serializado listo para producción
- `models/best_model_metadata.json` - Métricas y metadata del modelo
- `mlruns/` - Experimentos completos registrados en MLflow

### Customizing Data Sampling

You can adjust the sampling parameters in the `params.yaml` file:

```yaml
# Target size in MB for each file (maximum size)
TARGET_SIZE_MB: 100

# Sampling ratios for each file
SAMPLING_RATIOS:
  title.akas.tsv.gz: 0.05      # 5% of original
  title.basics.tsv.gz: 0.1     # 10% of original
  # ... other files
```

Increasing the sampling ratios will include more data but result in larger file sizes.

## 📈 MLflow Tracking

Todos los experimentos están registrados en MLflow:

```bash
# Ver experimentos en la UI de MLflow
mlflow ui

# Acceder a: http://localhost:5000
```

## 📝 Configuración (params.yaml)

El archivo `params.yaml` contiene todos los parámetros configurables:

```yaml
rating_classifier:
  test_size: 0.2
  random_state: 42
  
  logistic_regression:
    C: 1.0
    max_iter: 1000
  
  linear_svm:
    C: 1.0
    max_iter: 2000
```
## Troubleshooting y buenas prácticas de ejecución

### Errores frecuentes con DVC

- **Error de caché no encontrada (`cache missing`)**  
  - Ejecuta: `dvc pull` para traer los artefactos desde el remoto configurado.  
  - Si el remoto no está configurado, revisa la sección de `remote` en `dvc.yaml` y valida que las credenciales existan.

- **Cambios en `params.yaml` que no se reflejan en el entrenamiento**  
  - Asegúrate de correr `dvc repro` completo o al menos las etapas que dependen de esos parámetros.  
  - Usa `dvc dag` para visualizar el grafo de dependencias y entender qué etapas deben ejecutarse.

- **Problemas de espacio en disco con los datos de IMDb**  
  - Ajusta las tasas de muestreo en `params.yaml` (sección `SAMPLING_RATIOS`) para reducir el tamaño de los archivos.  
  - Limpia caché antigua con `dvc gc` (después de validar que no perderás versiones importantes).

### Recomendaciones para MLflow

- Levantar la UI localmente:

  ```bash
  mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns


## 👥 Equipo - Grupo 21

- **Luis Felipe González** - Data Manager/MLOps (DVC/versionado)
- **Daniel Ricardo Marín** - Data Scientist (calidad/limpieza)
- **Manuel Alejandro Bayona** - Cloud Engineer (S3, backups)
- **Fabián Jiménez** - BI Analyst (visualización/dashboard)

## 📄 Licencia

Este proyecto es parte del curso de MLOps - MIAD Universidad de los Andes.

## 🔗 Repositorios Relacionados

- [API REST](https://github.com/mbayonal/api_imdb) - Servicio de predicción con FastAPI
- [Dashboard](https://github.com/mbayonal/dashboard_imdb) - Interfaz web con Streamlit
