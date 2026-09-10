# Inventario de imágenes EcoAves

El script `scripts/gbif_inventory.py` usa la API de GBIF, no scraping de
resultados visuales. Conserva una respuesta JSON por especie y crea un CSV con
la URL, licencia, autoría, ubicación, fecha, hash y estado de cada imagen.

## Instalación

```powershell
python -m pip install -r requirements-dataset.txt
```

## Primero: solo metadatos

```powershell
python scripts/gbif_inventory.py --max-per-species 100
```

## Descargar imágenes con licencia redistribuible

El script acepta por defecto CC0, CC BY y CC BY-SA. No descarga material con
licencia ausente, desconocida o no compatible:

```powershell
python scripts/gbif_inventory.py --max-per-species 300 --download
```

## Usar imágenes CC BY-NC solo para investigación

Este modo descarga `CC BY-NC` y las marca como `research_only`. No descarga
`CC BY-NC-ND`, porque esa licencia no permite obras derivadas. Antes de usar
estas imágenes en una tesis, publicación o modelo distribuido, se deben revisar
las condiciones de la licencia y conservar la atribución de cada autor.

```powershell
python scripts/gbif_inventory.py --max-per-species 300 --download --allow-noncommercial
```

Las imágenes quedarán identificadas por `license_scope=research_only` y
`status=downloaded_research_only`. No deben mezclarse con material que se vaya
a publicar o redistribuir sin una revisión legal y de licencia.

## Revisión manual obligatoria

En `dataset/metadata/image_inventory.csv` revisar `manual_review`, `license`,
`creator` y `status`. Cambiar `manual_review` a `approved` solo después de
confirmar que la especie es correcta. No usar registros `rejected_*` ni
`duplicate_*` para entrenar.

## Separar por lugar y fecha

Después de aprobar las imágenes, ejecutar:

```powershell
python scripts/prepare_splits.py
```

El script crea `dataset/processed/splits/train.csv`,
`validation.csv` y `test.csv`. Agrupa por coordenadas redondeadas a dos
decimales y por mes de registro, y nunca separa un mismo grupo entre dos
conjuntos. Para resultados definitivos, conviene reservar una zona geográfica
completa para prueba.

## Primera prueba con MobileNet

Instalar las dependencias del entrenamiento:

```powershell
python -m pip install -r requirements-ml.txt
```

Ejecutar la prueba de transfer learning:

```powershell
python scripts/train_mobilenet.py --epochs 8 --batch-size 8
```

El modelo se guarda en `models/mobilenet_v3_small.pt`. Esta primera ejecución
solo entrena la cabeza clasificadora con la red visual congelada. Si una
división no contiene todas las especies, el script lo informa; sus métricas no
deben interpretarse como una evaluación científica hasta ampliar el dataset.

Para medir el modelo usando únicamente `test.csv`:

```powershell
python scripts/evaluate_mobilenet.py `
	--model models/mobilenet_with_unknown.pt
```

El comando muestra accuracy, macro-F1, precision, recall y F1 por clase. También
guarda `models/evaluation/classification_report.json` y
`models/evaluation/confusion_matrix.csv`. Si faltan especies en test, aparece
una advertencia y la evaluación debe considerarse solamente una prueba técnica.

## Precauciones

- Conservar licencia y atribución junto con cada archivo.
- No publicar coordenadas exactas de especies sensibles sin autorización.
- Los registros de GBIF son candidatos, no etiquetas ornitológicas definitivas.
- La eliminación automática de casi duplicados requiere confirmación humana.