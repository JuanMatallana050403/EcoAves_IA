# IA propia para reconocimiento de aves

## 1. Respuesta corta

Sí, es posible construir una IA propia que reconozca aves en imágenes y
sonidos. Sin embargo, no conviene intentar crear un modelo grande desde cero.
La estrategia realista es:

1. Recolectar datos propios de Perú y San Martín.
2. Etiquetarlos correctamente.
3. Ajustar (`fine-tuning`) un modelo preentrenado.
4. Evaluarlo con datos que el modelo nunca haya visto.
5. Publicarlo como un servicio de inferencia o convertirlo para ejecutarlo en
   el teléfono.

La IA propia no reemplazaría necesariamente a OpenAI desde el primer día. Se
puede comenzar con OpenAI como línea base y, cuando exista suficiente dataset,
comparar contra el modelo propio.

> Entrenar un modelo fundacional del tamaño de GPT o crear una IA general
> desde cero no es necesario para este proyecto. El objetivo debe ser un
> clasificador especializado de aves.

## 2. Qué existe actualmente

La app actual hace lo siguiente:

```text
App móvil
   ├── captura imagen/audio
   ├── envía Base64 a la Edge Function proxy-openai
   ├── la función consulta OpenAI
   └── guarda la predicción en el registro
```

Cloudinary almacena las evidencias y Supabase almacena los metadatos.
Actualmente, el nuevo Cloudinary no contiene evidencias de EcoAves; solo tiene
archivos de ejemplo. Por tanto, todavía hay que construir el dataset propio.

OpenAI puede ayudar a generar predicciones iniciales, pero una predicción de
OpenAI no debe considerarse automáticamente una etiqueta verdadera. Las
etiquetas finales deben ser revisadas por una persona con conocimiento
ornitológico.

## 3. Qué significa "IA propia"

Hay tres niveles posibles:

### Nivel A: modelo propio en un servidor

Se entrena un modelo especializado y se publica una API:

```text
App → Cloudinary → API propia de IA → predicción → Supabase
```

Es la opción recomendada para la primera versión propia.

### Nivel B: modelo propio en el teléfono

El modelo se convierte a TensorFlow Lite, ONNX, Core ML o un formato similar:

```text
App → modelo local → predicción offline
```

Permite trabajar sin internet, pero exige reducir el tamaño y consumo del
modelo. Es una segunda etapa.

### Nivel C: modelo entrenado completamente desde cero

Se diseña la arquitectura y se entrenan todos los pesos con un dataset enorme.
No es recomendable para este proyecto: requiere muchos datos, GPU, tiempo y
mantenimiento, y probablemente tendría peor resultado que el ajuste de un
modelo existente.

## 4. Reconocimiento de imágenes

### Tareas posibles

- Clasificar una foto completa: qué especie aparece.
- Detectar el ave dentro de la foto: dónde está el ave.
- Reconocer varias aves en una misma imagen.
- Rechazar fotos que no contienen aves.
- Devolver especie, confianza y categoría "desconocida".

### Modelos recomendados

| Objetivo | Alternativas |
|---|---|
| Clasificación | EfficientNet, ConvNeXt, ResNet, MobileNet, ViT |
| Detección | YOLO, RT-DETR, Detectron2 |
| Modelo liviano móvil | MobileNet, EfficientNet-Lite, YOLO-Nano |
| Embeddings | DINOv2, CLIP u otro modelo visual preentrenado |

Para un MVP se recomienda **transfer learning con EfficientNet o MobileNet**.
Si las fotografías tienen fondos complejos o el ave ocupa una parte pequeña,
se debe añadir detección con YOLO antes de clasificar.

## 5. Reconocimiento de audio

El audio requiere un flujo diferente a la imagen:

```text
Audio m4a
   → conversión a WAV/PCM
   → eliminación o reducción de ruido opcional
   → espectrograma Mel
   → modelo acústico
   → especie y confianza
```

### Alternativas

- BirdNET como línea base especializada.
- Perch, si sus condiciones de uso son compatibles.
- PANNs o YAMNet como extractor de características.
- CNN sobre espectrogramas Mel.
- Fine-tuning de un modelo bioacústico preentrenado.

Para el MVP de audio conviene usar un modelo preentrenado como extractor y
entrenar un clasificador propio para las especies de la región. Es más
realista que entrenar todo el modelo acústico desde cero.

## 6. Dataset necesario

### Imágenes

Como orientación para un MVP:

- 10 a 20 especies objetivo.
- 100 a 300 imágenes etiquetadas por especie para una primera prueba.
- 500 o más por especie para un modelo más robusto.
- Diferentes distancias, ángulos, iluminación, fondos y estaciones.
- Fotografías con y sin el ave.
- Una clase `unknown` o `other` para evitar que el modelo siempre fuerce una
  especie conocida.

Estas cantidades no son una garantía de precisión. La diversidad y calidad de
las etiquetas son más importantes que acumular muchas imágenes repetidas.

### Audios

Como punto de partida:

- 10 a 20 especies objetivo.
- 50 a 150 grabaciones por especie para una prueba inicial.
- Grabaciones de distintos individuos, lugares, distancias y niveles de ruido.
- Segmentos positivos donde realmente se escuche el ave.
- Segmentos negativos con lluvia, insectos, viento, tráfico y silencio.

En audio es preferible tener muchas grabaciones independientes de individuos
diferentes, no cortar una sola grabación larga en cientos de fragmentos y
usarlos como si fueran datos independientes.

### Fuentes posibles

- Registros propios de la aplicación.
- Xeno-canto, revisando su licencia y condiciones de uso.
- Macaulay Library/eBird, respetando sus licencias y restricciones.
- iNaturalist, según la licencia de cada observación.
- Datasets públicos de BirdCLEF u otros proyectos bioacústicos.

Cada archivo debe conservar la fuente, licencia, autor, fecha y ubicación. No
se debe mezclar material con licencias incompatibles en un modelo que luego se
quiera publicar.

## 7. Etiquetado

### Imagen

Para cada imagen se recomienda guardar:

```text
species_id
common_name
scientific_name
is_bird
bounding_box (opcional)
source
license
location
recorded_at
annotator
annotation_status
```

### Audio

Para cada audio:

```text
species_id
start_time
end_time
is_bird_call
call_type (opcional)
background_noise
source
license
location
recorded_at
annotator
annotation_status
```

Herramientas posibles:

- CVAT o Label Studio para imágenes.
- Audacity para revisar y cortar audios.
- Sonic Visualiser para análisis acústico.
- Roboflow, si sus condiciones de uso son aceptables.

## 8. División correcta del dataset

No se deben dividir los datos aleatoriamente si existen fotografías o audios
del mismo individuo, lugar o grabación en entrenamiento y validación. Eso
produce resultados artificialmente altos.

La división recomendada es:

```text
70 % entrenamiento
15 % validación
15 % prueba
```

Además, la prueba debe separar por lo menos alguno de estos factores:

- Lugar.
- Fecha.
- Recolector.
- Individuo.
- Grabación original.

Una prueba importante sería dejar una zona geográfica completa fuera del
entrenamiento para medir si el modelo generaliza a otra zona de Perú.

## 9. Métricas que deben reportarse

No basta con mostrar un porcentaje de acierto.

### Imagen

- Accuracy.
- Precision, recall y F1 por especie.
- Macro-F1 para evitar que las especies frecuentes oculten a las minoritarias.
- Top-1 y Top-3 accuracy.
- Matriz de confusión.
- Tasa de rechazo de fotos sin aves.
- Calibración de confianza.

### Audio

- F1 por especie.
- Macro-F1.
- Recall de cantos reales.
- Falsos positivos por hora de audio.
- Matriz de confusión.
- Rendimiento con ruido y grabaciones de lugares nuevos.

En la tesis se deben comparar al menos:

```text
Modelo OpenAI
Modelo propio
Modelo propio + revisión humana
```

No se debe afirmar que el modelo propio es mejor sin realizar esta
comparación sobre el mismo conjunto de prueba.

## 10. Arquitectura recomendada

### Primera etapa: inferencia en servidor

```text
[App móvil]
     │
     ├── sube evidencia a Cloudinary
     │
     ├── envía record_id/file_url a la API propia
     ▼
[Servicio de IA]
     ├── descarga o recibe la evidencia
     ├── ejecuta modelo de imagen/audio
     ├── devuelve especie, confianza y versión
     ▼
[Supabase records]
```

La Edge Function de Supabase puede validar al usuario y coordinar la llamada,
pero no es el lugar ideal para ejecutar un modelo pesado. Para inferencia se
puede utilizar:

- FastAPI en un servidor con GPU.
- Cloud Run.
- Modal.
- RunPod.
- AWS/GCP/Azure.
- Un servidor propio con Docker.

### Segunda etapa: inferencia offline

Cuando el modelo sea pequeño y estable:

```text
[App móvil] → modelo TFLite/ONNX local → resultado offline
```

La app puede sincronizar después la predicción y la evidencia con Cloudinary y
Supabase.

## 11. Información que debe guardar cada predicción

Los campos actuales de `records` no identifican claramente qué modelo produjo
la predicción. Se recomienda añadir:

```sql
ALTER TABLE records ADD COLUMN IF NOT EXISTS ai_provider TEXT;
ALTER TABLE records ADD COLUMN IF NOT EXISTS ai_model TEXT;
ALTER TABLE records ADD COLUMN IF NOT EXISTS ai_model_version TEXT;
ALTER TABLE records ADD COLUMN IF NOT EXISTS ai_source TEXT;
ALTER TABLE records ADD COLUMN IF NOT EXISTS ai_raw_result JSONB;
```

Valores de ejemplo:

```text
ai_provider      = own_model
ai_model         = ecoaves-image-classifier
ai_model_version = 2026.1
ai_source        = server
```

Para un historial completo conviene crear una tabla separada:

```sql
CREATE TABLE ai_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  evidence_type TEXT NOT NULL,
  predicted_species TEXT,
  confidence REAL,
  raw_result JSONB,
  reviewed_by TEXT,
  review_status TEXT DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Así se pueden comparar OpenAI, el modelo propio y futuras versiones sin perder
las predicciones anteriores.

## 12. Cómo integrar el modelo en la app

La app no debería tener lógica mezclada entre OpenAI y el modelo propio. Se
recomienda definir un proveedor de IA:

```ts
interface BirdAIProvider {
  identifyImage(uri: string): Promise<BirdIdentificationResult>;
  identifyAudio(uri: string): Promise<BirdIdentificationResult>;
}
```

Después se pueden implementar:

```text
OpenAIProvider       → Edge Function proxy-openai
OwnModelProvider     → API propia de EcoAves
OnDeviceProvider     → modelo TFLite/ONNX local
```

La pantalla de captura solo debe conocer el proveedor activo, no los detalles
internos de cada servicio. Esto permite comparar modelos sin reescribir toda la
app.

## 13. Plan de trabajo recomendado

### Fase 1 — Línea base

- Mantener OpenAI funcionando.
- Guardar `provider`, modelo y confianza.
- Guardar las evidencias originales.
- No usar las respuestas de OpenAI como verdad definitiva.

### Fase 2 — Dataset propio

- Elegir 10 especies frecuentes o prioritarias.
- Definir taxonomía y nombres válidos.
- Recolectar imágenes y audios en campo.
- Añadir etiquetas revisadas por un experto.
- Registrar ubicación, fecha, recolector y fuente.

### Fase 3 — Primer modelo de imágenes

- Entrenar transfer learning con MobileNet/EfficientNet.
- Incluir clase `unknown`.
- Evaluar por zona y fecha no vistas.
- Guardar matriz de confusión y errores.

### Fase 4 — Primer modelo de audio

- Convertir audios a segmentos y espectrogramas.
- Usar BirdNET/PANNs/YAMNet como línea base.
- Entrenar el clasificador regional.
- Evaluar ruido y grabaciones nuevas.

### Fase 5 — API propia

- Crear un servicio FastAPI o equivalente.
- Cargar el modelo una sola vez al iniciar el servidor.
- Validar tamaño, tipo y autenticación.
- Devolver predicción, confianza y versión del modelo.

### Fase 6 — Integración

- Añadir `OwnModelProvider` a la app.
- Ejecutar OpenAI y modelo propio en modo comparación durante las pruebas.
- Permitir revisión humana.
- Guardar el resultado final validado en Supabase.

### Fase 7 — Offline

- Convertir el modelo a TFLite/ONNX.
- Medir tamaño, memoria, batería y latencia.
- Ejecutar una versión pequeña en el dispositivo.
- Sincronizar predicciones cuando vuelva internet.

## 14. Recursos necesarios

### Personas

- Una persona de desarrollo/ML.
- Un experto o asesor que valide las especies.
- Recolectores de campo.

### Software

- Python.
- PyTorch o TensorFlow.
- Librosa/torchaudio para audio.
- OpenCV/Pillow para imágenes.
- CVAT o Label Studio para etiquetas.
- FastAPI para el servicio.
- Docker para despliegue.

### Hardware

Para un prototipo:

- GPU con 8 a 16 GB de VRAM, local o alquilada.
- 16 a 32 GB de RAM.
- Almacenamiento para originales, versiones procesadas y modelos.

El entrenamiento puede hacerse en la nube; el teléfono no debe entrenar el
modelo.

## 15. Costos aproximados

Dependen del tamaño del dataset y del proveedor:

- Etiquetado: costo de tiempo humano o de un servicio externo.
- Entrenamiento inicial: horas de GPU.
- Inferencia: costo por segundo/minuto si se usa GPU en servidor.
- Almacenamiento: Cloudinary para archivos y Supabase para metadata.
- Mantenimiento: nuevas especies, correcciones y nuevas versiones.

El modelo propio puede reducir el costo de llamadas a OpenAI, pero no elimina
los costos de almacenamiento, servidor, etiquetado y mantenimiento.

## 16. Riesgos importantes

- Confundir especies visualmente parecidas.
- Sesgo hacia las especies con más datos.
- Datos repetidos del mismo lugar o individuo.
- Confundir ruido con un canto.
- Confianza alta en una predicción incorrecta.
- Pérdida de generalización fuera de San Martín.
- Problemas de licencia de fotografías o audios.
- Exposición de coordenadas de especies sensibles.
- Guardar claves privadas dentro del APK.

La app debe mostrar "no identificado" o "requiere revisión" cuando la
confianza sea baja. Nunca debe forzar una especie solo porque es la más
probable.

## 17. Qué sería novedoso para la tesis

Usar una API de OpenAI sin más no constituye por sí solo una IA novedosa. La
contribución puede estar en:

1. Dataset regional de aves del Perú con georreferenciación.
2. Modelo especializado para especies de San Martín.
3. Reconocimiento combinado visual y acústico.
4. Funcionamiento offline-first para trabajo de campo.
5. Sincronización tolerante a pérdida de conectividad.
6. Validación humana y trazabilidad de modelos.
7. Comparación experimental entre OpenAI y un modelo propio.
8. Evaluación por zonas geográficas no vistas durante el entrenamiento.

Una formulación sólida sería:

> "Diseño y evaluación de un sistema móvil offline-first para el
> reconocimiento visual y acústico de aves de San Martín, utilizando modelos
> especializados entrenados con datos regionales y comparados con un servicio
> multimodal comercial."

## 18. Recomendación final

No empezar entrenando una IA gigante. El camino recomendado es:

```text
OpenAI como línea base
        ↓
Dataset propio revisado
        ↓
Modelo de imágenes regional
        ↓
Modelo de audio regional
        ↓
API propia
        ↓
Modelo offline en el teléfono
```

El primer objetivo medible debe ser un modelo propio para 10 especies, con
imágenes y audios revisados, una prueba separada por zona y una comparación
objetiva contra OpenAI. Eso es alcanzable, defendible académicamente y puede
crecer después hacia más especies.
