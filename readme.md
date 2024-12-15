# nnUNet-em

## Introducción

En este repositorio se encuentra una modificación del código de [nnUNet](https://github.com/MIC-DKFZ/nnUNet), estas modificaciones permiten la segmentación de imágenes de resonancia magnética de cerebro en 3D para identificar lesiones de esclerosis múltiple, especificamente en el dataset [Hospital Clínic de Barcelona](https://www.clinicbarcelona.org/).
Este dataset contiene imágenes de resonancia magnética de aproximadamente 100 pacientes las cuales han sido segmentadas por expertos en el área.


## Utilización del modelo

### Usando la imagen de Docker

Para utilizar el modelo se puede hacer uso de la imagen de Docker que se encuentra en el proyecto, para ello se debe ejecutar el siguiente comando:

```bash docker compose build```

Si quieres que la imagen se vuelva a construir sin usar la caché, puedes ejecutar el siguiente comando:

```bash docker compose build --no-cache```

Una vez que la imagen se haya construido, se puede ejecutar el siguiente comando para correr el contenedor:

```bash docker compose up```

Hay que tener en cuenta los siguientes puntos para la ejecución del contenedor:

- Los datos de entrada deben estar en la carpeta `input` en la raíz del proyecto, estos datos tienen que seguir un patrón de nombres específico para que el modelo pueda procesarlos. El patrón de nombres es el siguiente: `{ID}_{TIMEPOINT}.nii.gz`, donde ID es el identificador del paciente y TIMEPOINT es el momento en el que se tomó la imagen.

- Los datos de salida se guardarán en la carpeta `output` en la raíz del proyecto, estos datos tendrán el mismo nombre que los datos de entrada pero siguendo el siguiente patrón: `{ID}_{TIMEPOINT}.nii.gz`.


