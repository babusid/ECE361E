import tensorflow as tf

MODELS = ['VGG11', 'VGG16', 'MobileNet']

for m in MODELS:
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(f'{m}/{m}_saved_model') # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open(f'{m}/{m}.tflite', 'wb') as f:
        f.write(tflite_model)
