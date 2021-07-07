import tensorflow as tf
from pathlib import Path


net_name = '1624913277'
net_path = Path.cwd() / 'models' / net_name


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(str(net_path)) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.

with open(Path.cwd() / 'tflite_models' / f'{net_name}.tflite', 'wb') as f:
  f.write(tflite_model)

print('Done')