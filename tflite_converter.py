import tensorflow as tf
from pathlib import Path
import os

'''
This modeule is used to convert a full keras model to a tflite model
'''

net_name = '1630867357_lstmRegul1e-3_DSmax_lstmUnits1024_convFilters1024_lre-7'
net_path = Path.cwd() / 'models' / net_name
server_network_path = '/home/pi/Documents/STA2-DEPLOY/networks'



# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(str(net_path)) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.

with open(Path.cwd() / 'tflite_models' / f'{net_name}.tflite', 'wb') as f:
  f.write(tflite_model)



file_to_copy = f'C:/Users/robin/Documents/STA2/tflite_models/{net_name}.tflite'
os.system(f'scp {file_to_copy} pi@niwood.ddns.net:{server_network_path}')

print('Done - Network transformed and copied to server')