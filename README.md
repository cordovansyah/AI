# RML

Tensorflow Summary

Use venv to avoid missing components during training and conversion (optional)
source ./venv/bin/activate


IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

Training Image to Tensorflow
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph_jagung.pb \
  --output_labels=tf_files/retrained_labels_jagung.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/foto_jagung


Converting Tensorflow Protobuffer to TFLite 
1. Python
IMAGE_SIZE=224
tflite_convert \
  --graph_def_file=tf_files/retrained_graph.pb \
  --output_file=tf_files/custom_folowers.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT

2. Bazel Build
bazel-bin/tensorflow/contrib/lite/toco/toco \
  - -input_format=TENSORFLOW_GRAPHDEF \
  --input_file=tf_files/restrained_graph.pb \
  --output_format=TFLITE \
  --output_file=tf_files/custom_flowers.tflite \
  --inference_type=QUANTIZED_UINT8 \
  --inference_input_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=final_result \
  --input_shapes=1,224,224,3\
  --mean_values=128 \
  --std_values=128 \
  --default_ranges_min=0 \
  --default_ranges_max=6


MUST-DO
- Test with Bazel Build


INTERPRETER & REFERENCES
https://medium.com/datadriveninvestor/how-to-deploy-a-tensorflow-lite-model-to-ios-4b230bb91ac0
https://github.com/bourdakos1/Core-ML-Custom-Object-Detection 
https://www.tensorflow.org/install/pip

- Installing Bazel : https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x
- 

