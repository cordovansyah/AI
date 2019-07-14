# Tensorflow Lite Script Summary for Object Detection (Based on personal experiment)

## TF Record Generator Flow (macOS)
1. Clone the [Tensorflow Research repository](https://github.com/tensorflow/models)<br>
2. As a sandbox test, build your VSCode project in the ```/research``` directory
3. Within your project, prepare three main folders, they are ```/annotations```, ```/images```, and ```/data```
4. Annotate your images with online annotation tool such as Lotus, labelimg and many more.
5. Inside ```/annotations``` go insert all .xml outputs from your annotation
6. Convert your .xml to unitary .csv file with code below. <b>Make sure</b> to store your .xml data in ```/annotations```. Oh by the way, this code should be built as a separated file outside the three main folders, aforementioned. You can name it as ```conversion.py``` (<b>PS: Don't forget to configure your Python and Tensorflow for VSCode if you use this editor</b>)
```
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('data_jagung.csv', index=None)
    print('Successfully converted xml to csv.')


main()
```
7. Once the conversion finished, you can see the .csv right away. Can't you?
8. Its time convert your .csv to .TFRecord file with this code below. Anyway, name it ```generator.py``` for short haha
```
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    if row_label == 'jagung':
        return 1
    else:
        return 0
       
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):    
    print(os.getcwd())
    writer = tf.python_io.TFRecordWriter('data/test.record')
    # path = os.path.join(FLAGS.image_dir)
    path = 'images'
    # examples = pd.read_csv(FLAGS.csv_input)
    examples = pd.read_csv('data_jagung.csv')
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
```
9. There you go, tfrecord is available on your ```data``` folder

### Other resources

## Training Image to Tensorflow
```IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph_jagung.pb \
  --output_labels=tf_files/retrained_labels_jagung.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/foto_jagung
```
## Converting Tensorflow Protobuffer to TFLite 
### 1. Python
```
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
```
### 2. Bazel Build
```
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
  ```

Additional Notes: Use venv to avoid missing components during training and conversion (optional)
```
source ./venv/bin/activate
```
