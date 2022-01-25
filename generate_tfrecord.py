"""data preprocessing"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import ast

from PIL import Image
from object_detection.utils import dataset_util


def create_tf_example(row):
    filename = row.image_path.encode('utf8')

    with tf.io.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    if row.annotations == []:
        classes.append(0)

    for annot in row.annotations:
        xmins.append(annot['x'] / width)
        xmaxs.append((annot['x'] + annot['width']) / width)
        ymins.append(annot['y'] / height)
        ymaxs.append((annot['y'] + annot['height']) / height)
        classes_text.append('COTS'.encode('utf8'))
        classes.append(1)

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


def main():
    cwl = '/content/tensorflow-great-barrier-reef/workspace/annotations'
    data_df = pd.read_csv(cwl + '/train.csv')
    data_df['image_path'] = '/content/tensorflow-great-barrier-reef/train_images/video_' + \
                            data_df['video_id'].astype(str) + '/' + data_df[
        'video_frame'].astype(str) + '.jpg'
    data_df["annotations"] = data_df["annotations"].apply(ast.literal_eval)

    training_ratio = 0.8
    split_index = int(training_ratio * len(data_df))
    while data_df.iloc[split_index - 1].sequence == data_df.iloc[split_index].sequence:
        split_index += 1
    train_df = data_df.iloc[:split_index].sample(frac=1).reset_index(drop=True)
    test_df = data_df.iloc[split_index:].sample(frac=1).reset_index(drop=True)
    print(train_df.head())
    print(test_df.head())
    train_positive_count = len(train_df[train_df.annotations.map(tuple) != tuple([])])
    test_positive_count = len(test_df[test_df.annotations.map(tuple) != tuple([])])

    print('Training ratio (all samples):',
          float(len(train_df)) / (len(train_df) + len(test_df)))
    print('Training ratio (positive samples):',
          float(train_positive_count) / (train_positive_count + test_positive_count))

    train_df = train_df[train_df.annotations.map(tuple) != tuple([])].reset_index()
    print('Number of positive images used for training:', len(train_df))
    test_df = test_df[test_df.annotations.map(tuple) != tuple([])].reset_index()
    print('Number of positive images used for validation:', len(test_df))

    output_path = cwl + '/train.record'
    writer = tf.io.TFRecordWriter(output_path)
    for index, row in train_df.iterrows():
        tf_example = create_tf_example(row)
        if index % 500 == 0:
            print('Processed {0} train images.'.format(index))
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

    output_path = cwl + '/test.record'
    writer = tf.io.TFRecordWriter(output_path)
    for index, row in test_df.iterrows():
        tf_example = create_tf_example(row)
        if index % 500 == 0:
            print('Processed {0} test images.'.format(index))
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    main()
