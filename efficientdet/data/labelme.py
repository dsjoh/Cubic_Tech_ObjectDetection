"""
Utils module to work with labelme annotations using rectangles

Read more about labelme here:
    https://github.com/wkentaro/labelme
"""

import json
import random
from pathlib import Path
from functools import partial
from typing import Mapping, Tuple, Sequence, Union, Iterator, Iterable

import tensorflow as tf

import efficientdet.utils.io as io_utils
import efficientdet.utils.bndbox as bb_utils
from efficientdet.typing import Annotation, ObjectDetectionInstance


def _load_bbox_from_rectangle(
        points: Iterable[Sequence[float]]) -> Sequence[float]:
    return sum(points, [])


def _load_bbox_from_polygon(
        points: Iterable[Sequence[float]]) -> Sequence[float]:
    xs, ys = zip(*points)
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    return [xmin, ymin, xmax, ymax]


def _load_labelme_instance(
        images_base_path: Union[str, Path],
        annot_path: Union[str, Path],
        im_input_size: Tuple[int, int],
        class2idx: Mapping[str, int]) -> ObjectDetectionInstance:
    # Reads a labelme annotation and returns
    # a list of tuples containing the ground 
    # truth boxes and its respective label

    with Path(annot_path).open() as f:
        annot = json.load(f)

    bbs = []
    labels = []

    image_path = Path(images_base_path) / annot['imagePath']
    image = io_utils.load_image(str(image_path), 
                                im_input_size, 
                                normalize_image=True)
    
    w, h = annot['imageWidth'], annot['imageHeight']

    for shape in annot['shapes']:
        shape_type = shape['shape_type']
        if shape_type == 'rectangle':
            points = _load_bbox_from_rectangle(shape['points'])
        elif shape_type == 'polygon':
            points = _load_bbox_from_polygon(shape['points'])
        else:
            raise ValueError(
                f'Unexpected shape type: {shape_type} in file {annot_path}')
                
        label = shape['label']
        bbs.append(points)
        labels.append(class2idx[label])

    boxes = tf.stack(bbs)
    boxes = bb_utils.normalize_bndboxes(boxes, (h, w))

    return image, (tf.stack(labels), boxes)


def _labelme_gen(
        images_base_path: Union[str, Path],
        annot_files: Sequence[Path],
        im_input_size: Tuple[int, int],
        class2idx: Mapping[str, int]) -> Iterator[ObjectDetectionInstance]:

    for f in annot_files:
        yield _load_labelme_instance(images_base_path=images_base_path,
                                     annot_path=f,
                                     im_input_size=im_input_size,
                                     class2idx=class2idx)


def _scale_boxes(image: tf.Tensor, 
                 annots: Annotation,
                 to_size: Tuple[int, int]) -> ObjectDetectionInstance:
    # Wrapper function to call scale_boxes on tf dataset pipeline
    h, w = to_size[0], to_size[1]
    labels = annots[0]
    boxes = annots[1]

    x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h
    
    return image, (labels, tf.concat([x1, y1, x2, y2], axis=1))


def build_dataset(annotations_path: Union[str, Path],
                  images_path: Union[str, Path],
                  class2idx: Mapping[str, int],
                  im_input_size: Tuple[int, int],
                  shuffle: bool = True) -> tf.data.Dataset:
    """
    Create model input pipeline using tensorflow datasets

    Parameters
    ----------
    annotation_path: Union[Path, str]
        Path to the labelme dataset. The dataset path should contain both
        the annotations and the images
    class2idx: Mapping[str, int]
        Mapping to convert string labels to numbers
    im_input_size: Tuple[int, int]
        Model input size. Images will automatically be resized to this
        shape
    shuffle: bool, default True
        Shuffle the dataset or not
    
    Examples
    --------
    >>> from efficientdet.data.labelme import build_dataset
    >>> ds = build_dataset('data/labelme-ds/', 
    ...                    'data/labelme-ds/images', 
    ...                    class2idx=dict(class_name=0),
    ...                    im_input_size=(128, 128))
    >>> for images, (labels, bbs) in ds.take(1):
    ...   print(images.shape)
    ...   print(labels, bbs.shape)
    ...
    (2, 128, 128)
    ([[1, 0]
      [13, -1]], (2, 2, 4))

    Returns
    -------
    tf.data.Dataset

    """
    annotations_path = Path(annotations_path)
    images_path = Path(images_path)

    # List sorted annotation files
    annot_files = sorted(annotations_path.glob('*.json'))
    if shuffle:
        random.shuffle(annot_files)

    # Scale the boxes with the same shape
    scale_boxes = partial(_scale_boxes, to_size=im_input_size)

    generator = lambda: _labelme_gen(images_path, 
                                     annot_files, 
                                     im_input_size, 
                                     class2idx)
    output_shapes = (tf.TensorShape([*im_input_size, 3]), 
                     (tf.TensorShape([None]),
                      tf.TensorShape([None, 4])))

    ds = (tf.data.Dataset
          .from_generator(generator=generator, 
                          output_types=(tf.float32, (tf.int32, tf.float32)),
                          output_shapes=output_shapes)
          .map(scale_boxes))

    return ds
