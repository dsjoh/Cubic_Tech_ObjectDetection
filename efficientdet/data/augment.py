import abc
from typing import Tuple

import tensorflow as tf

import efficientdet.utils.bndbox as bb_utils
from efficientdet.typing import Annotation, ObjectDetectionInstance


@tf.function
def horizontal_flip(image: tf.Tensor, 
                    annots: Annotation) -> ObjectDetectionInstance:
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Flip the image
    image = tf.image.flip_left_right(image)
    
    # Flip the box
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)

    bb_w = x2 - x1
    delta_W = tf.expand_dims(boxes[:, 0], axis=-1)

    x1 = tf.cast(w, tf.float32) - delta_W - bb_w
    x2 = tf.cast(w, tf.float32) - delta_W

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    return image, (labels, boxes)

@tf.function
def vertical_flip(image: tf.Tensor, 
                  annots: Annotation) -> ObjectDetectionInstance:
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Flip the image
    image = tf.image.flip_up_down(image)
    
    # Flip the box
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)

    bb_h = y2 - y1
    delta_H = tf.expand_dims(boxes[:, 1], axis=-1)

    y1 = tf.cast(h, tf.float32) - delta_H - bb_h
    y2 = tf.cast(h, tf.float32) - delta_H

    boxes = tf.stack([y1, x1, y2, x2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])

    return image, (labels, boxes)

@tf.function
def crop(image: tf.Tensor, 
         annots: Annotation) -> ObjectDetectionInstance:
    
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Get random crop dims
    crop_factor_w = tf.random.uniform(shape=[], minval=.4, maxval=1.)
    crop_factor_h = tf.random.uniform(shape=[], minval=.4, maxval=1.)

    crop_width = tf.cast(tf.cast(w, tf.float32) * crop_factor_w, tf.int32)
    crop_height = tf.cast(tf.cast(h, tf.float32) * crop_factor_h, tf.int32)

    # Pick coordinates to start the crop
    x = tf.random.uniform(shape=[], maxval=w - crop_width - 1, dtype=tf.int32)
    y = tf.random.uniform(shape=[], maxval=h - crop_height - 1, dtype=tf.int32)

    # Crop the image and resize it back to original size
    crop_im = tf.image.crop_to_bounding_box(
        image, y, x, crop_height, crop_width)
    crop_im = tf.image.resize(crop_im, (h, w))
    
    # Clip the boxes to fit inside the crop
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=-1)
    
    # Cast crop coordinates to float, so they can be used for clipping
    x = tf.cast(x, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    y = tf.cast(y, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)
    
    # Adjust boxes coordinates after the crop
    widths = x2 - x1
    heights = y2 - y1

    x1 = x1 - x
    y1 = y1 - y
    x2 = x1 + widths
    y2 = y1 + heights

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])
    boxes = bb_utils.clip_boxes(tf.expand_dims(boxes, 0), 
                                (crop_height, crop_width))
    boxes = tf.reshape(boxes, [-1, 4])

    # Create a mask to avoid tiny boxes
    widths = tf.gather(boxes, 2, axis=-1) - tf.gather(boxes, 0, axis=-1)
    heights = tf.gather(boxes, 3, axis=-1) - tf.gather(boxes, 1, axis=-1)
    areas = widths * heights
    
    # Min area is the 1 per cent of the whole area
    min_area = .01 * (crop_height * crop_width)
    large_areas = tf.reshape(tf.greater_equal(areas, min_area), [-1])

    # Get only large enough boxes
    boxes = tf.boolean_mask(boxes, large_areas, axis=0)
    labels = tf.boolean_mask(labels, large_areas)

    # Scale the boxes to original image
    boxes = bb_utils.scale_boxes(
        boxes, image.shape[:-1], (crop_height, crop_width))

    return crop_im, (labels, boxes)

@tf.function
def erase(image: tf.Tensor, 
          annots: Annotation,
          patch_aspect_ratio: Tuple[float, float] = (.2, .2)) \
              -> ObjectDetectionInstance:

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]
    h = tf.cast(h, tf.float32)
    w = tf.cast(w, tf.float32)
    
    # Generate patch
    h_prop = tf.random.uniform(shape=[], 
                               minval=0, 
                               maxval=patch_aspect_ratio[0])
    w_prop = tf.random.uniform(shape=[], 
                               minval=0, 
                               maxval=patch_aspect_ratio[1])
    patch_h = tf.cast(tf.multiply(h, h_prop), tf.int32)
    patch_w = tf.cast(tf.multiply(w, w_prop), tf.int32)
    patch = tf.zeros([patch_h, patch_w], tf.float32)

    # Generate random location for patches
    x = tf.random.uniform(shape=[1], 
                          maxval=tf.cast(w, tf.int32) - patch_w, 
                          dtype=tf.int32)
    y = tf.random.uniform(shape=[1], 
                          maxval=tf.cast(h, tf.int32) - patch_h, 
                          dtype=tf.int32)
    
    # Pad patch with ones so it has the same shape as the image
    pad_vert = tf.concat([y, tf.cast(h, tf.int32) - y - patch_h], axis=0)
    pad_hor = tf.concat([x, tf.cast(w, tf.int32) - x - patch_w], axis=0)
    paddings = tf.stack([pad_vert, pad_hor])
    paddings = tf.cast(paddings, tf.int32)
    
    patch = tf.pad(patch, paddings, constant_values=1.)

    return tf.multiply(image, tf.expand_dims(patch, -1)), annots


@tf.function
def zoom(image: tf.Tensor, 
         annots: Annotation, 
         zoom_factor: float) -> ObjectDetectionInstance:
    labels = annots[0]
    boxes = annots[1]

    im_shape = tf.shape(image)
    h, w = im_shape[0], im_shape[1]

    # Calculate new size
    new_size = tf.cast([h, w] * zoom_factor, tf.int32)
    new_size = tf.maximum(new_size, [1, 1])

    # Resize the image
    image = tf.image.resize(image, new_size)

    # Scale the boxes
    scale = tf.cast(new_size / [h, w], tf.float32)
    boxes = boxes * tf.tile(scale, [2])

    return image, (labels, boxes)



@tf.function
def no_transform(image: tf.Tensor, 
                 annots: Annotation) -> ObjectDetectionInstance:
    return image, annots



class Augmentation(abc.ABC):

    def __init__(self, min_zoom: float = 0.4, max_zoom: float = 1.2, prob: float = .32) -> None:
        self.prob = prob
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    @abc.abstractmethod
    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        raise NotImplementedError

    def __call__(self,
                 image: tf.Tensor, 
                 annot: Annotation) -> ObjectDetectionInstance:

        image, annot = tf.cond(tf.random.uniform([1]) < self.prob,
                               lambda: horizontal_flip(image, annot),
                               lambda: no_transform(image, annot))

        return image, annot





class RandomHorizontalFlip(Augmentation):

    def __init__(self, prob: float = .3) -> None:
        super(RandomHorizontalFlip, self).__init__(prob=prob)


    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return horizontal_flip(image, annot)


class RandomCrop(Augmentation):

    def __init__(self, prob: float = .3) -> None:
        super(RandomCrop, self).__init__(prob=prob)

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return crop(image, annot)


class RandomErase(Augmentation):

    def __init__(self, 
                 prob: float = .15, 
                 patch_proportion: Tuple[float, float] = (.2, .2)) -> None:
        super(RandomErase, self).__init__(prob=prob)
        self.patch_proportion = patch_proportion

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return erase(image, annot, self.patch_proportion)


class RandomVerticalFlip(Augmentation):

    def __init__(self, prob: float = .3) -> None:
        super(RandomVerticalFlip, self).__init__(prob=prob)

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return vertical_flip(image, annot)
    

class RandomZoom(Augmentation):

    def __init__(self, min_zoom: float = 0.3, max_zoom: float = 1.2, prob: float = .5) -> None:
        super(RandomZoom, self).__init__(min_zoom=min_zoom, max_zoom=max_zoom)

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        zoom_factor = tf.random.uniform([], self.min_zoom, self.max_zoom)
        return zoom(image, annot, zoom_factor)


class RandomCutMix(Augmentation):

    def __init__(self, prob: float = .3, beta: float = 1.0) -> None:
        super(RandomCutMix, self).__init__(prob=prob)
        self.beta = beta

    def augment(self, 
                image: tf.Tensor, 
                annot: Annotation) -> ObjectDetectionInstance:
        return self.cutmix(image, annot)

    def cutmix(self, image: tf.Tensor, annot: Annotation) -> ObjectDetectionInstance:
        labels = annot[0]
        boxes = annot[1]

        im_shape = tf.shape(image)
        h, w = im_shape[0], im_shape[1]

        # Create lambda
        l = tf.random.gamma(shape=[], alpha=self.beta, beta=self.beta)

        # Create mask
        cut_rat = tf.math.sqrt(1. - l)
        cut_w = tf.cast(tf.cast(w, tf.float32) * cut_rat, tf.int32)
        cut_h = tf.cast(tf.cast(h, tf.float32) * cut_rat, tf.int32)

        # Uniform random cut center
        cx = tf.random.uniform(shape=[], minval=0, maxval=w, dtype=tf.int32)
        cy = tf.random.uniform(shape=[], minval=0, maxval=h, dtype=tf.int32)

        # Get start and end coordinates for cut
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, h)

        # Get the second image and labels
        second_image, second_annot = tf.cond(tf.random.uniform([1]) < self.prob,
                                             lambda: horizontal_flip(image, annot),
                                             lambda: no_transform(image, annot))
        second_labels = second_annot[0]
        second_boxes = second_annot[1]

        # Create new image by replacing cut with second image
        mask = tf.ones_like(image)
        mask[bby1:bby2, bbx1:bbx2, :] = 0
        image = image * mask + second_image * (1 - mask)

        # Interpolate labels
        labels = l * labels + (1 - l) * second_labels

        # Adjust boxes
        mask = tf.ones_like(boxes)
        mask[:, bbx1:bbx2, bby1:bby2, :] = 0
        boxes = boxes * mask + second_boxes * (1 - mask)

        return image, (labels, boxes)