"""Helpers to prepare a Keras model for Tensorflow Serving (Google ML Engine)"""

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import copy, rmtree, move
from tempfile import mkdtemp
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras

ROOT_DIR = os.getcwd()


@dataclass
class ModelConfig:
    """
    name (str): e.g. to identify the model on ML Engine.
    version (str): e.g. to identify the model version on ML Engine.
    filepath (Optional[str]): Local path where model is saved, e.g. "models_mle/ImageCategorizer/v1/base_models"
    input_layer_name (Optional[str]): e.g. used to refer to the first layer in keras models.
    output_layer_name (Optional[str]): e.g. used to refer to the last layer in keras models.
    input_shape (Optional[Tuple[int, ...]]): e.g. (299, 299) for the width and height of images in image models.
    output_shape (Optional[Tuple[int, ...]]): e.g. (100, ) for a classifier with 100 classes.
    """
    name: str
    version: str
    filepath: Optional[str] = None
    input_layer_name: Optional[str] = None
    output_layer_name: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = ()
    output_shape: Optional[Tuple[int, ...]] = ()


def serving_input_receiver_fn_template(layer_name: str) -> tf.estimator.export.ServingInputReceiver:
    """Return a constructed tf.estimator.export.ServingInputReceiver().

    Needed for exporting a Tensorflow model. See:
        tensorflow_estimator.python.estimator.estimator.Estimator#export_savedmodel

    Args:
        layer_name (str): name of output layer in tensorflow model.

    Returns: tf.estimator.export.ServingInputReceiver()

    """
    # set a placeholder for input image bytes as string tensor
    input_ph = tf.placeholder(tf.string, shape=[None])

    # this will generate the input tensor to the keras model after applying preprocessing
    images_tensor = tf.map_fn(prepare_image, input_ph, back_prop=False, dtype=tf.float32)

    # NOTE: image byte string must have '_bytes' as a suffix for MLE to correctly identify it as byte data
    serving_input_receiver_output = tf.estimator.export.ServingInputReceiver(
        {layer_name: images_tensor},  # input tensor name expected by the keras model
        {'image_bytes': input_ph},  # image byte string read from request input json data this argument
    )
    return serving_input_receiver_output


def export_keras_estimator(model_config: ModelConfig) -> None:
    """Export a keras model to a Tensorflow Estimator and saves it.

    Estimator is saved at "models_mle/{model_config.name}/{model_config.version}".

    Args:
        model_config (utils.ModelConfig.ModelConfig): Has to instantiate the parameters name, version, filepath,
                                                      and input_layer_name.

    """
    if not all([model_config.filepath, model_config.input_layer_name]):
        raise ValueError("model_config is missing values for the attributes filepath or input_layer_name")

    h5_model_path = os.path.join(model_config.filepath)
    tf_model_path = mkdtemp(f'models_mle_{model_config.name}_{model_config.version}')

    # export checkpoint model for keras estimator
    estimator = keras.estimator.model_to_estimator(keras_model_path=h5_model_path, model_dir=tf_model_path)

    # Copy files one dir up, otherwise ML Engine cannot find the checkpoints
    # TODO: find out why this is necessary
    move_path = tf_model_path + '/keras'
    parent_dir = str(Path(move_path).parent)
    for f in os.listdir(move_path):
        copy(os.path.join(move_path, f), parent_dir)

    export_path = os.path.join(ROOT_DIR, 'models_mle', model_config.name, model_config.version)

    source = estimator.export_saved_model(
        export_dir_base=export_path,
        serving_input_receiver_fn=partial(serving_input_receiver_fn_template, layer_name=model_config.input_layer_name),
    )

    # Delete temporary tensorflow model checkpoints, since we saved them in the SavedModel format now
    rmtree(tf_model_path)

    # rename exported tf model directory
    source = os.path.join(source.decode('utf-8'))
    target = os.path.join(ROOT_DIR, 'models_mle', model_config.name, model_config.version, 'tf')
    if os.path.exists(target):
        rmtree(target)
    os.rename(source, target)

    root = os.path.join(ROOT_DIR, 'models_mle', model_config.name, model_config.version)
    if os.path.exists(os.path.join(root, 'variables')):
        rmtree(os.path.join(root, 'variables'))
    for filename in os.listdir(os.path.join(root, 'tf')):
        move(os.path.join(root, 'tf', filename), os.path.join(root, filename))
    os.rmdir(target)

    print(f'Saved TF model ({model_config.version}) to {root}')


def prepare_image(image_str_tensor):
    """Ingest and process raw image byte-string.

    Args:
        image_str_tensor (tf.str): Tensorflow string type representing image bytes.

    Returns:
        preprocessed image tensor input to keras model.

    """
    # decode 3-D image tensor from jpeg byte string
    image = tf.image.decode_jpeg(image_str_tensor, channels=3)
    return image_preprocessing(image)


def image_preprocessing(image):
    """Image preprocessing for input to a CNN model.

    Implement the standard preprocessing that needs to be applied to the
    image tensors before passing them to the model (based off of Xception.preprocessing).

    Args:
        image (tensor): 3D tensor representing image data

    Returns:
        prepreocessed image tensor

    """
    # convert to float to allow for division operation
    image = tf.to_float(image)
    # below is the same as `xception.preprocess_input(image)`
    image /= 127.5
    image -= 1.0
    return image

