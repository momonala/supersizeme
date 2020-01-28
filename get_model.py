import io
import logging
import socket
from base64 import b64encode

import numpy as np
from PIL import Image
from googleapiclient import discovery

from values import MODEL_SOURCE, SOCKET_TIME_OUT_IN_SEC, NAME_GOOGLE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_model_from_env():
    """Use local model or Google ML Engine."""
    if MODEL_SOURCE == 'local':
        logger.info('Using local model')
        return get_local_rdn_model()
    elif MODEL_SOURCE == 'mlengine':
        logger.info('Using mlengine model')
        return get_mlengine_rdn_model()
    else:
        raise ValueError('must define MODEL_SOURCE to be `mlengine` or `local`')


def get_local_rdn_model():
    from ISR.models import RDN
    rdn = RDN(arch_params={"C": 6, "D": 20, "G": 64, "G0": 64, "x": 2})
    rdn.model.load_weights(
        "models/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5"
    )
    return rdn


def get_mlengine_rdn_model():
    socket.setdefaulttimeout(SOCKET_TIME_OUT_IN_SEC)
    service = discovery.build(serviceName='ml', version='v1', cache_discovery=False)
    mlengine_service = service.projects()
    return mlengine_service


class RDNModel(object):

    def __init__(self):
        self.model = get_model_from_env()
        self.predict = {
            'mlengine': self._predict_mlengine,
            'local': self._predict_local,
        }.get(MODEL_SOURCE)

    def _predict_local(self, image_bytes):
        """Upsample the image using a local model.

        Args:
            image_bytes (file-like): An object with a read method like BytesIO/SpooledTemporaryFile.

        Returns:
            np.ndarray: high resolution RGB image of data type np.uint8
        """
        img = Image.open(image_bytes)
        low_res_img = np.array(img)
        # low_res_img = np.expand_dims(np.array(img), axis=0)  # needed if using Keras.models.load_model
        high_res_img = self.model.predict(low_res_img)
        high_res_img = high_res_img.astype(np.uint8)
        # high_res_img = np.squeeze(high_res_img, axis=0)  # needed if using Keras.models.load_model
        return high_res_img

    def _predict_mlengine(self, image_bytes):
        """Upsample the image using Google ML Engine.

        Args:
            image_bytes (file-like): An object with a read method like BytesIO/SpooledTemporaryFile.

        Returns:
            np.ndarray: high resolution RGB image of data type np.uint8
        """

        image = Image.open(image_bytes)
        img_byte_array = io.BytesIO()
        image.convert('RGB').save(img_byte_array, format='JPEG', quality=100)

        # convert JPEG bytes to a base64 (alphanumeric) string
        base64_bytes = b64encode(img_byte_array.getvalue())
        image_b64_bytestrings = [base64_bytes.decode('utf-8')]
        model_input = [{'image_bytes': {'b64': image_b64_bytestring}} for image_b64_bytestring in image_b64_bytestrings]

        # perform mlengine prediction
        response = self.model.predict(
            name=NAME_GOOGLE,
            body={'instances': model_input}
        ).execute()
        layer_name = list(response['predictions'][0].keys())[0]
        high_res_img = [np.array(prediction[layer_name]) for prediction in response['predictions']]

        # clip float32 values to [0, 1], then normalize to [0, 255] at datatype uint8 (for PIL)
        return np.squeeze((np.clip(high_res_img, 0, 1) * 255).astype(np.uint8), axis=0)

