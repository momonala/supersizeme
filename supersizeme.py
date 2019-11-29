import io
import logging
from datetime import datetime

from PIL import Image
from flask import request

from get_model import RDNModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

rdn_model = RDNModel()


def upscale_image_from_bytes():
    """Upscale an image with super super resolution.

    Returns:
        (file-like): An json serializable image bytes object.
    """
    logger.info(f"Attempting to upsample image")

    #  image_bytes: (file-like): An object with a read method like BytesIO/SpooledTemporaryFile.
    image_bytes = io.BytesIO(request.data) if request.data != b"" else None
    if image_bytes is None:
        logger.info(f'No image data found.')
        return 400

    t0 = datetime.now()
    try:
        super_res_img_array = rdn_model.predict(image_bytes)
    except Exception as e:
        logger.error(f"Something failed! {e}", exc_info=True)
        return 500

    time_elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"Sucessfully upsampled image in {time_elapsed} seconds")

    # cast np.array to bytes for HTTP response
    output_image = Image.fromarray(super_res_img_array)
    byte_buffer = io.BytesIO()
    output_image.save(byte_buffer, format='PNG')
    output_image_bytes = byte_buffer.getvalue()
    return output_image_bytes
