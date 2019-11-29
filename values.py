"""Pass ENV VARS along."""
import os
from models.export_keras_to_tf import ModelConfig

MODEL_SOURCE = os.getenv('MODEL_SOURCE', None)
MODEL_SUPER_RESOLUTION = ModelConfig(
    name='SuperResolution',
    version='v1',
    input_layer_name='LR',
    filepath='models/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.h5',
)