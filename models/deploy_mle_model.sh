#!/usr/bin/env bash

ACTION=$1
PROJECT_NAME="ct-machine-learning"
MODEL_NAME="SuperResolution"
MODEL_VERSION="v1"
MODEL_BINARIES="gs://ct-machine-learning-ml-models-ml-engine/models_mle/$MODEL_NAME/$MODEL_VERSION"


## create the model on ML Engine (only needed to run once)
if [ "$ACTION" = "create" ]; then
    gcloud ai-platform models create $MODEL_NAME \
    --regions europe-west1 \
    --project $PROJECT_NAME \
    --description "upsample images with super resolution" \
    --labels name=super_resolution \
    --enable-logging
fi

## list the models present on ML Engine
if [ "$ACTION" = "describe" ]; then
    gcloud ai-platform models describe $MODEL_NAME \
    --project $PROJECT_NAME
fi

## create a new model version (instance) on ML Engine
## NOTE: this can only be run after model 'create' has been run once (above)
## NOTE: will not let you overwrite an existing version with the same version name (must delete the existing one)
if [ "$ACTION" = "version" ]; then
    gcloud ai-platform versions create $MODEL_VERSION \
    --project $PROJECT_NAME \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --config "deploy/super_resolution/config_model_version.yaml"
fi
