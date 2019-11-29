#!/usr/bin/env python3
import logging
import os

import connexion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = connexion.App(__name__)
app.add_api("swagger.yaml")
# set the WSGI application callable to allow using uWSGI: uwsgi --http :8080 -w app
application = app.app

if __name__ == "__main__":
    # run our standalone gevent server
    app.run(port=8080, server="gevent")
