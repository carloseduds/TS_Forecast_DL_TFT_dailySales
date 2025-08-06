FROM python:3.10

EXPOSE 5000

ARG MLFLOW_VERSION
ARG SERVER_DIR=/server
ARG MLFLOW_ARTIFACT_STORE

ENV MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_STORE}

COPY ./docker/mlflow/requirements.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    mkdir -p ${SERVER_DIR} ${MLFLOW_ARTIFACT_ROOT}

COPY ./docker/mlflow/scripts/wait-for-it.sh ${SERVER_DIR}/
RUN chmod +x ${SERVER_DIR}/wait-for-it.sh

WORKDIR ${SERVER_DIR}

CMD sh -c "./wait-for-it.sh ${MLFLOW_BACKEND_STORE}:${POSTGRES_PORT} -t ${WAIT_FOR_IT_TIMEOUT} -- \
    mlflow server \
    --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${MLFLOW_BACKEND_STORE}:${POSTGRES_PORT}/${POSTGRES_DATABASE} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host ${MLFLOW_TRACKING_SERVER_HOST} \
    --port ${MLFLOW_TRACKING_SERVER_PORT}"
