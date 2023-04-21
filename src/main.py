import json
import mlflow
import os
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.faiss_index import FaissIndex
from src.models.predict import FunnelModel
from starlette.middleware.cors import CORSMiddleware


class Query(BaseModel):
    query: str


PARENT_DIR = os.path.abspath(os.path.join('', os.pardir))
DOCUMENTS_PATH = PARENT_DIR + '/data/processed/documents.json'
ML_RUNS_PATH = PARENT_DIR + '/models/ml_runs/'
MODEL_IS_READY, INDEX_IS_READY = False, False

with open(DOCUMENTS_PATH) as f:
    documents = json.load(f)

if mlflow.get_tracking_uri() != 'file:///' + ML_RUNS_PATH:
    mlflow.set_tracking_uri('file:///' + ML_RUNS_PATH)
EXP_ID = mlflow.get_experiment_by_name('QuoraRankingExtendedTraining').experiment_id
RUN_ID = mlflow.search_runs(experiment_ids=[EXP_ID])['run_id'][0]

MODEL_URI = "runs:/{}/model".format(RUN_ID)
VOCAB_URI = "runs:/{}/vocab".format(RUN_ID)
knrm, MODEL_IS_READY = mlflow.pytorch.load_model(MODEL_URI), True
emb_layer = knrm.embeddings.state_dict()['weight']
vocab = mlflow.artifacts.load_dict(VOCAB_URI)

index = FaissIndex(documents)
index, INDEX_IS_READY = index.prepare_index(emb_layer, vocab), True
model = FunnelModel(index, knrm, documents, vocab)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/ping')
def ping():
    if not MODEL_IS_READY:
        return {'status': "Model is not ready"}
    return {'status': 'ok'}


@app.post('/query')
def query(query: Query):
    if not MODEL_IS_READY or not INDEX_IS_READY:
        return {'status': "FAISS is not initialized!"}
    suggestions = model.get_docs(query.query)

    return {'suggestions': suggestions}
