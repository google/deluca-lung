import os
import pickle
from google.cloud import storage
import hypertune

from deluca.lung.core import ROOT


def dump(obj, args, name, metric, step, bucket_name="vizier_cloud_codelab"):
    directory = "/tmp/results"

    args = vars(args)
    filename = "-".join([f"{k}_{v}" for k, v in args.items()])
    filename = f"{filename}.pkl"

    if not os.path.exists(directory):
        os.makedirs(directory)

    obj = {
        "result": obj,
        "args": args
    }

    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    with open(os.path.join(ROOT, "vizier/jobs", name, "run"), "r") as f:
        run = int(f.read().strip())
    print(run)
    blob = bucket.blob(f"{name}/{run}/{filename}")
    blob.upload_from_filename(filename=path)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="score",
        metric_value=metric,
        global_step=step,
    )
