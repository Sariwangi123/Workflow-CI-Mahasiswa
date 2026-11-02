"""MLflow Project launcher records its exact run ID and a portable model."""
from pathlib import Path
import json
import os
import sys
import mlflow

root = Path(__file__).resolve().parent
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
tracking = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5050' if os.name == 'nt' else (root/'mlruns').as_uri())
# MLflow Project memakai perintah python; pastikan interpreter .venv yang sama.
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
mlflow.set_tracking_uri(tracking)
# Training dijalankan oleh perintah mlflow run pada workflow.
summary = json.loads((root/'MLProject/training_artifacts/tuning/run.json').read_text())
run_id = summary['run_id']
assert mlflow.MlflowClient().get_run(run_id).info.status == 'FINISHED'
artifacts = root/'release_artifacts'
artifacts.mkdir(exist_ok=True)
mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='model', dst_path=str(artifacts))
manifest = {'run_id':run_id, 'model_uri':f'runs:/{run_id}/model',
            'git_sha':os.getenv('GITHUB_SHA', 'local-validation'), 'tracking_uri':tracking}
(artifacts/'run.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
if os.getenv('GITHUB_OUTPUT'):
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
        f.write(f'run_id={run_id}\n')
print(json.dumps(manifest, indent=2))
