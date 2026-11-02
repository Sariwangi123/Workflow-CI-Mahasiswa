from pathlib import Path
import argparse
import json
import os
import sys
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr
from automate_RizkaSuhana import make_preprocessor

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

def metrics(model, X, y, prefix):
    pred, probs = model.predict(X), model.predict_proba(X)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    result = {prefix+'accuracy': accuracy_score(y, pred), prefix+'macro_f1': f1_score(y, pred, average='macro'),
              prefix+'weighted_f1': f1_score(y, pred, average='weighted'), prefix+'log_loss': log_loss(y, probs, labels=model.classes_),
              prefix+'roc_auc_ovr_weighted': roc_auc_score(y, probs, labels=model.classes_, multi_class='ovr', average='weighted')}
    for label in model.classes_:
        for metric in ['precision', 'recall', 'f1-score']:
            result[f'{prefix}{label}_{metric.replace("-", "_")}'] = report[label][metric]
    return result, report

def run(tune=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE/'student_status_preprocessing')
    parser.add_argument('--output', type=Path, default=HERE/'training_artifacts')
    parser.add_argument('--tracking-uri', default=os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5050'))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    if not os.getenv('MLFLOW_RUN_ID'): mlflow.set_experiment('Student Study Status - RizkaSuhana')
    out = args.output / ('tuning' if tune else 'baseline')
    out.mkdir(parents=True, exist_ok=True)
    def load(split):
        return pd.read_csv(args.data/f'X_{split}_selected.csv'), pd.read_csv(args.data/f'y_{split}.csv').squeeze('columns')
    X, y = load('train')
    Xv, yv = load('validation')
    pipeline = Pipeline([('preprocessing', make_preprocessor(X.columns)),
                         ('classifier', LogisticRegression(max_iter=1500, random_state=42))])
    mlflow.sklearn.autolog(disable=tune, log_input_examples=True, log_model_signatures=True)
    with mlflow.start_run(run_name='manual_tuning' if tune else 'baseline_autolog') as active:
        mlflow.set_tags({'prediction_time': 'end_of_first_semester', 'dataset': 'UCI-697', 'topic': 'student_status'})
        if tune:
            search = GridSearchCV(pipeline, [
                {'classifier':[LogisticRegression(max_iter=1500, random_state=42)],
                 'classifier__C':[.1, 1, 10], 'classifier__class_weight':[None, 'balanced']},
                {'classifier':[RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=2)],
                 'classifier__max_depth':[8, None], 'classifier__min_samples_leaf':[2, 5],
                 'classifier__class_weight':['balanced']}],
                 cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='f1_macro', n_jobs=2, refit=True)
            search.fit(X, y)
            model = search.best_estimator_
            pd.DataFrame(search.cv_results_).astype(str).to_csv(out/'cv_results.csv', index=False)
            mlflow.log_metric('best_cv_macro_f1', search.best_score_)
            mlflow.log_params({k: str(v) for k,v in model.get_params(deep=True).items()})
            mlflow.log_dict({'metric':'macro_f1', 'cv':'3-fold stratified seed 42', 'test_used_for_selection':False}, 'metric_info.json')
            (out/'estimator.html').write_text(estimator_html_repr(model), encoding='utf-8')
            train_metrics, _ = metrics(model, X, y, 'training_')
            train_pred = model.predict(X)
            train_metrics.update({'training_accuracy_score': accuracy_score(y, train_pred),
                'training_f1_score': f1_score(y, train_pred, average='weighted'),
                'training_precision_score': precision_score(y, train_pred, average='weighted', zero_division=0),
                'training_recall_score': recall_score(y, train_pred, average='weighted'),
                'training_roc_auc': roc_auc_score(y, model.predict_proba(X), labels=model.classes_, multi_class='ovr', average='weighted')})
            mlflow.log_metrics(train_metrics)
            mlflow.log_metric('training_score', model.score(X, y))
            ConfusionMatrixDisplay.from_estimator(model, X, y, cmap='Blues')
            plt.tight_layout(); plt.savefig(out/'training_confusion_matrix.png'); plt.close()
            mlflow.sklearn.log_model(model, 'model', signature=infer_signature(X, model.predict(X)), input_example=X.head(3))
        else:
            model = pipeline.fit(X, y)
        validation, report = metrics(model, Xv, yv, 'validation_')
        mlflow.log_metrics(validation)
        (out/'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        # Validation is used for candidate comparison; final test is a separate explicit step.
        joblib.dump(model, out/'model.joblib')
        mlflow.log_artifact(str(args.data/'manifest.json'))
        mlflow.log_artifacts(str(out), artifact_path='evaluation')
        result = {'run_id':active.info.run_id, 'model_uri':f'runs:/{active.info.run_id}/model',
                  'tracking_uri':args.tracking_uri, 'validation':validation,
                  'selected_classifier':type(model.named_steps['classifier']).__name__}
        (out/'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))

if __name__ == '__main__': run()
