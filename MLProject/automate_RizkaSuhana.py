"""Preprocessing identik dengan notebook; split sebelum mempelajari transformasi."""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'student_status_raw' / 'data.csv'
OUTPUT = Path(__file__).resolve().parent / 'student_status_preprocessing'
SEED = 42
CLASSES = ['Dropout', 'Enrolled', 'Graduate']
CATEGORICAL = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
    'Gender', 'Scholarship holder', 'International']
UNCERTAIN_TIME = ['Debtor', 'Tuition fees up to date', 'Unemployment rate', 'Inflation rate', 'GDP']

def select_data(raw):
    df = raw.copy()
    df.columns = df.columns.str.strip()
    if df['Target'].isna().any() or set(df['Target'].unique()) != set(CLASSES):
        raise ValueError('Target harus berisi tiga kelas UCI tanpa nilai kosong.')
    dropped = [c for c in df if '2nd sem' in c or c in UNCERTAIN_TIME]
    # Hilangkan baris identik agar duplikat tidak tersebar lintas split.
    df = df.drop_duplicates()
    X = df.drop(columns=['Target'] + dropped)
    # Fitur identik dengan label berbeda juga dikelompokkan secara konservatif:
    # keluarkan semua pola ambigu, jangan memilih label secara arbitrer.
    ambiguous = X.duplicated(keep=False)
    return X.loc[~ambiguous].reset_index(drop=True), df.loc[~ambiguous, 'Target'].reset_index(drop=True), dropped

def make_preprocessor(columns):
    numeric = [c for c in columns if c not in CATEGORICAL]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL),
    ], verbose_feature_names_out=True)

def preprocess(raw_path=RAW, output=OUTPUT):
    raw = pd.read_csv(raw_path, sep=';')
    X, y, dropped = select_data(raw)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)
    preprocessor = make_preprocessor(X.columns)
    preprocessor.fit(X_train)
    names = preprocessor.get_feature_names_out()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, features, labels in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
        features.to_csv(output / f'X_{split}_selected.csv', index=False)
        pd.DataFrame(preprocessor.transform(features), columns=names).to_csv(output / f'X_{split}.csv', index=False)
        labels.to_csv(output / f'y_{split}.csv', index=False)
    joblib.dump(preprocessor, output / 'preprocessor.joblib')
    manifest = {'raw_sha256': hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'retained_rows': len(X), 'removed_rows': len(raw)-len(X),
        'features': list(X.columns), 'categorical': CATEGORICAL, 'dropped_features': dropped,
        'transformed_features': list(names), 'classes': CLASSES, 'seed': SEED,
        'prediction_time': 'end_of_first_semester',
        'split_sizes': {'train': len(X_train), 'validation': len(X_val), 'test': len(X_test)},
        'target_counts': y.value_counts().to_dict()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.raw, args.output), indent=2))
