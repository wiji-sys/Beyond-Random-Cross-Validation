from __future__ import annotations

import json
import math
import platform
import random
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
display = print
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_SEED = 42
N_SPLITS = 5
BOOTSTRAP_ITERATIONS = 500
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 120)
pd.set_option('display.float_format', lambda x: f'{x:,.6f}')

print('Python      :', sys.version.split()[0])
print('Pandas      :', pd.__version__)
print('NumPy       :', np.__version__)
print('SciPy       :', scipy.__version__)
print('Scikit-learn:', sklearn.__version__)
print('Platform    :', platform.platform())

import os, hashlib
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
TABLE_DIR=ROOT
MODEL_DATA_FILE=ROOT/'02_modeling_dataset.csv'
df=pd.read_csv(MODEL_DATA_FILE)

required=[
    'provinsi_pt','kabupaten_pt','jk','nm_jenj_didik',
    'jumlah_lulusan','jumlah_terdaftar','active_share','mbkm_share',
    'log_pt','pt_high_share','prof_share','cert_share'
]
missing=sorted(set(required)-set(df.columns))
assert not missing, f'Kolom hilang: {missing}'
assert df['jumlah_lulusan'].ge(0).all()
assert df['jumlah_terdaftar'].gt(0).all()
assert df['provinsi_pt'].nunique() >= N_SPLITS

for col in ['provinsi_pt','kabupaten_pt','jk','nm_jenj_didik']:
    df[col]=df[col].astype('string').str.normalize('NFKC').str.replace(r'\s+',' ',regex=True).str.strip()

df['log_enrolled']=np.log1p(df['jumlah_terdaftar'])
df['row_id']=np.arange(len(df))

CATEGORICAL=['jk','nm_jenj_didik']
NUMERIC_RATE=['active_share','mbkm_share','log_pt','pt_high_share','prof_share','cert_share']
NUMERIC_COUNT=NUMERIC_RATE+['log_enrolled']
TARGET='jumlah_lulusan'
EXPOSURE='jumlah_terdaftar'
GROUP='provinsi_pt'
ID_COLS=['row_id','provinsi_pt','kabupaten_pt','jk','nm_jenj_didik']

assert not df.duplicated(['provinsi_pt','kabupaten_pt','jk','nm_jenj_didik']).any()

summary=pd.DataFrame([{
    'rows':len(df), 'provinces':df[GROUP].nunique(),
    'districts':df['kabupaten_pt'].nunique(),
    'total_graduates':df[TARGET].sum(),
    'total_registered':df[EXPOSURE].sum(),
    'minimum_target':df[TARGET].min(),
    'maximum_target':df[TARGET].max(),
}])
summary.to_csv(TABLE_DIR/'03_data_summary.csv',index=False)
display(summary)
def dense_onehot(drop=None):
    return OneHotEncoder(handle_unknown='ignore', drop=drop, sparse_output=False)


def linear_preprocessor():
    return ColumnTransformer([
        ('num',Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler()),
        ]),NUMERIC_RATE),
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',dense_onehot(drop='first')),
        ]),CATEGORICAL),
    ],sparse_threshold=0)


def tree_rate_preprocessor():
    return ColumnTransformer([
        ('num',SimpleImputer(strategy='median'),NUMERIC_RATE),
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',dense_onehot()),
        ]),CATEGORICAL),
    ],sparse_threshold=0)


def tree_count_preprocessor():
    return ColumnTransformer([
        ('num',SimpleImputer(strategy='median'),NUMERIC_COUNT),
        ('cat',Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',dense_onehot()),
        ]),CATEGORICAL),
    ],sparse_threshold=0)

MODEL_NAMES=[
    'ExposureRateBaseline',
    'NB2_Offset',
    'PoissonRate',
    'TweedieRate',
    'HistGBPoissonRate',
    'RandomForestLogCount',
    'ExtraTreesLogCount',
]


def fit_bundle(model_name: str, train: pd.DataFrame):
    y=train[TARGET].to_numpy(float)
    exposure=train[EXPOSURE].to_numpy(float)
    X=train[CATEGORICAL+NUMERIC_COUNT]

    if model_name=='ExposureRateBaseline':
        return {'kind':'baseline','rate':float(y.sum()/exposure.sum())}

    if model_name=='NB2_Offset':
        pre=linear_preprocessor()
        design=pre.fit_transform(train[CATEGORICAL+NUMERIC_RATE])
        design=sm.add_constant(design,has_constant='add')
        result=NegativeBinomial(
            y, design, offset=np.log(exposure), loglike_method='nb2'
        ).fit(disp=False,maxiter=400)
        if not result.mle_retvals.get('converged',False):
            raise RuntimeError('NB2 tidak konvergen.')
        return {'kind':'nb2','preprocessor':pre,'result':result}

    if model_name in {'PoissonRate','TweedieRate','HistGBPoissonRate'}:
        rate=y/exposure*1000.0
        weights=exposure/exposure.mean()
        if model_name=='PoissonRate':
            pipe=Pipeline([
                ('preprocessor',linear_preprocessor()),
                ('model',PoissonRegressor(alpha=0.1,max_iter=300,solver='newton-cholesky')),
            ])
        elif model_name=='TweedieRate':
            pipe=Pipeline([
                ('preprocessor',linear_preprocessor()),
                ('model',TweedieRegressor(power=1.5,alpha=0.1,link='log',max_iter=300,solver='newton-cholesky')),
            ])
        else:
            pipe=Pipeline([
                ('preprocessor',tree_rate_preprocessor()),
                ('model',HistGradientBoostingRegressor(
                    loss='poisson',learning_rate=0.05,max_iter=200,
                    max_leaf_nodes=31,min_samples_leaf=20,
                    l2_regularization=1.0,random_state=RANDOM_SEED,
                )),
            ])
        pipe.fit(X,rate,model__sample_weight=weights)
        return {'kind':'rate','pipeline':pipe}

    if model_name in {'RandomForestLogCount','ExtraTreesLogCount'}:
        if model_name=='RandomForestLogCount':
            estimator=RandomForestRegressor(
                n_estimators=150,min_samples_leaf=2,max_features=0.8,
                n_jobs=1,random_state=RANDOM_SEED,
            )
        else:
            estimator=ExtraTreesRegressor(
                n_estimators=150,min_samples_leaf=2,max_features=1.0,
                n_jobs=1,random_state=RANDOM_SEED,
            )
        pipe=Pipeline([
            ('preprocessor',tree_count_preprocessor()),
            ('model',estimator),
        ])
        pipe.fit(X,np.log1p(y))
        return {'kind':'log_count','pipeline':pipe}

    raise KeyError(model_name)


def predict_bundle(bundle, test: pd.DataFrame) -> np.ndarray:
    exposure=test[EXPOSURE].to_numpy(float)
    X=test[CATEGORICAL+NUMERIC_COUNT]
    kind=bundle['kind']
    if kind=='baseline':
        pred=bundle['rate']*exposure
    elif kind=='nb2':
        design=bundle['preprocessor'].transform(test[CATEGORICAL+NUMERIC_RATE])
        design=sm.add_constant(design,has_constant='add')
        pred=bundle['result'].predict(exog=design,offset=np.log(exposure))
    elif kind=='rate':
        rate=bundle['pipeline'].predict(X)
        pred=rate*exposure/1000.0
    elif kind=='log_count':
        pred=np.expm1(bundle['pipeline'].predict(X))
    else:
        raise KeyError(kind)
    return np.clip(np.asarray(pred,float),1e-9,None)


def metric_dict(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-9,None)
    return {
        'mae':float(mean_absolute_error(y,p)),
        'rmse':float(mean_squared_error(y,p)**0.5),
        'rmsle':float(mean_squared_log_error(y,p)**0.5),
        'mean_poisson_deviance':float(mean_poisson_deviance(y,p)),
        'log_r2':float(r2_score(np.log1p(y),np.log1p(p))),
        'spearman_r':float(stats.spearmanr(y,p).statistic),
        'observed_total':float(y.sum()),
        'predicted_total':float(p.sum()),
        'predicted_to_observed_ratio':float(p.sum()/y.sum()),
    }


def save_figure(fig,name):
    path=FIGURE_DIR/name
    fig.savefig(path,dpi=300,bbox_inches='tight')
    plt.close(fig)
    return path

def run_repeat(item):
    repeat, seed = item
    with threadpool_limits(limits=1):
        groups=np.array(sorted(df[GROUP].unique()))
        rng=np.random.default_rng(seed)
        # Equal numbers of province groups (7,7,7,7,6), not outcome stratification.
        group_folds=np.array_split(rng.permutation(groups),5)
        preds={m:np.full(len(df),np.nan) for m in MODEL_NAMES}
        assignments=[]
        for fold, test_groups in enumerate(group_folds,1):
            mask=df[GROUP].isin(test_groups).to_numpy()
            train=df.loc[~mask]; test=df.loc[mask]
            assert not set(train[GROUP]) & set(test[GROUP])
            assignments.extend({'repeat':repeat,'seed':seed,'province':g,'fold':fold} for g in test_groups)
            for m in MODEL_NAMES:
                bundle=fit_bundle(m,train)
                preds[m][mask]=predict_bundle(bundle,test)
        rows=[]; prov_rows=[]
        oof=df[ID_COLS+[TARGET,EXPOSURE]].copy()
        for m in MODEL_NAMES:
            assert np.isfinite(preds[m]).all()
            oof[m]=preds[m]
            p=df[[GROUP,TARGET]].copy();p['pred']=preds[m]
            p=p.groupby(GROUP,as_index=False)[[TARGET,'pred']].sum()
            rows.append({'repeat':repeat,'seed':seed,'model':m,**metric_dict(p[TARGET],p['pred'])})
            for r in p.to_dict('records'):prov_rows.append({'repeat':repeat,'model':m,**r})
        oof.to_csv(ROOT/f'oof_repeat_{repeat:02}.csv',index=False)
        return rows,assignments,prov_rows

if __name__=='__main__':
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'statsmodels':__import__('statsmodels').__version__}
    (ROOT/'versions.json').write_text(json.dumps(versions,indent=2))
    metrics=[];assignments=[];prov=[]
    # Repeats fixed before results inspection; estimator seeds remain 42.
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(r,a,p) in enumerate(pool.map(run_repeat,enumerate(range(4201,4221),1)),1):
            metrics.extend(r);assignments.extend(a);prov.extend(p)
            print('Completed repeat',i,'of 20',flush=True)
            pd.DataFrame(metrics).to_csv(ROOT/'repeat_metrics.csv',index=False)
    m=pd.DataFrame(metrics)
    m['rank']=m.groupby('repeat')['rmsle'].rank(method='average')
    m.to_csv(ROOT/'repeat_metrics.csv',index=False)
    pd.DataFrame(assignments).to_csv(ROOT/'province_fold_assignments.csv',index=False)
    pd.DataFrame(prov).to_csv(ROOT/'province_predictions.csv',index=False)
    rows=[]
    for name,g in m.groupby('model'):
        rows.append({'model':name,'rmsle_mean':g.rmsle.mean(),'rmsle_sd':g.rmsle.std(ddof=1),'rmsle_min':g.rmsle.min(),'rmsle_max':g.rmsle.max(),'rank_median':g['rank'].median(),'rank_min':g['rank'].min(),'rank_max':g['rank'].max(),'first_count':int(g['rank'].eq(1).sum())})
    summary=pd.DataFrame(rows).sort_values('rmsle_mean');summary.to_csv(ROOT/'ranking_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)
