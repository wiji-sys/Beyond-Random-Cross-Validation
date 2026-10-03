import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import runpy,json,platform
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
r=runpy.run_path(str(ROOT/'model_definitions.py'),run_name='model_definitions')
df=r['df']; folds=pd.read_csv(ROOT/'original_fold_assignments.csv')
models=['RandomForestLogCount','ExtraTreesLogCount']; rows=[];factors=[];predictions=[]
with threadpool_limits(limits=1):
 for scheme in folds.scheme.unique():
  for model in models:
   raw=np.full(len(df),np.nan);corrected=raw.copy()
   for fold in range(1,6):
    ids=folds.loc[(folds.scheme==scheme)&(folds.fold==fold),'row_id'].to_numpy()
    mask=df.row_id.isin(ids);train=df.loc[~mask];test=df.loc[mask]
    b=r['fit_bundle'](model,train);pipe=b['pipeline'];cols=r['CATEGORICAL']+r['NUMERIC_COUNT']
    # Classical global Duan factor from fitted training residuals only.
    e=np.log1p(train[r['TARGET']].to_numpy())-pipe.predict(train[cols])
    factor=float(np.mean(np.exp(e)))
    logpred=pipe.predict(test[cols]);raw[ids]=np.clip(np.expm1(logpred),1e-9,None)
    corrected[ids]=np.clip(np.exp(logpred)*factor-1,1e-9,None)
    factors.append({'scheme':scheme,'model':model,'fold':fold,'training_n':len(train),'smearing_factor':factor})
   for label,pred in [('uncorrected',raw),('duan_training_residuals',corrected)]:
    assert np.isfinite(pred).all()
    t=df[['row_id','provinsi_pt','jumlah_lulusan']].copy();t['predicted']=pred;t['scheme']=scheme;t['model']=model;t['correction']=label
    predictions.append(t)
    p=t.groupby('provinsi_pt',as_index=False)[['jumlah_lulusan','predicted']].sum()
    rows.append({'scheme':scheme,'model':model,'correction':label,**r['metric_dict'](p.jumlah_lulusan,p.predicted)})
   print('Completed',scheme,model,flush=True)
metrics=pd.DataFrame(rows);metrics.to_csv(ROOT/'smearing_province_metrics.csv',index=False)
pd.DataFrame(factors).to_csv(ROOT/'training_smearing_factors.csv',index=False)
pd.concat(predictions).to_csv(ROOT/'oof_predictions.csv',index=False)
old=pd.read_csv(ROOT/'original_metrics.csv');old=old[old.level=='province']
audit=metrics[metrics.correction=='uncorrected'].merge(old,on=['scheme','model'],suffixes=('_rerun','_original'))
audit['rmsle_difference']=audit.rmsle_rerun-audit.rmsle_original
audit[['scheme','model','rmsle_difference']].to_csv(ROOT/'replication_audit.csv',index=False)
assert audit.rmsle_difference.abs().max()<1e-10
print(metrics[['scheme','model','correction','rmsle','mae','predicted_to_observed_ratio']].to_string(index=False))
