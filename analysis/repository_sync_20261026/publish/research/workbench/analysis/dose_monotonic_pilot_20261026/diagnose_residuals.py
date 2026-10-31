"""Post-result error description only; not a corrected predictor or promotion metric."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent
d=pd.read_csv(O/'oof_predictions.csv',float_precision='round_trip')
rows=[]
for (dataset,model,panel),g in d.groupby(['dataset','model','panel_id']):
 y=g.observed_percent.to_numpy();p=g.predicted_percent.to_numpy()
 mse=float(np.mean((p-y)**2));level=float((p.mean()-y.mean())**2);shape=float(np.mean(((p-p.mean())-(y-y.mean()))**2))
 assert abs(mse-level-shape)<1e-8
 rows.append(dict(dataset=dataset,model=model,panel_id=panel,panel_MSE=mse,panel_mean_offset_squared=level,centered_shape_MSE=shape))
f=pd.DataFrame(rows);f.to_csv(O/'residual_decomposition_panels.csv',index=False,encoding='utf-8-sig')
summary=[]
for (dataset,model),g in f.groupby(['dataset','model']):
 total=float(g.panel_MSE.mean());level=float(g.panel_mean_offset_squared.mean());shape=float(g.centered_shape_MSE.mean())
 summary.append(dict(dataset=dataset,model=model,macro_panel_MSE=total,macro_mean_offset_squared=level,macro_centered_shape_MSE=shape,mean_offset_share_of_squared_error=level/total,scope='post-result descriptive decomposition; test-label means used only for diagnosis, never model input or adjusted prediction; not a new performance claim'))
(O/'residual_decomposition.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
