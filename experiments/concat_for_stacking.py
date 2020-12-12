import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ayniy.utils import Data


def load_oof_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
    return oof


def load_pred_from_run_id(run_id: str, to_rank: False):
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        pred = rankdata(pred) / len(pred)
    return pred


run_ids = [
    'run000',
    'run001',
    'run002',
    'run003',
    'run004'
]
fe_name = 'stack000'

if __name__ == "__main__":
    y_train = Data.load('../input/pickle/y_train_fe000.pkl')
    oofs = [load_oof_from_run_id(ri, to_rank=False) for ri in run_ids]
    preds = [load_pred_from_run_id(ri, to_rank=False) for ri in run_ids]

    X_train = pd.DataFrame(np.stack(oofs)).T
    X_test = pd.DataFrame(np.stack(preds)).T

    X_train.columns = run_ids
    X_test.columns = run_ids
    print(X_train.head())

    Data.dump(X_train, f'../input/pickle/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/pickle/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/pickle/X_test_{fe_name}.pkl')
