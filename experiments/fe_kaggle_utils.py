import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kaggle_utils.features import count_null, count_encoding, count_encoding_interact, target_encoding, matrix_factorization
from kaggle_utils.features.category_encoding import CategoricalEncoder
from kaggle_utils.features.groupby import GroupbyTransformer, DiffGroupbyTransformer, RatioGroupbyTransformer


if __name__ == '__main__':
    train = pd.read_csv('../input/atmacup08-dataset/train.csv')
    test = pd.read_csv('../input/atmacup08-dataset/test.csv')
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(train.shape, test.shape)
    # (8359, 16) (8360, 11)
    categorical_cols = [
        'Name',
        'Platform',
        'Genre',
        'Publisher',
        'Developer',
        'Rating'
    ]
    numerical_cols = [
        'Critic_Score',
        'Critic_Count',
        'User_Score',
        'User_Count'
    ]
    target_col = 'Global_Sales'

    # target transformation
    train_test[target_col] = np.log1p(train_test[target_col])

    # label encoding
    ce = CategoricalEncoder(categorical_cols)
    train_test = ce.transform(train_test)
    train_test['is_User_Score_tbd'] = (train_test['User_Score'] == 'tbd').astype(int)
    train_test['User_Score'] = train_test['User_Score'].replace('tbd', np.nan).astype(float)

    # base
    train_test[[
        # 'Name',
        'Platform',
        'Year_of_Release',
        'Genre',
        # 'Publisher',
        'Critic_Score',
        'Critic_Count',
        'User_Score',
        'User_Count',
        # 'Developer',
        'Rating'
    ] + [target_col]].to_feather('../input/feather/train_test.ftr')

    # count null
    encode_col = [
        'Name',
        'Platform',
        'Year_of_Release',
        'Genre',
        'Publisher',
        'Critic_Score',
        'Critic_Count',
        'User_Score',
        'User_Count',
        'Developer',
        'Rating'
    ]
    count_null(train_test, encode_col).to_feather('../input/feather/count_null.ftr')

    # count encoding
    count_encoding(train_test, categorical_cols).to_feather('../input/feather/count_encoding.ftr')
    count_encoding_interact(train_test, categorical_cols).to_feather('../input/feather/count_encoding_interact.ftr')

    # target encoding
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    _train = train_test.dropna(subset=[target_col]).copy()
    _test = train_test.loc[train_test[target_col].isnull()].copy()
    num_bins = np.int(1 + np.log2(len(_train)))
    _train[target_col] = pd.qcut(
        _train[target_col],
        num_bins,
        labels=False
    )
    target_encoding(_train, _test, categorical_cols, target_col, cv).to_feather('../input/feather/target_encoding.ftr')

    # matrix factorization
    features_svd, features_lda = matrix_factorization(
        train_test, [
            'Platform',
            'Genre',
            'Rating'
        ],
        {'n_components_lda': 5, 'n_components_svd': 5}
    )

    features_svd.columns = [str(c) for c in features_svd.columns]
    features_lda.columns = [str(c) for c in features_lda.columns]
    features_svd.to_feather('../input/feather/features_svd.ftr')
    features_lda.to_feather('../input/feather/features_lda.ftr')

    # aggregation
    groupby_dict = [{
        'key': [
            'Platform'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Genre'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Developer'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Publisher'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Name'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Publisher',
            'Name'
        ],
        'var': [
            'Year_of_Release',
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Platform',
            'Genre'
        ],
        'var': [
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Genre',
            'Developer'
        ],
        'var': [
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Developer',
            'Platform'
        ],
        'var': [
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    }, {
        'key': [
            'Genre',
            'Developer',
            'Platform'
        ],
        'var': [
            'User_Count',
            'Critic_Count',
            'Critic_Score',
            'User_Score'
        ],
        'agg': ['mean', 'sum', 'median', 'min', 'max', 'var', 'std']
    },
    ]
    nunique_dict = [{
        'key': ['Publisher'],
        'var': ['Platform', 'Name', 'Genre', 'Developer', 'Year_of_Release'],
        'agg': ['nunique']
    }, {
        'key': ['Developer'],
        'var': ['Platform', 'Name', 'Genre', 'Year_of_Release'],
        'agg': ['nunique']
    }, {
        'key': ['Publisher', 'Name'],
        'var': ['Platform', 'Genre', 'Developer', 'Year_of_Release'],
        'agg': ['nunique']
    }
    ]

    original_cols = train_test.columns
    groupby = GroupbyTransformer(param_dict=nunique_dict)
    train_test = groupby.transform(train_test)
    groupby = GroupbyTransformer(param_dict=groupby_dict)
    train_test = groupby.transform(train_test)
    diff = DiffGroupbyTransformer(param_dict=groupby_dict)
    train_test = diff.transform(train_test)
    ratio = RatioGroupbyTransformer(param_dict=groupby_dict)
    train_test = ratio.transform(train_test)
    train_test[list(set(train_test.columns) - set(original_cols))].to_feather('../input/feather/aggregation.ftr')
