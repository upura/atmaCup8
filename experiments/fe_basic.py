import itertools
import warnings

from kaggler.preprocessing import TargetEncoder
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def count_null(train: pd.DataFrame, col_definition):
    train['count_null'] = train.isnull().sum(axis=1)
    for f in col_definition:
        if sum(train[f].isnull().astype(int)) > 0:
            train[f'cn_{f}'] = train[f].isnull().astype(int)
    return train.loc[:, train.columns.str.contains('cn_')]


def count_encoding(train: pd.DataFrame, col_definition):
    for f in col_definition:
        count_map = train[f].value_counts().to_dict()
        train[f'ce_{f}'] = train[f].map(count_map)
    return train.loc[:, train.columns.str.contains('ce_')]


def count_encoding_interact(train: pd.DataFrame, col_definition):
    for col1, col2 in tqdm(list(itertools.combinations(col_definition, 2))):
        col = col1 + '_' + col2
        _tmp = train[col1].astype(str) + "_" + train[col2].astype(str)
        count_map = _tmp.value_counts().to_dict()
        train[f'cei_{col}'] = _tmp.map(count_map)

    return train.loc[:, train.columns.str.contains('cei_')]


class CategoryVectorizer():
    def __init__(self, categorical_columns, n_components,
                 vectorizer=CountVectorizer(),
                 transformer=LatentDirichletAllocation(),
                 name='CountLDA'):
        self.categorical_columns = categorical_columns
        self.n_components = n_components
        self.vectorizer = vectorizer
        self.transformer = transformer
        self.name = name + str(self.n_components)

    def transform(self, dataframe):
        features = []
        for (col1, col2) in self.get_column_pairs():
            try:
                sentence = self.create_word_list(dataframe, col1, col2)
                sentence = self.vectorizer.fit_transform(sentence)
                feature = self.transformer.fit_transform(sentence)
                feature = self.get_feature(dataframe, col1, col2, feature, name=self.name)
                features.append(feature)
            except:
                pass
        features = pd.concat(features, axis=1)
        return features

    def create_word_list(self, dataframe, col1, col2):
        col1_size = int(dataframe[col1].values.max() + 1)
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(dataframe[col1].values, dataframe[col2].values):
            col2_list[int(val1)].append(col2 + str(val2))
        return [' '.join(map(str, ls)) for ls in col2_list]

    def get_feature(self, dataframe, col1, col2, latent_vector, name=''):
        features = np.zeros(shape=(len(dataframe), self.n_components), dtype=np.float32)
        self.columns = ['_'.join([name, col1, col2, str(i)]) for i in range(self.n_components)]
        for i, val1 in enumerate(dataframe[col1]):
            features[i, :self.n_components] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=self.columns)

    def get_column_pairs(self):
        return [(col1, col2) for col1, col2 in itertools.product(self.categorical_columns, repeat=2) if col1 != col2]

    def get_numerical_features(self):
        return self.columns


def matrix_factorization(train: pd.DataFrame, col_definition, option):
    """
    col_definition: encode_col
    option: n_components_lda, n_components_svd
    """

    cf = CategoryVectorizer(col_definition,
                            option['n_components_lda'],
                            vectorizer=CountVectorizer(),
                            transformer=LatentDirichletAllocation(n_components=option['n_components_lda'],
                                                                  n_jobs=-1, learning_method='online', random_state=777),
                            name='CountLDA')
    features_lda = cf.transform(train).astype(np.float32)

    cf = CategoryVectorizer(col_definition,
                            option['n_components_svd'],
                            vectorizer=CountVectorizer(),
                            transformer=TruncatedSVD(n_components=option['n_components_svd'], random_state=777),
                            name='CountSVD')
    features_svd = cf.transform(train).astype(np.float32)

    return features_svd, features_lda


class GroupbyTransformer():
    def __init__(self, param_dict=None):
        self.param_dict = param_dict

    def _get_params(self, p_dict):
        key = p_dict['key']
        if 'var' in p_dict.keys():
            var = p_dict['var']
        else:
            var = self.var
        if 'agg' in p_dict.keys():
            agg = p_dict['agg']
        else:
            agg = self.agg
        if 'on' in p_dict.keys():
            on = p_dict['on']
        else:
            on = key
        return key, var, agg, on

    def _aggregate(self, dataframe):
        self.features = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            all_features = list(set(key + var))
            new_features = self._get_feature_names(key, var, agg)
            features = dataframe[all_features].groupby(key)[
                var].agg(agg).reset_index()
            features.columns = key + new_features
            self.features.append(features)
        return self

    def _merge(self, dataframe, merge=True):
        for param_dict, features in zip(self.param_dict, self.features):
            key, var, agg, on = self._get_params(param_dict)
            if merge:
                dataframe = dataframe.merge(features, how='left', on=on)
            else:
                new_features = self._get_feature_names(key, var, agg)
                dataframe = pd.concat([dataframe, features[new_features]], axis=1)
        return dataframe

    def transform(self, dataframe):
        self._aggregate(dataframe)
        return self._merge(dataframe, merge=True)

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join([a, v, 'groupby'] + key) for v in var for a in _agg]

    def get_feature_names(self):
        self.feature_names = []
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            self.feature_names += self._get_feature_names(key, var, agg)
        return self.feature_names

    def get_numerical_features(self):
        return self.get_feature_names()


class DiffGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['diff', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[base_feature] - dataframe[v]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['diff', a, v, 'groupby'] + key) for v in var for a in _agg]


class RatioGroupbyTransformer(GroupbyTransformer):
    def _aggregate(self):
        raise NotImplementedError

    def _merge(self):
        raise NotImplementedError

    def transform(self, dataframe):
        for param_dict in self.param_dict:
            key, var, agg, on = self._get_params(param_dict)
            for a in agg:
                for v in var:
                    new_feature = '_'.join(['ratio', a, v, 'groupby'] + key)
                    base_feature = '_'.join([a, v, 'groupby'] + key)
                    dataframe[new_feature] = dataframe[v] / dataframe[base_feature]
        return dataframe

    def _get_feature_names(self, key, var, agg):
        _agg = []
        for a in agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return ['_'.join(['ratio', a, v, 'groupby'] + key) for v in var for a in _agg]


def target_encoding(train: pd.DataFrame, test: pd.DataFrame, encode_col, option: dict):
    warnings.simplefilter('ignore')

    te = TargetEncoder(cv=option['cv'])

    train_fe = te.fit_transform(train[encode_col], train[target_col])
    train_fe.columns = ['te_' + c for c in train_fe.columns]

    test_fe = te.transform(test[encode_col])
    test_fe.columns = ['te_' + c for c in test_fe.columns]

    return pd.concat([train_fe, test_fe]).reset_index(drop=True)


def aggregation(train: pd.DataFrame, col_definition: dict):
    """
    col_definition: groupby_dict, nunique_dict
    """
    print(train.shape)
    groupby = GroupbyTransformer(param_dict=col_definition['nunique_dict'])
    train = groupby.transform(train)
    print(train.shape)
    groupby = GroupbyTransformer(param_dict=col_definition['groupby_dict'])
    train = groupby.transform(train)
    print(train.shape)
    diff = DiffGroupbyTransformer(param_dict=col_definition['groupby_dict'])
    train = diff.transform(train)
    print(train.shape)
    ratio = RatioGroupbyTransformer(param_dict=col_definition['groupby_dict'])
    train = ratio.transform(train)

    return train


if __name__ == '__main__':
    train = pd.read_csv('../input/atmacup08-dataset/train.csv')
    test = pd.read_csv('../input/atmacup08-dataset/test.csv')
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
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)

    # label encoding
    for c in categorical_cols:
        le = preprocessing.LabelEncoder()
        train_test[c] = le.fit_transform(train_test[c].astype(str).fillna('unk').values)

    train_test['is_User_Score_tbd'] = (train_test['User_Score'] == 'tbd').astype(int)
    train_test['User_Score'] = train_test['User_Score'].replace('tbd', np.nan).astype(float)

    # target transformation
    train_test[target_col] = np.log1p(train_test[target_col])

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
    print(num_bins)
    _train[target_col] = pd.qcut(
        _train[target_col],
        num_bins,
        labels=False
    )
    target_encoding(_train, _test, categorical_cols, {'cv': cv}).to_feather('../input/feather/target_encoding.ftr')

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
    train_test = aggregation(train_test,
                             {'groupby_dict': groupby_dict,
                              'nunique_dict': nunique_dict})

    train_test[list(set(train_test.columns) - set(original_cols))].to_feather('../input/feather/aggregation.ftr')
