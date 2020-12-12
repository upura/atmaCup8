import re

import nltk
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, _document_frequency
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils.validation import check_is_fitted


def analyzer_bow_en(text):
    sb = nltk.stem.snowball.SnowballStemmer('english')
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower()     # 小文字化
    text = text.replace('\n', '')   # 改行削除
    text = text.replace('\t', '')   # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    text = text.split(' ')  # スペースで区切る
    text = [sb.stem(t) for t in text]

    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None):   # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if word in stop_words:  # ストップワードに含まれるものは除外
            continue
        if len(word) < 2:   # 1文字、0文字（空文字）は除外
            continue
        words.append(word)

    return ' '.join(words)


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b  : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        """
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features] document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        doc_len = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # In each row, repeat `doc_len` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(doc_len), sz)

        # Compute BM25 score only for non-zero elements
        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X


def get_tfidf(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict, option: dict):
    """
    col_definition: text_col
    option: n_components, lang={'ja', 'en'}
    """
    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    vectorizer = make_pipeline(
        TfidfVectorizer(),
        make_union(
            TruncatedSVD(n_components=option['n_components'], random_state=7),
            NMF(n_components=option['n_components'], random_state=7),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=option['n_components'], random_state=7)
            ),
            n_jobs=1,
        ),
    )

    if option['lang'] == 'en':
        X = [analyzer_bow_en(text) for text in train[col_definition['text_col']].fillna('')]
    else:
        raise ValueError
    X = vectorizer.fit_transform(X).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'tfidf_svd_{}'.format(i) for i in range(option['n_components'])] + [
        'tfidf_nmf_{}'.format(i) for i in range(option['n_components'])] + [
        'tfidf_bm25_{}'.format(i) for i in range(option['n_components'])])
    train = pd.concat([train, X], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


def get_count(train: pd.DataFrame, test: pd.DataFrame, col_definition: dict, option: dict):
    """
    col_definition: text_col
    option: n_components, lang={'ja', 'en'}
    """

    n_train = len(train)
    train = pd.concat([train, test], sort=False).reset_index(drop=True)
    vectorizer = make_pipeline(
        CountVectorizer(min_df=2, max_features=20000,
                        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), stop_words='english'),
        make_union(
            TruncatedSVD(n_components=option['n_components'], random_state=7),
            NMF(n_components=option['n_components'], random_state=7),
            make_pipeline(
                BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                TruncatedSVD(n_components=option['n_components'], random_state=7)
            ),
            n_jobs=1,
        ),
    )

    if option['lang'] == 'en':
        X = [analyzer_bow_en(text) for text in train[col_definition['text_col']].fillna('')]
    else:
        raise ValueError
    X = vectorizer.fit_transform(X).astype(np.float32)
    X = pd.DataFrame(X, columns=[
        'count_svd_{}'.format(i) for i in range(option['n_components'])] + [
        'count_nmf_{}'.format(i) for i in range(option['n_components'])] + [
        'count_bm25_{}'.format(i) for i in range(option['n_components'])])
    train = pd.concat([train, X], axis=1)
    test = train[n_train:].reset_index(drop=True)
    train = train[:n_train]
    return train, test


if __name__ == '__main__':
    train = pd.read_csv('../input/atmacup08-dataset/train.csv')
    test = pd.read_csv('../input/atmacup08-dataset/test.csv')
    print(train.shape, test.shape)
    # (8359, 16) (8360, 11)
    target_col = 'Global_Sales'

    train, test = get_tfidf(train,
                            test,
                            col_definition={'text_col': 'Name'},
                            option={'n_components': 5, 'lang': 'en'})
    print(train.shape, test.shape)

    train, test = get_count(train,
                            test,
                            col_definition={'text_col': 'Name'},
                            option={'n_components': 5, 'lang': 'en'})
    print(train.shape, test.shape)
    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test.iloc[:, 16:].to_feather('../input/feather/texts.ftr')
