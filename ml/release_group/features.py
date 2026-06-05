"""Sparse feature pipeline for release group KNN (from note_book_guillaume.ipynb)."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer


class ListToSparseTransformer(BaseEstimator, TransformerMixin):
    """
    Impute scalar columns and multi-label binarize list columns into one sparse matrix.
    """

    def __init__(self, categorical_cols, numeric_mean_cols, list_cols):
        self.categorical_cols = categorical_cols
        self.numeric_mean_cols = numeric_mean_cols
        self.list_cols = list_cols
        self.imputers: dict = {}
        self.mlbs: dict = {}

    def fit(self, X, y=None):
        if self.categorical_cols:
            self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
            self.imputers["categorical"].fit(X[self.categorical_cols])

        if self.numeric_mean_cols:
            self.imputers["numeric"] = SimpleImputer(strategy="mean")
            self.imputers["numeric"].fit(X[self.numeric_mean_cols])

        for col in self.list_cols:
            self.mlbs[col] = MultiLabelBinarizer(sparse_output=True)
            self.mlbs[col].fit(X[col].apply(self._ensure_list))

        return self

    def transform(self, X):
        X = X.copy()

        if self.categorical_cols:
            X[self.categorical_cols] = self.imputers["categorical"].transform(
                X[self.categorical_cols]
            )

        if self.numeric_mean_cols:
            X[self.numeric_mean_cols] = self.imputers["numeric"].transform(
                X[self.numeric_mean_cols]
            )

        X_scalar = X[self.categorical_cols + self.numeric_mean_cols].to_numpy(dtype=np.float32)

        list_feature_matrices = []
        for col in self.list_cols:
            X_col = self.mlbs[col].transform(X[col].apply(self._ensure_list))
            list_feature_matrices.append(X_col)

        matrices = [sparse.csr_matrix(X_scalar)] + list_feature_matrices
        return sparse.hstack(matrices, format="csr")

    @staticmethod
    def _ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        return [x]
