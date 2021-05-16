from sklearn.model_selection import KFold


def kfold_split(X, y, y1, n_splits=5, seed=42):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold.split(X, y, y1)
