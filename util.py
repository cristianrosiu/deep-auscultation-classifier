from sklearn.model_selection import StratifiedKFold

def kfold_split(X, y, n_splits=5, seed=42):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return kfold.split(X, y)
    