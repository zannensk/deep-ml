import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    test_indices = []
    train_indices = []
    classes = np.unique(y)

    for cls in classes:
        class_indices = np.where(y == cls)[0]
        np.random.shuffle(class_indices)
        n_test = int(len(class_indices) * test_size)

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

