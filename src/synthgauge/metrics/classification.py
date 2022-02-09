from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score, recall_score, f1_score


def classification_comparison(real, synth, key, target, sklearn_classifier,
                              random_state=None, test_prop=0.2, **kwargs):
    """Classification utility metric

    This metric fits two classification models to `real` and `synth`
    respectively, and then tests them both against withheld `real` data. We
    obtain utility scores by subtracting the precision, recall and f1 scores
    of the predictions obtained by the synth model from those obtained by the
    real model.

    Parameters
    ----------
    real : pandas dataframe
        Dataframe containing the real data.
    synth : pandas dataframe
        Dataframe containing the synthetic data.
    key : list of strings
        list of column names to use as the input in the classification.
    target : str
        column to use as target in the classification.
    sklearn_classifier : scikit-learn estimator
        classifier with fit and predict methods.
    random_state : int, RandomState instance or None, default=42
        Controls the shuffling steps during the train-test split and the
        classification algorithm itself. Pass an int for reproducible output
        across multiple function calls.
    test_prop : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    Returns
    -------
    ClassificationResult : namedtuple
        precision_difference : float
            precision of model trained on real data subtracted by precision of
            model trained on synthetic data.
        recall_difference : float
            recall of model trained on real data subtracted by recall of
            model trained on synthetic data.
        f1_difference : float
            f1 score of model trained on real data subtracted by f1 score of
            model trained on synthetic data.

    Notes
    -----
    Some preprocessing is carried out before the models are trained. Numeric
    features are scaled and categorical features are one-hot-encoded.

    A score of zero tells us the synthetic data is just as good as the real at
    training classifier models. Increases in these scores indicate poorer
    utility.
    """
    real_X = real[key]
    real_y = real[target]

    synth_X = synth[key]
    synth_y = synth[target]

    # split data
    real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(
        real_X, real_y, test_size=test_prop, random_state=random_state)
    synth_X_train, _, synth_y_train, _ = train_test_split(
        synth_X, synth_y, test_size=test_prop, random_state=random_state)

    # preprocessing
    # scale numeric
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    # OHE categorical
    categor_transformer = Pipeline(steps=[('encoder', OneHotEncoder())])

    num_feats = real[key].select_dtypes(include='number').columns
    cat_feats = real[key].select_dtypes(exclude='number').columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, num_feats),
            ('categor', categor_transformer, cat_feats)
        ]
    )
    # train real model
    real_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', sklearn_classifier(random_state=random_state, **kwargs))
    ])

    real_pipeline.fit(real_X_train, real_y_train.values.ravel())
    # test real on real
    y_real_predicts_real = real_pipeline.predict(real_X_test)
    # train synth model
    synth_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', sklearn_classifier(random_state=random_state, **kwargs))
    ])

    synth_pipeline.fit(synth_X_train, synth_y_train.values.ravel())
    # test synth on real
    y_synth_predicts_real = synth_pipeline.predict(real_X_test)

    # compare results
    real_precision = precision_score(
        real_y_test, y_real_predicts_real, average='macro')
    real_recall = recall_score(
        real_y_test, y_real_predicts_real, average='macro')
    real_f1 = f1_score(real_y_test, y_real_predicts_real, average='macro')

    synth_precision = precision_score(
        real_y_test, y_synth_predicts_real, average='macro')
    synth_recall = recall_score(
        real_y_test, y_synth_predicts_real, average='macro')
    synth_f1 = f1_score(real_y_test, y_synth_predicts_real, average='macro')

    diff_precision = real_precision - synth_precision
    diff_recall = real_recall - synth_recall
    diff_f1 = real_f1 - synth_f1

    ClassificationResult = namedtuple('ClassificationResult',
                                      ('precision_difference',
                                       'recall_difference',
                                       'f1_difference'))

    return ClassificationResult(diff_precision, diff_recall, diff_f1)


if __name__ == '__main__':
    pass
