import category_encoders as ce
import pandas as pd

# Categorical Encoding
def encode_categorical(df, encoding='label', cat_cols=None, y_col=None, enc_params={}):
    """ df is a dataframe may work of categorical features only
    Args:
        - encoding: 'label' or 'one-hot' or ... (implement others)
        - cat_cols: name of categorical columns, default of None means all columns are categorical
    """
    cat_cols = list(df.columns) if cat_cols is None else cat_cols
    assert df.select_dtypes(include=['category', 'object']).shape[1] == len(cat_cols)
    
    available_encodings = {
        'back_diff_contrast': ce.BackwardDifferenceEncoder(enc_params),
        'base_n': ce.BaseNEncoder(enc_params),
        'binary': ce.BinaryEncoder(enc_params),
        'count': ce.CountEncoder(enc_params),
        'hashing': ce.HashingEncoder(enc_params),
        'helmert_contrast': ce.HelmertEncoder(enc_params),
        'label': ce.OrdinalEncoder(enc_params),
        'ordinal': ce.OrdinalEncoder(enc_params),
        'one_hot': ce.OneHotEncoder(enc_params),
        'poly': ce.PolynomialEncoder(enc_params),
        'sum_contrast': ce.SumEncoder(enc_params),
        'catboost': ce.CatBoostEncoder(enc_params), 
        'glmm': ce.GLMMEncoder(enc_params), 
        'jse': ce.JamesSteinEncoder(enc_params), 
        'leave_one_out': ce.LeaveOneOutEncoder(enc_params), 
        'm_est': ce.MEstimateEncoder(enc_params), 
        'target': ce.TargetEncoder(enc_params),
        'woe': ce.WOEEncoder(enc_params)
    }
    
    supervised_encoding = ['catboost', 'glmm', 'jse', 'leave_one_out', 'm_est', 'target', 'woe']
    
    enc = available_encodings.get(encoding, None)
    
    if enc is None:
        raise Exception(f"{encoding} encoding is invalid or unimplemented. Please use one of these as the encoding parameters: {list(available_encodings.keys())}")
    
    if encoding not in supervised_encoding:
        enc.fit(df[cat_cols])
    else:
        if y_col is None:
            raise Exception(f"{encoding} is a supervised encoding method. Please provide y_col.")
        enc.fit(df[cat_cols], df[y_col])
    
    enc_df = enc.transform(df[cat_cols])
    
    # merge continuous columns of df with enc_df
    fin_df = pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)
    
    # return enc too to help decode later or use on test data if needed
    return fin_df, enc


def encode_categorical_test(df, encoder, cat_cols=None):
    if cat_cols is None:
        assert df.select_dtypes(include=['category']).shape[1] == df.shape[1]
        return encoder.transform(df)
    else:
        enc_df = encoder.transform(df[cat_cols])
        return pd.concat([df.drop(columns=cat_cols), enc_df], axis=1)