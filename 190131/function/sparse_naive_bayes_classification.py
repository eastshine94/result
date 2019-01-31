import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import preprocessing
import itertools
from sklearn.naive_bayes import MultinomialNB
from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.repr import plt2MD
from brightics.common.repr import pandasDF2MD
from brightics.common.repr import dict2MD
from brightics.function.utils import _model_dict
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters
from brightics.common.datatypes.vector import sparse_vector_json_list_to_csr_matrix

def sparse_naive_bayes_train(table, group_by=None, **params):
    check_required_parameters(_sparse_naive_bayes_train, params, ['table'])
    if group_by is not None:
        return _function_by_group(_sparse_naive_bayes_train, table, group_by=group_by, **params)
    else:
        return _sparse_naive_bayes_train(table, **params)


def _sparse_naive_bayes_train(table, sparse_vector_col, label_col, alpha=1.0, fit_prior=True, class_prior=None):

    sparse_vector_json_series = table[sparse_vector_col]
    input_csr_matrix = sparse_vector_json_list_to_csr_matrix(sparse_vector_json_series.values)
    label = table[label_col]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(label)
    label_correspond = label_encoder.transform(label)
    
    if class_prior is not None:
        tmp_class_prior = [0] * len(class_prior)
        for elems in class_prior: 
            tmp = elems.split(":")
            tmp_class_prior[label_encoder.transform([tmp[0]])[0]] = float(tmp[1])
        class_prior = tmp_class_prior
    
    nb_model = MultinomialNB(alpha, fit_prior, class_prior)
    nb_model.fit(input_csr_matrix, label_correspond)

    prediction_correspond = nb_model.predict(input_csr_matrix)
        
    get_param = dict()
    get_param['Lambda'] = alpha
    # get_param['Prior Probabilities of the Classes'] = class_prior
    get_param['Fit Class Prior Probability'] = fit_prior
    get_param['sparse_vector_col'] = sparse_vector_col
    get_param['Label Column'] = label_col

    cnf_matrix = confusion_matrix(label_correspond, prediction_correspond)
    
    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes=label_encoder.classes_,
                      title='Confusion Matrix')
    fig_confusion_matrix = plt2MD(plt)
    accuracy = nb_model.score(input_csr_matrix, label_correspond) * 100
    
    rb = BrtcReprBuilder()
    rb.addMD(strip_margin("""
    | ## Naive Bayes Classification Result
    | 
    | ### Parameters
    | {table_parameter} 
    | ### Predicted vs Actual
    | {image1}
    | #### Accuacy = {accuracy}%
    |
    """.format(image1=fig_confusion_matrix, accuracy=accuracy, table_parameter=dict2MD(get_param))))
    
    model = _model_dict('naive_bayes_model')
    model['sparse_vector_col'] = sparse_vector_col
    model['label_col'] = label_col
    model['label_encoder'] = label_encoder
    model['nb_model'] = nb_model
    model['_repr_brtc_'] = rb.get()
    
    return {'model' : model}


def _plot_confusion_matrix(cm, classes,
                          title='Confusion matrix'):

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.tight_layout()


def sparse_naive_bayes_predict(table, model, **params):
    check_required_parameters(_sparse_naive_bayes_predict, params, ['table', 'model'])
    if '_grouped_data' in model:
        return _function_by_group(_sparse_naive_bayes_predict, table, model, **params)
    else:
        return _sparse_naive_bayes_predict(table, model, **params)


def _sparse_naive_bayes_predict(table, model, suffix, display_log_prob=False, prediction_col='prediction', prob_prefix='probability', log_prob_prefix='log_probability'):
    sparse_vector_col = model['sparse_vector_col']
    sparse_vector_json_series = table[sparse_vector_col]
    input_csr_matrix = sparse_vector_json_list_to_csr_matrix(sparse_vector_json_series)
    
    nb_model = model['nb_model']
    label_encoder = model['label_encoder']
    
    prediction_correspond = nb_model.predict(input_csr_matrix)
    prediction = label_encoder.inverse_transform(prediction_correspond)
    
    if(suffix == 'label'):
        suffixes = label_encoder.classes_
    else:
        suffixes = range(0, len(label_encoder.classes_)) 

    prob = nb_model.predict_proba(input_csr_matrix)    
    prob_cols = ['{prefix}_{suffix}'.format(prefix=prob_prefix, suffix=suffix) for suffix in suffixes]
    prob_df = pd.DataFrame(data=prob, columns=prob_cols)
    
    result = table
    result[prediction_col] = prediction
    
    if display_log_prob == True:
        log_prob = nb_model.predict_log_proba(input_csr_matrix)
        logprob_cols = ['{prefix}_{suffix}'.format(prefix=log_prob_prefix, suffix=suffix) for suffix in suffixes]
        logprob_df = pd.DataFrame(data=log_prob, columns=logprob_cols)
        result = pd.concat([result, prob_df, logprob_df], axis=1)
    else:
        result = pd.concat([result, prob_df], axis=1)

    return {'out_table' : result}

