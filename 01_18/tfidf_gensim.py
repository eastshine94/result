
import numpy as np
import pandas as pd
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from brightics.function.sparse import sparse_encode
from brightics.common.report import ReportBuilder, strip_margin, pandasDF2MD, plt2MD

def tfidf_train(table, tokens_col, tf_weighing='n', df_weighing='t', document_normalization = 'c'):
    
    out_table = table.copy()   
    _corpus = out_table[tokens_col]
    _smartirs = tf_weighing + df_weighing + document_normalization
    
    _dictionary = Dictionary(_corpus)
    _corpus = [_dictionary.doc2bow(text) for text in _corpus]
    
    _model = TfidfModel(_corpus, smartirs=_smartirs)
    _corpus = [text for text in _model[_corpus]]
    
    _sparse_matrix = corpus2csc(_corpus, num_terms=len(_dictionary.token2id)).T

    _values = [value for value in _dictionary.values()]
    _keys = [key for key in _dictionary.keys()]
    _dic = pd.DataFrame({'indice':_keys, 'word': _values})
    rb = ReportBuilder()
    rb.addMD(strip_margin("""
    | ## Dictionary
    | {table1}
    """.format(table1=pandasDF2MD(_dic)
               )))

    out_table['sparse_vectors'] = sparse_encode(_sparse_matrix)['sparse_vectors']

    fit_model = dict()
    fit_model['dictionary'] = _dictionary
    fit_model['model'] = _model
    fit_model['report'] =  rb.get()
    return {'out_table': out_table, 'fit_model': fit_model}

def tfidf_test(table, fit_model, tokens_col):
    out_table = table.copy()
    _dictionary = fit_model['dictionary']
    _model = fit_model['model']
    
    
    _corpus = out_table[tokens_col]
    _corpus = [_dictionary.doc2bow(text) for text in _corpus]
    _corpus = [text for text in _model[_corpus]]
    _sparse_matrix = corpus2csc(_corpus, num_terms=len(_dictionary.token2id)).T
    
    out_table['sparse_vectors'] = sparse_encode(_sparse_matrix)['sparse_vectors']
    

    
    return {'out_table': out_table }