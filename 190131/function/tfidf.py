import pandas as pd
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from brightics.function.utils import _model_dict
from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.repr import pandasDF2MD
from brightics.common.datatypes.vector import csr_matrix_to_sparse_vector_json_list
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters


def tfidf_gensim(table, group_by=None, **params):
    check_required_parameters(_tfidf_gensim, params, ['table'])
    if group_by is not None:
        grouped_model = _function_by_group(_tfidf_gensim, table, group_by=group_by, **params)
        return grouped_model
    else:
        return _tfidf_gensim(table, **params)


def _tfidf_gensim(table,
                input_col,
                output_col_name="sparse_vectors",
                tf_weighing='n',
                df_weighing='t',
                document_normalization='c'):

    out_table = table.copy()
    tokens = out_table[input_col]
    smartirs = tf_weighing + df_weighing + document_normalization

    dictionary = Dictionary(tokens)
    word_count_vector_list = [dictionary.doc2bow(text) for text in tokens]

    tfidf_model = TfidfModel(word_count_vector_list, smartirs=smartirs)
    tfidf_vector_list = [*tfidf_model[word_count_vector_list]]

    sparse_matrix = corpus2csc(tfidf_vector_list, num_terms=len(dictionary.token2id)).T

    rb = BrtcReprBuilder()

    dictionary_data = [
        [index, word, tfidf_model.dfs[index], tfidf_model.idfs[index]] 
        for index, word in dictionary.items()
    ]
    dictionary_table = pd.DataFrame(data=dictionary_data, columns=['index', 'word', 'count', 'idf'])
    dictionary_table = dictionary_table.sort_values(["count"], ascending=[False])

    rb.addMD(strip_margin("""
    | ## TFIDF Result
    | ### Dictionary
    | {table1}
    """.format(table1=pandasDF2MD(dictionary_table)
               )))

    out_table[output_col_name] = csr_matrix_to_sparse_vector_json_list(sparse_matrix)

    model = _model_dict('tfidf_model')
    model['dictionary_table'] = dictionary_table
    model['dictionary'] = dictionary
    model['tfidf_model'] = tfidf_model
    model['input_col'] = input_col
    model['output_col_name'] = output_col_name
    model['_repr_brtc_'] = rb.get()

    return {'out_table': out_table, 'model': model}


def tfidf_model_gensim(table, model, **params):
    check_required_parameters(_tfidf_model_gensim, params, ['table', 'model'])
    if '_grouped_data' in model:
        return _function_by_group(_tfidf_model_gensim, table, model, **params)
    else:
        return _tfidf_model_gensim(table, model, **params)


def _tfidf_model_gensim(table, model, output_col_name=None):
    if output_col_name is None:
        output_col_name = model['output_col_name']

    out_table = table.copy()

    token_col = model['input_col']
    dictionary = model['dictionary']
    tfidf_model = model['tfidf_model']

    tokens = out_table[token_col]

    word_count_vector_list = [dictionary.doc2bow(text) for text in tokens]
    tfidf_vector_list = [*tfidf_model[word_count_vector_list]]
    sparse_matrix = corpus2csc(tfidf_vector_list, num_terms=len(dictionary.token2id)).T

    out_table[output_col_name] = csr_matrix_to_sparse_vector_json_list(sparse_matrix)

    return {'out_table': out_table }
