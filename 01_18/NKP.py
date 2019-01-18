from brightics.common.report import ReportBuilder, strip_margin, pandasDF2MD
from brightics.function.utils import _model_dict
from twkorean import TwitterKoreanProcessor

import pandas as pd
import numpy as np
import re


def twitter_tokenizer(table, input_col, token_col_name = 'tokens', pos_col_name = 'pos',stemming=False, normalization=False, morpheme=None) :
    processor = TwitterKoreanProcessor(stemming=stemming, normalization=normalization)
    out_table = table.copy()
    tokens_col_data = []
    pos_col_data = []
    for i in out_table.index :
        try:
            sentence = out_table.at[i,input_col]
            tokenize = processor.tokenize(sentence)
            tokens_list = []
            pos_list = []
            for token in tokenize:
                if(morpheme is None or token.pos in morpheme):
                    tokens_list.append(token.text)
                    pos_list.append(token.pos)

            if (tokens_list == []) :
                out_table.drop(i,inplace=True)
            else :
                tokens_col_data.append(tokens_list)
                pos_col_data.append(pos_list)
        except:
            out_table.drop(i,inplace=True)
    out_table[token_col_name] = tokens_col_data
    out_table[pos_col_name] = pos_col_data

    return {'out_table': out_table} 





def remove_stopwords(table, input_col, stopwords, output_col_name='removed'):
    out_table = table.copy()
    
    def _remove_stopwords(str_list):
        return [_ for _ in str_list if _ not in stopwords]
    
    out_table[output_col_name] = table[input_col].apply(_remove_stopwords)
    
    return {'out_table': out_table}