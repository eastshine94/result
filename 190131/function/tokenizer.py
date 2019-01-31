from twkorean import TwitterKoreanProcessor
import pandas as pd

from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters

TWITTER_MORPHEMES = ["Noun", "Verb", "Adjective", "Determiner", "Adverb", "Conjunction",
                    "Exclamation", "Josa", "PreEomi", "Eomi", "Suffix", "Punctuation",
                    "Foreign", "Alpha", "Number", "Unknown", "KoreanParticle", "Hashtag",
                    "ScreenName", "Email", "URL"]


def _split_token(doc, processor, morphemes):
    token_list = []
    pos_list = []

    korean_token_list = processor.tokenize(doc)
    for token in korean_token_list:
        if token.pos in morphemes:
            token_list.append(token.text)
            pos_list.append(token.pos)

    return [token_list, pos_list]



def twitter_tokenizer(table, group_by=None, **params):
    check_required_parameters(_twitter_tokenizer, params, ['table'])
    if group_by is not None:
        grouped_model = _function_by_group(_twitter_tokenizer, table, group_by=group_by, **params)
        return grouped_model
    else:
        return _twitter_tokenizer(table, **params)


def _twitter_tokenizer(table,
                      input_col,
                      token_col_name='tokens',
                      pos_col_name='pos',
                      stemming=False,
                      normalization=False,
                      morphemes=None):

    if morphemes is None:
        morphemes = TWITTER_MORPHEMES

    processor = TwitterKoreanProcessor(stemming=stemming, normalization=normalization)
    out_table = table.copy()
    document = out_table[input_col]

    token_series = document.apply(lambda _:
                                  _split_token(_, processor=processor, morphemes=morphemes))
    out_table[[token_col_name, pos_col_name]] = pd.DataFrame(data=token_series.tolist())

    return  {'out_table' : out_table}
