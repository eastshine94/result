import io
from collections import Counter

import pandas as pd
from wordcloud import WordCloud

from brightics.common.repr import BrtcReprBuilder
from brightics.common.repr import strip_margin
from brightics.common.repr import pandasDF2MD
from brightics.common.repr import png2MD
from brightics.common.groupby import _function_by_group
from brightics.common.utils import check_required_parameters


def generate_wordcloud(table, group_by=None, **params):
    check_required_parameters(_generate_wordcloud, params, ['table'])
    if group_by is not None:
        grouped_model = _function_by_group(_generate_wordcloud, table, group_by=group_by, **params)
        return grouped_model
    else:
        return _generate_wordcloud(table, **params)


def _generate_wordcloud(table, input_col, width=640, height=480, background_color="white", max_font_size=None):
    font_path = './brightics/function/text_analytics/fonts/NanumGothic.ttf'  # todo

    counter = Counter()
    table[input_col].apply(counter.update)

    wordcloud = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color=background_color
        )
    wordcloud.generate_from_frequencies(dict(counter), max_font_size)

    img_bytes = io.BytesIO()
    wordcloud.to_image().save(img_bytes, format='PNG')
    fig_wordcloud = png2MD(img_bytes.getvalue())

    word_count_data = [[word, count] for word, count in counter.items()]
    word_count_table = pd.DataFrame(data=word_count_data, columns=['word', 'count'])
    word_count_table = word_count_table.sort_values(["count"], ascending=[False])

    rb = BrtcReprBuilder()
    rb.addMD(strip_margin("""
    | ## Word Cloud Result
    | ### Word Cloud
    | {fig_wordcloud}
    |
    | ### Word Counts
    | {table}
    """.format(fig_wordcloud=fig_wordcloud, table=pandasDF2MD(word_count_table))))

    result = dict()
    result['word_counts'] = word_count_table
    result['_repr_brtc_'] = rb.get()

    return {'result': result}
