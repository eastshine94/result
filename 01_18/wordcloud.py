
# coding: utf-8

from brightics.common.report import ReportBuilder, strip_margin, plt2MD, png2MD
from brightics.brightics_data_api import _png2uri
from wordcloud import WordCloud
from collections import Counter
import io

def generate_wordcloud(table, input_col, width=640, height=480, background_color="white", max_font_size=None):
    font_path = './brightics/function/text_analytics/fonts/NanumGothic.ttf' # todo
    
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
    
    rb = ReportBuilder()
    rb.addMD(strip_margin("""
    | ## Word Cloud Result
    | {fig}
    """.format(fig=fig_wordcloud)))

    result = dict()
    result['report'] = rb.get()

    return {'result': result}
