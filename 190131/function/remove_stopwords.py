def remove_stopwords(table, input_col, stopwords, output_col_name='removed'):
    out_table = table.copy()
    
    def _remove_stopwords(str_list):
        return [_ for _ in str_list if _ not in stopwords]
    
    out_table[output_col_name] = table[input_col].apply(_remove_stopwords)
    
    return {'out_table': out_table}