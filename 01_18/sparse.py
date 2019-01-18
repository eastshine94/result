import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import namedtuple
sparse = namedtuple("sparse_vector", ["size", "indices", "values"])
    
def sparse_encode(sparse_matrix) :

    data = sparse_matrix.data
    indices = sparse_matrix.indices
    indptr = sparse_matrix.indptr
    col_len = sparse_matrix.shape[1]
    row_len = sparse_matrix.shape[0]

    indices_list = []
    value_list = []
    for i in range(len(indptr) - 1) :
        col = []
        value = []
        for j in range(indptr[i],indptr[i + 1]) :
            col.append(indices[j])
            value.append(data[j] * 1.0)
        indices_list.append(col)
        value_list.append(value)
    size= [col_len for _ in range(row_len)]
    
    sparse_vectors = [str(sparse(size = size[i], indices= indices_list[i], values = value_list[i])) for i in range(row_len)] 
    
    
    return {'sparse_vectors' : sparse_vectors}


def _str_decode(string):
    #input string => sparse_vector(size = 278590, indices = [248981, 268248, 101083, 204774], values = [1.0, 1.0, 1.0, 1.0])
    input_string = string.replace(" ", "")

    size_index = input_string.find("=") + 1
    indices_index = input_string.find("[") + 1
    values_index = input_string.find("[", indices_index) + 1

    size = input_string[size_index:input_string.find(",")]  
    size = int(size)
    
    indices = input_string[indices_index: input_string.find("]", indices_index)]
    if indices != '' :
        indices = indices.split(',')
        indices = [int(i) for i in indices]


    values = input_string[values_index: input_string.find("]", values_index)]
    if values != '' :
        values = values.split(',')
        values = [float(i) for i in values]
    
    return {'size': size, 'indices': indices, 'values': values}


def sparse_decode(table, sparse_vector_col) :
    input_table = table.copy()
    sparse_vector = input_table[sparse_vector_col]
    row_size = len(input_table)
    col_size = _str_decode(sparse_vector.tolist()[0])['size']
    shape  = (row_size,col_size)
    
    indices = []
    values = []
    row_list = []

    for row, vector in enumerate(sparse_vector) : 
        str_to_vector = _str_decode(vector)
        extract_indices = str_to_vector['indices'] 
        extract_values = str_to_vector['values']
        for indice,value in zip(extract_indices,extract_values) : 
            row_list.append(row)
            values.append(value)
            indices.append(indice)

    
    sparse_matrix = csr_matrix((values, (row_list,indices)), shape=shape)
    
    return {'sparse_matrix':sparse_matrix}