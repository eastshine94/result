import json

from scipy.sparse import csr_matrix


class SparseVector(object):

    def __init__(self, size, indices, values):
        self.size = size
        self.indices = indices
        self.values = values

    def to_json(self):
        return json.dumps(
            {'size':self.size,
             'indices': self.indices,
             'values': self.values}
        )

    @classmethod
    def from_json(cls, json_str):
        dict_ = json.loads(json_str)
        size = dict_['size']
        indices = dict_['indices']
        values = dict_['values']
        return cls(size, indices, values)


def csr_matrix_to_sparse_vector_list(sparse_matrix):

    data = sparse_matrix.data
    csr_matrix_indices = sparse_matrix.indices
    csr_matrix_indptr = sparse_matrix.indptr
    csr_matrix_row_len = sparse_matrix.shape[0]
    csr_matrix_col_len = sparse_matrix.shape[1]

    sparse_vector_list = []

    for i in range(csr_matrix_row_len) :
        indices = []
        values = []

        for j in range(csr_matrix_indptr[i], csr_matrix_indptr[i + 1]):
            indices.append(int(csr_matrix_indices[j]))  # int64 to int32
            values.append(data[j] * 1.0)  # to float

        sparse_vector = SparseVector(csr_matrix_col_len, indices, values)
        sparse_vector_list.append(sparse_vector)

    return sparse_vector_list


def csr_matrix_to_sparse_vector_json_list(sparse_matrix):
    sparse_vector_list = csr_matrix_to_sparse_vector_list(sparse_matrix)
    return [
        sparse_vector.to_json() for sparse_vector in sparse_vector_list
    ]


def _get_size(sparse_vector_list):
    return sparse_vector_list[0].size
    

def sparse_vector_list_to_csr_matrix(sparse_vector_list):
    row_size = len(sparse_vector_list)
    col_size = _get_size(sparse_vector_list)
    shape = (row_size, col_size)

    indices = []
    values = []
    row_list = []

    for row, sparse_vector in enumerate(sparse_vector_list):
        indices.extend(sparse_vector.indices)
        values.extend(sparse_vector.values)
        row_list.extend([row] * len(sparse_vector.values))

    sparse_matrix = csr_matrix((values, (row_list, indices)), shape=shape)

    return sparse_matrix


def sparse_vector_json_list_to_csr_matrix(sparse_vector_json_list):
    sparse_vector_list = [ 
        SparseVector.from_json(sparse_vector_json) 
        for sparse_vector_json in sparse_vector_json_list
    ]
    return sparse_vector_list_to_csr_matrix(sparse_vector_list)
