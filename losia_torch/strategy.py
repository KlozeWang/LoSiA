import torch

class PureGreedy:
    def __init__(self, keys_shape):
        self.keys_shape = keys_shape
    
    def importance_locate(self, q):
        sums = torch.sum(q, dim = 1)
        rows = torch.topk(sums, self.keys_shape[0]).indices
        torch.sort(rows)
        
        sums = torch.sum(q, dim = 0)
        cols = torch.topk(sums, self.keys_shape[1]).indices
        torch.sort(cols)
        return rows, cols

class Row2Column:
    def __init__(self, keys_shape):
        self.keys_shape = keys_shape
    
    def importance_locate(self, q):
        sums = torch.sum(q, dim = 1)
        rows = torch.topk(sums, self.keys_shape[0]).indices
        torch.sort(rows)

        sums = torch.sum(q[rows,:], dim = 0)
        cols = torch.topk(sums, self.keys_shape[1]).indices
        torch.sort(cols)
        return rows, cols

class Column2Row:
    def __init__(self, keys_shape):
        self.keys_shape = keys_shape

    
    def importance_locate(self, q):
        sums = torch.sum(q, dim = 0)
        cols = torch.topk(sums, self.keys_shape[1]).indices
        torch.sort(cols)

        sums = torch.sum(q[:,cols], dim = 1)
        rows = torch.topk(sums, self.keys_shape[0]).indices
        torch.sort(rows)
        return rows, cols