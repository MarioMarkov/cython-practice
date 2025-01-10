import cython
import numpy as np

cimport numpy as cnp
cnp.import_array()


@cython.cclass
class Tensor:
    
    data: cnp.ndarray[cnp.float64_t]
    shape: cnp.ndarray[cnp.int32_t]
    strides: cnp.ndarray[cnp.int32_t] 
    ndim: cython.int 
    
    @property
    def data(self):
        return self.data

    @property
    def shape(self):
        return self.shape
    
    @property
    def strides(self):
        return self.strides

    def __cinit__(self, cnp.ndarray[cnp.float64_t] data, cnp.ndarray[cnp.int32_t] shape , ndim: cython.int ):
        self.data = data
        self.shape = shape
        self.ndim = ndim

        self.strides: cnp.ndarray[cnp.int32_t] = np.empty(ndim, dtype=np.int32)
        stride = 1 
        for i in range(ndim - 1, -1,-1 ):
            self.strides[i] = stride
            stride *= shape[i]

        
    
    def __getitem__(self, indices:tuple):
        idx: cython.int
        i: cython.int
        idx: cnp.ndarray[cnp.int32_t] = np.array(indices, dtype=np.int32)


        for i in range(self.ndim):
            idx += indices[i] * self.strides[i]
        return self.data[idx]
        