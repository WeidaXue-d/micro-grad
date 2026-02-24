class Value:
   
    def __init__(self, data, _children=(), _op=''):
        self.data = data      
        self.grad = 0.0 

        self._backward = lambda: None      
        
        self._prev = set(_children)  
        self._op = _op               

    def __repr__(self):
        
        return f"Value(data={self.data})"
    
    def __add__(self, other):
       
        out_data = self.data + other.data
        out = Value (out_data,(self,other),'+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        out_data=self.data * other.data

        out = Value (out_data,(self,other),'*')

        return out



