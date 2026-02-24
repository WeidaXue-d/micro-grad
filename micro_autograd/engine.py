class Value:
   
    def __init__(self, data, _children=(), _op=''):
        self.data = data      
        self.grad = 0.0 

        self._backward= lambda: None      
        
        self._prev = set(_children)  
        self._op = _op               

    def __repr__(self):
        
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        #calculate the sum of values
        out_data=self.data + other.data

        #create a new Value object
        #input the self. and other. as parent with op '+'

        out=Value (out_data,(self,other),'+')

        return out
    
    def __mul__(self, other):
        out_data=self.data * other.data

        out=Value (out_data,(self,other),'*')

        return out



