import math

class Value:
    #Initialize attribute
    def __init__(self, data, _children=(), _op=''):
        self.data = data      
        self.grad = 0.0 
        self._backward = lambda: None      
        self._prev = set(_children)  
        self._op = _op  
    
    # String representation of the object             
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out_data = self.data + other.data
        out = Value (out_data,(self,other),'+') # Record the operation that created this node

        # Accumulate gradients using the chain rule
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
            return self + other
        
           
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out_data=self.data * other.data
        out = Value (out_data,(self,other),'*') # Record the operation that created this node

        #Accumulate gradients using the chain rule
        def _backward(): 
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance (other,(int,float))
        out_data = self.data ** other
        out = Value (out_data,(self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        
        out._backward = _backward

        return out

    
    def backward(self):
        topo=[]
        visited=set()    # list for record

        def build_topo(v):
            if v not in visited:
                visited.add(v)          
                for child in v._prev:   # Post-order traversal to build topological graph
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        
        #set init value
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
    
    def tanh(self):
        x = self.data 
        # tanh = (e^2x-1) / (e^2x+1)  
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) 

        # operation： 'tanh'
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad  #Quotient Rule

        out._backward = _backward

        return out
    

   
    
    
    








