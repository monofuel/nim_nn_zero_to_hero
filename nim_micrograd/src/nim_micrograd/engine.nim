import math 

# Micrograd Engine

type Value* = object
  data: float
  grad: float
  backward: proc ()
  prev: seq[Value]

proc newValue*(data: float): Value =
  result = Value(data: data, grad: 0)
  result.backward = proc () = discard

converter toValue*(x: float): Value =
  newValue(x)

converter toValue*(x: int): Value =
  newValue(float(x))

proc `+`*(self, other: Value): Value =
  result = Value(data: self.data + other.data, grad: 0)
  result.backward = proc () =
    self.grad += result.grad
    other.grad += result.grad

proc `*`*(self, other:  Value): Value =
  result = Value(data: self.data * other.data, grad: 0)
  result.backward = proc () =
    self.grad += other.data * result.grad
    other.grad += self.data * result.grad

proc `pow`*(self, other: Value): Value =
  result = Value(data: self.data.pow(other.data), grad: 0)
  result.backward = proc () =
    self.grad += other.data * self.data.pow(other.data - 1) * result.grad