import std/[math, strformat]

# Micrograd Engine

type Value* = ref object
  data: float
  grad: float = 0
  backward: proc ()
  prev: seq[Value]
  op: string

proc newValue*(data: float): Value =
  result = Value(data: data, grad: 0)
  result.backward = proc () = discard

converter toValue*(x: float): Value =
  newValue(x)

converter toValue*(x: int): Value =
  newValue(float(x))

proc `$`*(self: Value): string =
  return &"Value(op: {self.op}, data: {self.data}, grad: {self.grad})"

proc `+`*(self, other: Value): Value =
  result = Value(data: self.data + other.data, prev: @[self, other], op: "+")
  result.backward = proc () =
    self.grad += result.grad
    other.grad += result.grad

# handle +=
proc `+=`*(self: var Value, other: Value): Value =
  self.data += other.data
  self.backward = proc () =
    self.grad += other.grad

proc `*`*(self, other: Value): Value =
  result = Value(data: self.data * other.data, prev: @[self, other], op: "*")
  result.backward = proc () =
    self.grad += other.data * result.grad
    other.grad += self.data * result.grad

proc `pow`*(self, other: Value): Value =
  result = Value(data: self.data.pow(other.data), prev: @[self, other], op: "pow")
  result.backward = proc () =
    self.grad += other.data * self.data.pow(other.data - 1) * result.grad

  