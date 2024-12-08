import std/[math, sets, hashes, strformat]

# Micrograd Engine

type Value* = ref object
  data*: float
  grad*: float = 0
  prev: seq[Value]
  op: string

proc newValue*(data: float): Value =
  result = Value(data: data, grad: 0)

converter toValue*(x: float): Value =
  newValue(x)

converter toValue*(x: int): Value =
  newValue(float(x))

proc `$`*(self: Value): string =
  return &"Value(op: {self.op}, data: {self.data}, grad: {self.grad})"

proc `+`*(self, other: Value): Value =
  result = Value(
    data: self.data + other.data,
    prev: @[self, other],
    op: "+"
  )

proc `*`*(self, other: Value): Value =
  result = Value(
    data: self.data * other.data,
    prev: @[self, other],
    op: "*"
  )

proc `pow`*(self, other: Value): Value =
  result = Value(
    data: self.data.pow(other.data),
    prev: @[self, other],
    op: "pow"
  )  

proc `-`*(self: Value): Value =
  result = self * -1

proc `-`*(self, other: Value): Value =
  result = self + (-other)

proc relu*(self: Value): Value =
  result = Value(
    data: if self.data < 0: 0 else: self.data,
    prev: @[self],
    op: "relu"
  )

proc `/`*(self, other: Value): Value =
  result = self * other.pow(-1)

proc hash(x: Value): Hash =
  # Value is a ref object, assume uniqueness based on address
  hash(cast[int](x))

proc buildTopo(v: Value, topo: var seq[Value], visited: var HashSet[Value]) =
  if v notin visited:
    visited.incl(v)
    for child in v.prev:
      buildTopo(child, topo, visited)
    topo.add v

proc selfbackward*(res: Value) =
  # backward pass for just the current node
  case res.op:
  of "+":
    var
      self = res.prev[0]
      other = res.prev[1]
    self.grad += res.grad
    other.grad += res.grad
  of "-":
    # negate, does not affect gradient
    discard
  of "*":
    var
      self = res.prev[0]
      other = res.prev[1]
    self.grad += other.data * res.grad
    other.grad += self.data * res.grad
  of "pow":
    var
      self = res.prev[0]
      other = res.prev[1]
    self.grad += (other.data * self.data.pow(other.data - 1)) * res.grad
  of "relu":
    var self = res.prev[0]
    self.grad = self.grad + (if res.data > 0: res.grad else: 0)
  of "":
    # scalar value, do nothing
    discard 
  else:
    raise newException(ValueError, "unknown op for backward: " & res.op)

proc backward*(self: Value) =
  var topo: seq[Value] = @[]
  var visited = initHashSet[Value]()
  buildTopo(self, topo, visited)

  self.grad = 1.0
  for i in countdown(high(topo), 0):
    topo[i].selfBackward()