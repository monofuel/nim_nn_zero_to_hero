import std/[math, strformat]

# Micrograd Engine

type Value* = ref object
  data*: float
  grad: float = 0
  #backward: proc ()
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

proc `+=`*(self: var Value, other: Value) =
  self.data += other.data

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


proc `-`*(self, other: Value): Value =
  result = Value(
    data: self.data - other.data,
    prev: @[self, other],
    op: "-"
  )

proc `-`*(self: Value): Value =
  result = Value(
    data: -self.data,
    prev: @[self],
    op: "-"
  )

proc relu*(self: Value): Value =
  result = Value(
    data: if self.data > 0: self.data else: 0,
    prev: @[self],
    op: "relu"
  )

proc `/`*(self, other: Value): Value =
  result = Value(
    data: self.data / other.data,
    prev: @[self, other],
    op: "/"
  )