import std/[math, sets, hashes, strformat, sequtils], types

## Micrograd Engine

proc newTensor*[T](shape: seq[int], data: T = 0): Tensor[T] =
  ## Create a new tensor of given shape, initialized with data
  let size = shape.foldl(a * b, 1)  # Total elements (e.g., 27*27 = 729)
  result = Tensor[T](
    data: newSeqWith(size, data),   # Flat buffer of values (e.g., zeros)
    shape: shape,                   # e.g., @[27, 27]
    grad: newSeqWith(size, 0.0),    # Flat gradient buffer
    gradShape: shape,               # Matches data shape
    prev: @[],                      # Empty prev initially
    op: ""                          # No operation for constants
  )

proc newTensor*[T](data: T): Tensor[T] =
  ## Create a new tensor from a single value
  result = Tensor[T](
    data: @[data],                  # 1-element buffer
    shape: @[1],                    # Scalar shape
    grad: @[0.0],                   # 1-element gradient
    gradShape: @[1],
    prev: @[],
    op: ""
  )

converter toTensor*(x: float): Tensor[float] =
  ## Helper to convert a float to a tensor
  newTensor(x)

converter toTensor*(x: int): Tensor[int] =
  ## Helper to convert an int to a tensor
  newTensor(x)

proc `$`*[T](self: Tensor[T]): string =
  ## String representation of a Tensor
  let shapeStr = self.shape.join("x")  # e.g., "27x27"
  # Show first few data elements (up to 5) to avoid flooding output
  let dataSnippet = if self.data.len > 5: 
                      self.data[0..4].join(", ") & "..." 
                    else: 
                      self.data.join(", ")
  # Same for gradients
  let gradSnippet = if self.grad.len > 5: 
                      self.grad[0..4].join(", ") & "..." 
                    else: 
                      self.grad.join(", ")
  return &"Tensor[T](op: {self.op}, shape: {shapeStr}, data: [{dataSnippet}], grad: [{gradSnippet}])"

proc `+`*[T](self: Tensor[T], other: SomeNumber): Tensor[T] =
  ## Add a scalar to a tensor
  let scalarTensor = newTensor[T](self.shape, T(other))  # Match self's shape
  result = self + scalarTensor

proc `+`*[T](self: SomeNumber, other: Tensor[T]): Tensor[T] =
  ## Add a tensor to a scalar
  let scalarTensor = newTensor[T](other.shape, T(self))  # Match other's shape
  result = scalarTensor + other

proc `+`*[T](self, other: Tensor[T]): Tensor[T] =
  ## Element-wise addition of two tensors
  if self.shape != other.shape:
    raise newException(ValueError, "Tensor shapes must match for +: " & $self.shape & " vs " & $other.shape)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self, other],
    op: "+"
  )
  for i in 0..<size:
    result.data[i] = self.data[i] + other.data[i]

proc `*`*[T](self: Tensor[T], other: SomeNumber): Tensor[T] =
  ## Element-wise multiplication by a scalar
  let scalarTensor = newTensor[T](self.shape, T(other))  # Match self's shape
  result = self * scalarTensor

proc `*`*[T](self: SomeNumber, other: Tensor[T]): Tensor[T] =
  ## Element-wise multiplication of scalar with tensor
  let scalarTensor = newTensor[T](other.shape, T(self))  # Match other's shape
  result = scalarTensor * other

proc `*`*[T](self, other: Tensor[T]): Tensor[T] =
  ## Element-wise multiplication of two tensors
  if self.shape != other.shape:
    raise newException(ValueError, "Tensor shapes must match for *: " & $self.shape & " vs " & $other.shape)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self, other],
    op: "*"
  )
  for i in 0..<size:
    result.data[i] = self.data[i] * other.data[i]

proc `pow`*[T](self: Tensor[T], other: SomeNumber): Tensor[T] =
  ## Element-wise power with a scalar exponent
  let scalarTensor = newTensor[T](self.shape, T(other))  # Match self's shape
  result = self.pow(scalarTensor)                       # Call tensor-tensor pow

proc `pow`*[T](self, other: Tensor[T]): Tensor[T] =
  ## Element-wise power operation
  if self.shape != other.shape:
    raise newException(ValueError, "Tensor shapes must match for pow: " & $self.shape & " vs " & $other.shape)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self, other],
    op: "pow"
  )
  for i in 0..<size:
    when T is float:
      result.data[i] = pow(self.data[i], other.data[i])
    else:
      result.data[i] = self.data[i] ^ other.data[i]  # Integer exponentiation with ^ 

proc `-`*[T](self: Tensor[T]): Tensor[T] =
  ## Unary negation (element-wise)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self],
    op: "-"
  )
  for i in 0..<size:
    result.data[i] = -self.data[i]

proc `-`*[T](self, other: Tensor[T]): Tensor[T] =
  ## Element-wise subtraction (self - other)
  result = self + (-other)

proc relu*[T](self: Tensor[T]): Tensor[T] =
  ## Element-wise ReLU, clamps negative values to 0
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self],
    op: "relu"
  )
  for i in 0..<size:
    result.data[i] = if self.data[i] < T(0): T(0) else: self.data[i]

proc `/`*[T](self: Tensor[T], other: SomeNumber): Tensor[T] =
  ## Element-wise division by scalar
  let scalarTensor = newTensor[T](self.shape, T(other))
  result = self / scalarTensor

proc `/`*[T](self: SomeNumber, other: Tensor[T]): Tensor[T] =
  ## Element-wise scalar divided by tensor
  let scalarTensor = newTensor[T](other.shape, T(self))
  result = scalarTensor / other

proc `/`*[T](self, other: Tensor[T]): Tensor[T] =
  ## Element-wise division (self / other)
  if self.shape != other.shape:
    raise newException(ValueError, "Tensor shapes must match for /: " & $self.shape & " vs " & $other.shape)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self, other],
    op: "/"
  )
  for i in 0..<size:
    result.data[i] = self.data[i] / other.data[i]

proc tanh*[T](self: Tensor[T]): Tensor[T] =
  ## Element-wise tanh activation, smooth transition between -1 and 1
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self],
    op: "tanh"
  )
  for i in 0..<size:
    let x = float(self.data[i])  # Cast to float for math
    let t = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0)
    result.data[i] = T(t)        # Cast back to T

proc exp*[T](self: Tensor[T]): Tensor[T] =
  ## Element-wise exponential (e^x)
  let size = self.data.len
  result = Tensor[T](
    data: newSeq[T](size),
    shape: self.shape,
    grad: newSeq[float](size),
    gradShape: self.shape,
    prev: @[self],
    op: "exp"
  )
  for i in 0..<size:
    result.data[i] = T(exp(float(self.data[i])))  # Cast to float, compute, cast back

proc hash*[T](x: Tensor[T]): Hash =
  ## Hash a Tensor based on its address (ref object uniqueness)
  hash(cast[int](x))

proc buildTopo*[T](v: Tensor[T], topo: var seq[Tensor[T]], visited: var HashSet[Tensor[T]]) =
  ## Recursively build the list of unique nodes, top to bottom
  if v notin visited:
    visited.incl(v)
    for child in v.prev:
      buildTopo(child, topo, visited)
    topo.add v

proc selfbackward*[T](res: Tensor[T]) =
  ## Backward pass for just the current node
  case res.op:
  of "+":
    var
      self = res.prev[0]
      other = res.prev[1]
    for i in 0..<res.grad.len:
      self.grad[i] += res.grad[i]
      other.grad[i] += res.grad[i]
  of "*":
    var
      self = res.prev[0]
      other = res.prev[1]
    for i in 0..<res.grad.len:
      self.grad[i] += float(other.data[i]) * res.grad[i]
      other.grad[i] += float(self.data[i]) * res.grad[i]
  of "/":  # Division
    var
      self = res.prev[0]   # Numerator (x)
      other = res.prev[1]  # Denominator (y)
    for i in 0..<res.grad.len:
      let y = float(other.data[i])
      self.grad[i] += res.grad[i] / y                    # ∂(x/y)/∂x = 1/y
      other.grad[i] += -res.grad[i] * float(self.data[i]) / (y * y)  # ∂(x/y)/∂y = -x/y^2
  of "pow":
    var
      self = res.prev[0]
      other = res.prev[1]
    for i in 0..<res.grad.len:
      let base = float(self.data[i])
      let expn = float(other.data[i])
      self.grad[i] += (expn * pow(base, expn - 1.0)) * res.grad[i]
  of "relu":
    var self = res.prev[0]
    for i in 0..<res.grad.len:
      self.grad[i] += (if self.data[i] > 0: res.grad[i] else: 0.0)
  of "tanh":
    var self = res.prev[0]
    for i in 0..<res.grad.len:
      let t = float(self.data[i])  # tanh output
      self.grad[i] += (1.0 - t * t) * res.grad[i]
  of "exp":
    var self = res.prev[0]
    for i in 0..<res.grad.len:
      self.grad[i] += exp(float(self.data[i])) * res.grad[i]
  of "-":  # Unary negate
    var self = res.prev[0]
    for i in 0..<res.grad.len:
      self.grad[i] += -res.grad[i]  # ∂(-x)/∂x = -1
  of "":  # Constant tensor
    discard
  else:
    raise newException(ValueError, "Unknown op for backward: " & res.op)

proc backward*[T](self: Tensor[T]) =
  ## Perform a full backward pass
  var topo: seq[Tensor[T]] = @[]
  var visited = initHashSet[Tensor[T]]()
  buildTopo(self, topo, visited)

  # Set root gradient to 1.0 for all elements
  for i in 0..<self.grad.len:
    self.grad[i] = 1.0
  
  # Reverse pass
  for i in countdown(high(topo), 0):
    topo[i].selfbackward()