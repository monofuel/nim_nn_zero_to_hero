import std/[random, strformat, strutils, sequtils], engine


type
  Module* = ref object of RootObj
    grad: float
  Neuron* = ref object of Module
    w: seq[Value]
    b: Value
    nonlin: bool
  Layer* = ref object of Module
    neurons: seq[Neuron]
  MLP* = ref object of Module
    layers: seq[Layer]


proc zero_grad*[T](self: T) =
  for p in self.parameters():
    p.grad = 0.0

## Neuron
proc newNeuron*(nin: int): Neuron =
  result = Neuron()
  # get a random weight between -1 and 1
  result.w = newSeq[Value](nin)
  for i in 0 ..< nin:
    result.w[i] = newValue(rand(2.0) - 1.0)
  result.b = newValue(0.0)

proc call*(n: Neuron, x: seq[Value]): Value =
  var y = n.b
  for i in 0 ..< len(n.w):
    y = y + n.w[i] * x[i]
  if n.nonlin:
    y = y.relu()
  return y

proc parameters*(n: Neuron): seq[Value] =
  result = @[n.b]
  for w in n.w:
    result.add(w)

proc `$`*(n: Neuron): string =
  return &"Neuron(w: {$n.w}, b: {$n.b})"

## Layer

proc newLayer*(nneurons, nin: int): Layer =
  result = Layer()
  result.neurons = newSeq[Neuron](nneurons)
  for i in 0 ..< nneurons:
    result.neurons[i] = newNeuron(nin)

proc call*(l: Layer, x: seq[Value]): seq[Value] =
  result = newSeq[Value](len(l.neurons))
  for i in 0 ..< len(l.neurons):
    result[i] = l.neurons[i].call(x)
  return result

proc parameters*(l: Layer): seq[Value] =
  result = @[]
  for n in l.neurons:
    result.add(n.parameters())

proc `$`*(l: Layer): string =
  return &"Layer(neurons: {$l.neurons})"

## MLP (Multi-Layer Perceptron)

# class MLP(Module):

#     def __init__(self, nin, nouts):
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]

#     def __repr__(self):
#         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

proc newMLP*(nin: int, nouts: seq[int]): MLP =
  result = MLP()
  let sz = @[nin] & nouts
  result.layers = newSeq[Layer](len(nouts))
  for i in 0 ..< len(nouts):
    result.layers[i] = newLayer(sz[i+1], sz[i])

proc call*(m: MLP, x: seq[Value]): seq[Value] =
  var y = x
  for l in m.layers:
    y = l.call(y)
  return y

proc parameters*(m: MLP): seq[Value] =
  result = @[]
  for l in m.layers:
    result.add(l.parameters())

proc `$`*(m: MLP): string =
  return &"MLP(layers: {$m.layers})"