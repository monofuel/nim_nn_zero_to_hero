import std/[random, strformat, strutils, sequtils], engine

type
  Module* = ref object of RootObj
    ## Base type for neural network modules

  Neuron* = ref object of Module
    ## Single neuron with weights and bias
    w*: seq[Value]  ## Input weights
    b*: Value       ## Bias

  Layer* = ref object of Module
    ## Layer of neurons
    neurons*: seq[Neuron]  ## Neuron collection

  MLP* = ref object of Module
    ## Multi-Layer Perceptron
    layers*: seq[Layer]    ## Network layers

proc zero_grad*[T](self: T) =
  ## Zeros all parameter gradients
  for p in self.parameters():
    p.grad = 0.0

## Neuron Operations
proc newNeuron*(nin: int): Neuron =
  ## Creates neuron with random weights (-1 to 1)
  result = Neuron()
  result.w = newSeq[Value](nin)
  for i in 0 ..< nin:
    result.w[i] = newValue(rand(2.0) - 1.0)
  result.b = newValue(0.0)

proc call*(n: Neuron, x: seq[Value]): Value =
  ## Computes weighted sum + bias
  var y = n.b
  for i in 0 ..< len(n.w):
    y = y + n.w[i] * x[i]
  return y

proc parameters*(n: Neuron): seq[Value] =
  ## Returns bias and weights
  result = @[n.b]
  for w in n.w:
    result.add(w)

proc `$`*(n: Neuron): string =
  ## Shows weights and bias
  return &"Neuron(w: {$n.w}, b: {$n.b})"

## Layer Operations
proc newLayer*(nneurons, nin: int): Layer =
  ## Creates layer with nneurons, each with nin inputs
  result = Layer()
  result.neurons = newSeq[Neuron](nneurons)
  for i in 0 ..< nneurons:
    result.neurons[i] = newNeuron(nin)

proc call*(l: Layer, x: seq[Value]): seq[Value] =
  ## Computes outputs for all neurons
  result = newSeq[Value](len(l.neurons))
  for i in 0 ..< len(l.neurons):
    result[i] = l.neurons[i].call(x)
  return result

proc parameters*(l: Layer): seq[Value] =
  ## Returns all neuron parameters
  result = @[]
  for n in l.neurons:
    result.add(n.parameters())

proc `$`*(l: Layer): string =
  ## Shows layer neurons
  return &"Layer(neurons: {$l.neurons})"

## MLP Operations
proc newMLP*(nin: int, nouts: seq[int]): MLP =
  ## Creates MLP with nin inputs and nouts layer sizes
  result = MLP()
  let sz = @[nin] & nouts
  result.layers = newSeq[Layer](len(nouts))
  for i in 0 ..< len(nouts):
    result.layers[i] = newLayer(sz[i+1], sz[i])

proc call*(m: MLP, x: seq[Value]): seq[Value] =
  ## Computes output through all layers
  var y = x
  for l in m.layers:
    y = l.call(y)
  return y

proc parameters*(m: MLP): seq[Value] =
  ## Returns all layer parameters
  result = @[]
  for l in m.layers:
    result.add(l.parameters())

proc `$`*(m: MLP): string =
  ## Shows MLP layers
  return &"MLP(layers: {$m.layers})"