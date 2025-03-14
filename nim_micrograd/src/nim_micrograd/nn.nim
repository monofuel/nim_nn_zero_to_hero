import std/[random, strformat, sequtils], ./[types, engine]

## Neural Network Operations

proc zero_grad*[T](self: T) =
  ## Zeros all parameter gradients
  for p in self.parameters():
    p.grad = 0.0

## Neuron Operations
proc newNeuron*(nin: int): Neuron =
  ## Creates neuron with random weights (-1 to 1)
  Neuron(
    w: newSeqWith(nin, newValue(rand(-1.0..1.0))),
    b: newValue(0.0)
  )

proc forward*(n: Neuron, x: seq[Value]): Value =
  ## Computes weighted sum + bias
  result = n.b
  for i in 0..<n.w.len:
    result = result + n.w[i] * x[i]

proc parameters*(n: Neuron): seq[Value] =
  ## Returns bias and weights
  n.w & n.b

proc `$`*(n: Neuron): string =
  ## Shows weights and bias
  &"Neuron(w: {n.w}, b: {n.b})"

## Layer Operations
proc newLayer*(nneurons, nin: int): Layer =
  ## Creates layer with nneurons, each with nin inputs
  Layer(
    neurons: newSeqWith(nneurons, newNeuron(nin))
  )

proc forward*(l: Layer, x: seq[Value]): seq[Value] =
  ## Computes outputs for all neurons
  l.neurons.mapIt(it.forward(x))

proc parameters*(l: Layer): seq[Value] =
  ## Returns all neuron parameters
  l.neurons.mapIt(it.parameters()).concat()

proc `$`*(l: Layer): string =
  ## Shows layer neurons
  &"Layer(neurons: {l.neurons})"

## MLP Operations
proc newMLP*(nin: int, nouts: seq[int]): MLP =
  ## Creates MLP with nin inputs and nouts layer sizes
  let sizes = @[nin] & nouts
  MLP(
    layers: toSeq(0..<nouts.len).mapIt(newLayer(sizes[it+1], sizes[it]))
  )

proc forward*(m: MLP, x: seq[Value]): seq[Value] =
  ## Computes output through all layers
  result = x
  for l in m.layers:
    result = l.forward(result)

proc parameters*(m: MLP): seq[Value] =
  ## Returns all layer parameters
  m.layers.mapIt(it.parameters()).concat()

proc `$`*(m: MLP): string =
  ## Shows MLP layers
  &"MLP(layers: {m.layers})"