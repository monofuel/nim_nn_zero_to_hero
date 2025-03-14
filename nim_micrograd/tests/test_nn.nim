import std/[strformat], nim_micrograd/[engine, types, nn]

const
  StepSize = 0.01
  Iterations = 100

var
  x = @[newValue(2.0), newValue(3.0), newValue(-1.0)]

echo "\n# Test Neurons"
var
  n1 = newNeuron(2)
  y1 = n1.forward(x)
echo $x
echo $n1
echo $y1

n1.zero_grad()

echo "\n# Test Layers"
var
  n2 = newLayer(3,2)
  y2 = n2.forward(x)
echo $n2
echo $y2

n2.zero_grad()


echo "\n# Test MLP"
var
  n = newMLP(3, @[4,4,1])

echo $n
echo $n.forward(x)

echo "\n# Test MLP training"

var
  xs = @[
    @[newValue(2.0), newValue(3.0), newValue(-1.0)],
    @[newValue(3.0), newValue(-1.0), newValue(0.5)],
    @[newValue(0.5), newValue(1.0), newValue(1.0)],
    @[newValue(1.0), newValue(1.0), newValue(-1.0)],
  ]
  ys = [newValue(1.0), newValue(-1.0), newValue(1.0)] # desired outputs

for i in 0 ..< Iterations:
  echo &"## Iteration {i}"

  var ypred: seq[Value]

  # forward pass for all examples
  for x in xs:
    ypred.add(n.forward(x))

  # echo ypred

  # sum the loss for all examples
  var loss = newValue(0.0)
  for i in 0 ..< len(ys):
    loss = loss + (ypred[i] - ys[i]).pow(2)

  echo &"loss = {loss}"

  # run a backwards pass on the loss for all examples
  loss.backward()

  # update weights, and zero gradients
  for p in n.parameters():
    p.data = p.data - StepSize * p.grad
  n.zero_grad()
  