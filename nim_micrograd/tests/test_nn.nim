import nim_micrograd/[engine, nn]

var
  x = @[newValue(2.0), newValue(3.0), newValue(-1.0)]

echo "\n# Test Neurons"
var
  n1 = newNeuron(2)
  y1 = n1.call(x)
echo $x
echo $n1
echo $y1

n1.zero_grad()

echo "\n# Test Layers"
var
  n2 = newLayer(3,2)
  y2 = n2.call(x)
echo $n2
echo $y2

n2.zero_grad()


echo "\n# Test MLP"
var
  n = newMLP(3, @[4,4,1])


echo $n
echo $n.call(x)