import nim_micrograd/engine, std/strformat

var
  a = newValue(-4.0)
  b = newValue(2.0)
  c = a + b
  d = a * b + b.pow(3)

c = c + (c + 1)
c = c + (1 + c + (-a))
d = d + (d * 2 + (b + a).relu())
d = d + (3 * d + (b - a).relu())

var
  e = c - d
  f = e.pow(2)
  g = f / 2.0

g = g + (10.0 / f)

echo &"g.data = {g.data:.4f}" # prints 24.7041, the outcome of this forward pass
assert &"{g.data:.4f}" == "24.7041"

g.backward()

echo &"a.grad = {a.grad:.4f}" # prints 138.8338, i.e. the numerical value of dg/da
echo &"b.grad = {b.grad:.4f}" # prints 645.5773, i.e. the numerical value of dg/db
assert &"{g.grad:.4f}" == "1.0000"
assert &"{f.grad:.4f}" == "0.4958"
assert &"{e.grad:.4f}" == "-6.9417"
assert &"{d.grad:.4f}" == "6.9417"
assert &"{a.grad:.4f}" == "138.8338"
assert &"{b.grad:.4f}" == "645.5773"