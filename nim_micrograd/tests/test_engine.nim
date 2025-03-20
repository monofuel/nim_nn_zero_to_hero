import nim_micrograd/engine, std/[strformat, sequtils]

var
  a = newTensor[float](@[1], -4.0)  # [-4.0]
  b = newTensor[float](@[1], 2.0)   # [2.0]
  c = a + b                         # [-2.0]
  d = a * b + b.pow(3)              # [-8.0 + 8.0] = [0.0]

c = c + (c + 1)                    # [-2.0 + (-2.0 + 1)] = [-3.0]
c = c + (1 + c + (-a))             # [-3.0 + (1 + -3.0 + 4.0)] = [-1.0]
d = d + (d * 2 + (b + a).relu())   # [0.0 + (0.0 * 2 + (2.0 + -4.0).relu())] = [0.0 + (0.0 + 0.0)] = [0.0]
d = d + (3 * d + (b - a).relu())   # [0.0 + (3 * 0.0 + (2.0 - -4.0).relu())] = [0.0 + (0.0 + 6.0)] = [6.0]

var
  e = c - d                        # [-1.0 - 6.0] = [-7.0]
  f = e.pow(2)                     # [(-7.0)^2] = [49.0]
  g = f / 2.0                      # [49.0 / 2.0] = [24.5]

g = g + (10.0 / f)                # [24.5 + 10.0 / 49.0] = [24.5 + 0.2040816] = [24.7040816]

echo &"g.data[0] = {g.data[0]:.4f}"  # Should print 24.7041
assert &"{g.data[0]:.4f}" == "24.7041"

# g.backward()

# echo &"a.grad = {a.grad:.4f}" # prints 138.8338, i.e. the numerical value of dg/da
# echo &"b.grad = {b.grad:.4f}" # prints 645.5773, i.e. the numerical value of dg/db
# assert &"{g.grad:.4f}" == "1.0000"
# assert &"{f.grad:.4f}" == "0.4958"
# assert &"{e.grad:.4f}" == "-6.9417"
# assert &"{d.grad:.4f}" == "6.9417"
# assert &"{a.grad:.4f}" == "138.8338"
# assert &"{b.grad:.4f}" == "645.5773"

# var
#   a2 = newValue(1)
#   b2 = a2 + a2
# b2.backward()
# echo &"a2.grad = {a2.grad:.4f}" # prints 2.0
# assert &"{a2.grad:.4f}" == "2.0000"