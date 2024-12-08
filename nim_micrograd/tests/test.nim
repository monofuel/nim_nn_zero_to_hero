import nim_micrograd/engine, std/strformat

# ```python
# from micrograd.engine import Value

# a = Value(-4.0)
# b = Value(2.0)
# c = a + b
# d = a * b + b**3
# c += c + 1
# c += 1 + c + (-a)
# d += d * 2 + (b + a).relu()
# d += 3 * d + (b - a).relu()
# e = c - d
# f = e**2
# g = f / 2.0
# g += 10.0 / f
# print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
# g.backward()
# print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
# print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
# ```

var
  a = newValue(-4.0)
  b = newValue(2.0)
  c = a + b
  d = a * b + b.pow(3)

c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()

var
  e = c - d
  f = e.pow(2)
  g = f / 2.0

g += 10.0 / f

assert &"{g.data:.4f}" == "24.7041"
echo &"g.data = {g.data:.4f}" # prints 24.7041, the outcome of this forward pass

# g.backward()

assert &"{a.grad:.4f}" == "138.8338"
assert &"{b.grad:.4f}" == "645.5773"
echo &"a.grad = {a.grad:.4f}" # prints 138.8338, i.e. the numerical value of dg/da
echo &"b.grad = {b.grad:.4f}" # prints 645.5773, i.e. the numerical value of dg/db