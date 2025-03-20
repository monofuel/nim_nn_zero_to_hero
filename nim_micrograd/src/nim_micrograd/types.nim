type
  Value* = ref object
    ## Represents a value with gradient tracking for autodiff
    data*: float  ## Actual value
    grad*: float = 0  ## Gradient (default 0)
    prev*: seq[Value]  ## Previous values for backprop
    op*: string  ## Operation that produced this value

  Neuron* = ref object
    ## Single neuron with weights and bias
    w*: seq[Value]  ## Input weights
    b*: Value       ## Bias

  Layer* = ref object
    ## Layer of neurons
    neurons*: seq[Neuron]

  MLP* = ref object
    ## Multi-Layer Perceptron
    layers*: seq[Layer]
