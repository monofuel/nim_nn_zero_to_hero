type
  Tensor*[T] = ref object
    ## A tensor with variable dimensions and gradient tracking
    data*: seq[T]             ## Flat buffer of values
    shape*: seq[int]          ## Dimensions (e.g., @[27, 27])
    grad*: seq[float]         ## Flat gradient buffer
    gradShape*: seq[int]      ## Matches data shape
    prev*: seq[Tensor[T]]## Computation graph
    op*: string              ## Operation

  Neuron*[T] = ref object
    ## Single neuron with weights and bias
    w*: seq[Tensor[T]]  ## Input weights
    b*: Tensor[T]       ## Bias

  Layer*[T] = ref object
    ## Layer of neurons
    neurons*: seq[Neuron[T]]

  MLP*[T] = ref object
    ## Multi-Layer Perceptron
    layers*: seq[Layer[T]]
