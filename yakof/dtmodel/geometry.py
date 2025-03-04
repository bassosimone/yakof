from ..frontend import bases, abstract, morphisms, spaces

ComputationSpace = abstract.TensorSpace(bases.XYZ())

ComputationTensor = abstract.Tensor[bases.XYZ]
