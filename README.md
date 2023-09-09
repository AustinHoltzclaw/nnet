# nnet
Original idea was to create a neural network library that avoids the need for submitting an SSR for Tensorflow or pytorch.
You could just be normal and use scikit learn. This was developed out of personal interest.
If you want to learn how to make this from scratch I highly recommend learning from https://nnfs.io. I could not be more greatful.

# What's new
Added more documentation

## Layers
Layer_Dense(<input> <output>, <weight_L1>, <weight_L2>, <bias_L1>, <bias_L2>)
Layer_Dropout(<Percent of missing neurons>) 0.2/20%

Activation
Activation_ReLU()
Activation_Sigmoid()
Activation_Softmax()

## Loss
Loss_BinaryCrossEntropy()
Loss_CategoricalCrossEntropy()
Loss_MeanAbsoluteError()
Loss_MeanSquaredError()

## Optimizers
Optimizer_SGD(<learning_rate=0.001>, <decay=0>, <momentum=0>)
Optimizer_Adagrad(<learning_rate=0.001>, <decay=0>, <epsilon=1e-7>)
Optimizer_RMSProp(<learning_rate=0.001>, <decay=0>, <epsilon=1e-7>, <rho=0.9>)
Optimizer_Adam(<learning_rate=0.001>, <decay=0>, <epsilon=1e-7>, <beta1=0.9>, <beta2=0.999>)

## Accuracy
Accuracy_Categorical(<binary=False>)
Accuracy_Regression()

## Model
Model.add(<Layer or Activation>)
Model.set(<loss=Loss()>, <optimizer=Optimizer()>, <accuracy=Accuracy()>)
Model.finalize()

Model.train(<Train>, <Test>, <batch_size=None>, <print_every=1> <validation_data=()>, <epochs=10>)
