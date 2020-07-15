# Some Tensorflow utilities for applying a sober filter to an image.
In this tutorial we implement a Sobel filter using tensorflow. Each calculation must be defined as a calculation graph.

## Set the sober filter as a calculation graph
For the graph we need:
* an image input
* result of the filter with horizontal Sobel
* result of the filter with vertical Sobel
* Sobel measure.


### tf.reset_default_graph()
Clears the default graph stack and resets the global default graph.

NOTE: The default graph is a property of the current thread. This function applies only to the current thread. Calling this function while a tf.Session or tf.InteractiveSession is active will result in undefined behavior. Using any previously created tf.Operation or tf.Tensor objects after calling this function will result in undefined behavior.


### tf.InteractiveSession
The only difference with a regular Session is that an InteractiveSession installs itself as the default session on construction. The methods Tensor.eval() and Operation.run() will use that session to run ops.

InteractiveSession supports less typing, as allows to run variables without needing to constantly refer to the session object.

### Define the cores (constants) of the filters
tf.constant([ [..], [..], [..] ], name="aName")

The name tf. constant comes from the symbolic APIs where the value is embedded in a Const node in the tf. ... constant is useful for asserting that the value can be embedded that way. If the argument dtype is not specified, then the type is inferred from the type of value .
