# How Does The Machine Learning Library TensorFlow Work?

> Archived from <https://outcomeschool.com/blog/how-does-the-machine-learning-library-tensorflow-work> on 2026-11-25.

<img alt="How Does The Machine Learning Library TensorFlow Work?" src="images/how-does-the-machine-learning-library-tensorflow-work.png"/><p>TensorFlow is an open-source machine learning framework. It works by letting us build a data flow graph, where each node is an operation, and then executing that graph on one or more CPUs or GPUs.</p><p>In this blog, we will learn how the Machine Learning library TensorFlow works.</p><p>I am <strong>Amit Shekhar</strong>, Founder @ <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com">Outcome School</a>, I have taught and mentored many developers, and their efforts landed them high-paying tech jobs, helped many tech companies in solving their unique problems, and created many open-source libraries being used by top companies. I am passionate about sharing knowledge through open-source, blogs, and videos.</p><p>I teach <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/program/ai-and-machine-learning">AI and Machine Learning</a> at Outcome School.</p><p>Let&#x27;s get started.</p><p>TensorFlow is an open-source machine learning framework for everyone. As I believe, when we know the internals of a library, we can get the most out of it. And we feel much more confident while working with that library. Today, TensorFlow is being used by the top companies like Google, Uber, Airbnb, Nvidia, Dropbox, etc.</p><p>Believe me, you can achieve anything out of it. It is mainly designed for deep <a target="_blank" rel="noopener noreferrer" href="neural-network.md">neural network</a> models.</p><p><strong>TensorFlow supports the following:</strong></p><ul><li>TensorFlow lets you deploy computation to one or more CPUs or <a target="_blank" rel="noopener noreferrer" href="how-does-a-gpu-work-for-deep-learning.md">GPUs</a> in a desktop, server, or mobile device in a very simple way. This way the things can be done very fast.</li><li>TensorFlow lets you express your computation as a data flow graph.</li><li>TensorFlow lets you visualize the graph using the in-built tensorboard. You can inspect and debug the graph very easily.</li><li>TensorFlow gives the best performance with an ability to iterate quickly, train models faster and run more experiments.</li><li>TensorFlow runs on nearly everything: GPUs and CPUs - including mobile and embedded platforms (see <a target="_blank" rel="noopener noreferrer" href="android-tensorflow-lite-machine-learning-example.md">TensorFlow Lite on Android</a>) - and even tensor processing units (TPUs), which are specialized hardware to do the tensor math on.</li></ul><p>PyTorch is another widely used library for the same kind of work. We have a blog on <a target="_blank" rel="noopener noreferrer" href="how-does-pytorch-work.md">how PyTorch works</a> that covers it in depth.</p><p><strong>Planning your move into AI?</strong></p><p>No matter which tech domain you work in today, with your effort, you can transition to these roles:</p><ul><li>AI Engineer</li><li>Forward Deployed Engineer</li><li>AI Agent Engineer</li><li>Harness Engineer</li><li>Voice AI Engineer</li><li>AI Architect</li><li>LLM Engineer</li><li>LLM Inference Engineer</li><li>Gen AI Engineer</li><li>MLOps Engineer</li><li>ML Engineer</li><li>Deep Learning Engineer</li></ul><p>Our <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/program/ai-and-machine-learning">AI and Machine Learning Program</a> at Outcome School takes you there.</p><p>How does the TensorFlow flexible enough to support all the above features?</p><p>The architecture of the TensorFlow helps it supporting all the above and much more.</p><p>First of all, you must know that in TensorFlow, you build a graph in which you define the constants, variables, operations and then you execute it. The graph is a data structure which consists of all the constants, variables and operations that you want to do.</p><p>The node represents an operation.</p><p>The edges are carriers of data structures (tensors), where an output of one operation (from one node) becomes the input for another operation.</p><p>You do the things in TensorFlow library like below.</p><div class="relative"><pre><code class="language-python code-highlight">import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print &quot;a: %i&quot; % sess.run(a), &quot;b: %i&quot; % sess.run(b)
    print &quot;Addition with constants: %i&quot; % sess.run(a+b)
    print &quot;Multiplication with constants: %i&quot; % sess.run(a*b)

# a=2, b=3
# Addition with constants: 5
# Multiplication with constants: 6

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print &quot;Addition with variables: %i&quot; % sess.run(add, feed_dict={a: 2, b: 3})
    print &quot;Multiplication with variables: %i&quot; % sess.run(mul, feed_dict={a: 2, b: 3})

# Addition with variables: 5
# Multiplication with variables: 6
</code></pre></div><p><strong>A quick note for you</strong></p><p>No matter which tech domain you work in, get familiar with these topics:</p><ul><li>LLM</li><li>RAG</li><li>MCP</li><li>Agent</li><li>Fine-tuning</li><li>Quantization</li></ul><p>We put it all together in one video:</p><p><a target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=lnfWvX66FUk">AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, and Quantization</a></p><p>No need to stop reading - bookmark it and watch later when you get time. Future you will thank you.</p><p>Now, let&#x27;s get back to the topic.</p><p>Let’s see it by building a mini TensorFlow. We will try to build a small library just for understanding how the TensorFlow works.</p><p>Create a class for the operations called TFOperation.</p><div class="relative"><pre><code class="language-python code-highlight">class TFOperation():

def __init__(self, input_nodes=[]):

    self.input_nodes = input_nodes
    self.output_nodes = []

    for node in input_nodes:
        node.output_nodes.append(self)

    _default_graph.operations.append(self)

def compute(self):
    pass
</code></pre></div><p>Then, extend TFOperation class to make add class.</p><div class="relative"><pre><code class="language-python code-highlight">class add(TFOperation, object):

    def __init__(self,x,y):

        super(add, self).__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var
</code></pre></div><p>Similarly, we can extend the TFOperation to make multiply class.</p><div class="relative"><pre><code class="language-python code-highlight">class multiply(TFOperation, object):

    def __init__(self,x,y):

        super(multiply, self).__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var
</code></pre></div><p>Now, create the Placeholder class for the variables.</p><div class="relative"><pre><code class="language-python code-highlight">class Placeholder():

    def __init__(self):

        self.output_nodes = []

        _default_graph.placeholders.append(self)
</code></pre></div><p>Now, create the Constant class.</p><div class="relative"><pre><code class="language-python code-highlight">class Constant():

    def __init__(self, initial_value=None):

        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)
</code></pre></div><p>Then, we will create the Graph class.</p><div class="relative"><pre><code class="language-python code-highlight">class Graph():

    def __init__(self):

        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):

        global _default_graph
        _default_graph = self
</code></pre></div><p>Now, let&#x27;s make the utility function <code>traverse_postorder</code> so that the operations happen in the correct order.</p><div class="relative"><pre><code class="language-python code-highlight">def traverse_postorder(operation):

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, TFOperation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
</code></pre></div><p>Finally create the Session class.</p><div class="relative"><pre><code class="language-python code-highlight">import numpy as np

class Session():

    def run(self, operation, feed_dict={}):

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if isinstance(node, Placeholder):

                node.output = feed_dict[node]

            elif isinstance(node, Constant):

                node.output = node.value

            else:
                # OPERATION
                node.inputs = [input_node.output for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output
</code></pre></div><p>So, our mini TensorFlow library is ready.</p><p>Let’s do some operations with constants and placeholder(variables).</p><p>We will do the following operations with our mini TensorFlow library.</p><div class="relative"><pre><code class="language-python code-highlight">y = mx + c
m = 10
c = 1
y = 10x + 1
</code></pre></div><div class="relative"><pre><code class="language-python code-highlight"># Create the Graph
g = Graph()
# Set the graph as default
g.set_as_default()
# Create a constant
m = Constant(10)
# Create an another constant
c = Constant(1)
# Create a variable using Placeholder
x = Placeholder()
# Create add and multiply operation
y = add(multiply(m,x), c)
# Create the session
sess = Session()
# Run the session to get the result
result = sess.run(operation=y, feed_dict={x:10})
# Print the result
print(result)
</code></pre></div><p>So, this is how the TensorFlow works.</p><p><strong>Stay updated:</strong> Subscribe to our <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.substack.com/subscribe">newsletter</a> to get our latest AI and Machine Learning blogs straight to your inbox.</p><h2 id="frequently-asked-questions"><a href="#frequently-asked-questions" aria-hidden="true" tabindex="-1"></a>Frequently Asked Questions</h2><h3 id="what-is-a-graph-in-tensorflow"><a href="#what-is-a-graph-in-tensorflow" aria-hidden="true" tabindex="-1"></a>What is a graph in TensorFlow?</h3><p>A graph is a data structure that holds all the constants, variables, and operations that we want to run. Each node represents an operation. The edges carry tensors, where the output of one node becomes the input for another node.</p><h3 id="can-tensorflow-run-on-mobile-devices"><a href="#can-tensorflow-run-on-mobile-devices" aria-hidden="true" tabindex="-1"></a>Can TensorFlow run on mobile devices?</h3><p>Yes. TensorFlow lets us deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device. It runs on nearly everything, including mobile and embedded platforms, and even on tensor processing units (TPUs).</p><h3 id="what-is-the-difference-between-a-placeholder-and-a-constant-in-tensorflow"><a href="#what-is-the-difference-between-a-placeholder-and-a-constant-in-tensorflow" aria-hidden="true" tabindex="-1"></a>What is the difference between a Placeholder and a Constant in TensorFlow?</h3><p>A Constant holds a fixed value that we give it when we create it. A Placeholder is used for a variable input, and it gets its value only when we run the session, through the <code>feed_dict</code>.</p><h3 id="why-are-the-operations-traversed-in-postorder-before-running-them"><a href="#why-are-the-operations-traversed-in-postorder-before-running-them" aria-hidden="true" tabindex="-1"></a>Why are the operations traversed in postorder before running them?</h3><p>Postorder traversal makes the operations happen in the correct order. It visits all the input nodes of an operation before the operation itself. So, when the session computes a node, the outputs of its inputs are already ready.</p><p>That&#x27;s it for now.</p><p>Thanks</p><p><strong>Amit Shekhar</strong><br/>Founder @ <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com">Outcome School</a></p><p>You can connect with me on:</p><ul><li><a target="_blank" rel="noopener noreferrer" href="https://x.com/amitiitbhu">X</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/in/amit-shekhar-iitbhu">LinkedIn</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://github.com/amitshekhariitbhu">GitHub</a></li></ul><p>Follow Outcome School on:</p><ul><li><a target="_blank" rel="noopener noreferrer" href="https://x.com/outcome_school">X</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://youtube.com/@OutcomeSchool">YouTube</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/company/outcomeschool">LinkedIn</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://github.com/OutcomeSchool">GitHub</a></li></ul><p><a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/blog"><strong>Read all of our high-quality blogs here.</strong></a></p>
