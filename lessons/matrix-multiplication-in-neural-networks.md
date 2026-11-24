# How Does Matrix Multiplication Work in Neural Networks?

> Archived from <https://outcomeschool.com/blog/matrix-multiplication-in-neural-networks> on 2026-11-25.

<img alt="How Does Matrix Multiplication Work in Neural Networks?" src="images/matrix-multiplication-in-neural-networks.png"/><p>Matrix multiplication in neural networks is the operation that takes the input numbers, multiplies them with the weights of all the neurons in a layer, and adds up the results, all in one step. Most of the work a neural network does is this one operation, repeated layer after layer.</p><p>In this blog, we will learn about how matrix multiplication works in neural networks. We will also see what vectors, matrices, and the dot product are, the row-times-column rule with a fully worked example, the shape rule that tells us which matrices can be multiplied, how one layer of neurons becomes one matrix multiplication plus a bias, how we process many inputs at once, and why GPUs are so good at this.</p><p>We will cover the following:</p><ul><li>What are Vectors and Matrices?</li><li>What is a Dot Product?</li><li>How does Matrix Multiplication Work?</li><li>The Shape Rule</li><li>One Layer of Neurons is One Matrix Multiplication</li><li>Processing Many Inputs at Once</li><li>Why is the GPU Perfect for Matrix Multiplication?</li></ul><p>I am <strong>Amit Shekhar</strong>, Founder @ <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com">Outcome School</a>, I have taught and mentored many developers, and their efforts landed them high-paying tech jobs, helped many tech companies in solving their unique problems, and created many open-source libraries being used by top companies. I am passionate about sharing knowledge through open-source, blogs, and videos.</p><p>I teach <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/program/ai-and-machine-learning">AI and Machine Learning</a> at Outcome School.</p><p>Let&#x27;s get started.</p><h2 id="what-are-vectors-and-matrices"><a href="#what-are-vectors-and-matrices" aria-hidden="true" tabindex="-1"></a>What are Vectors and Matrices?</h2><p>A <a target="_blank" rel="noopener noreferrer" href="neural-network.md">neural network</a> does not understand pictures, words, or sounds directly. It understands only numbers. So, before anything goes into a neural network, we turn it into a list of numbers.</p><p>A <strong>vector</strong> is a list of numbers written in a fixed order.</p><p>For example, we can describe a house with 3 numbers: the number of bedrooms, the number of bathrooms, and the number of floors.</p><div class="relative"><pre><code class="language-text code-highlight">[3, 2, 1]
</code></pre></div><p>This is a vector with 3 numbers in it. The order matters. The first number is always bedrooms, the second is always bathrooms, and the third is always floors.</p><p>A <strong>matrix</strong> is a table of numbers arranged in rows and columns. We can think of it as many vectors stacked on top of each other. For example, 2 houses described with 3 numbers each give us a matrix like below:</p><div class="relative"><pre><code class="language-text code-highlight">[ 3  2  1 ]    &lt;- house 1
[ 4  3  2 ]    &lt;- house 2
</code></pre></div><p>This matrix has 2 rows and 3 columns. We call it a <strong>2 x 3</strong> matrix. We always say the number of rows first and the number of columns second. This &quot;rows x columns&quot; is the <strong>shape</strong> of the matrix.</p><h2 id="what-is-a-dot-product"><a href="#what-is-a-dot-product" aria-hidden="true" tabindex="-1"></a>What is a Dot Product?</h2><p>Before jumping into matrix multiplication, we must know the dot product, because matrix multiplication is nothing but many dot products done together.</p><p>A <strong>dot product</strong> takes two vectors of the same length, multiplies the matching numbers, and adds all the results to get one single number.</p><p>Let&#x27;s say we go to a shop and buy 2 pens, 3 notebooks, and 1 bag. A pen costs 10, a notebook costs 5, and a bag costs 20.</p><div class="relative"><pre><code class="language-text code-highlight">quantities = [2, 3, 1]
prices     = [10, 5, 20]
</code></pre></div><p>To find the total bill, we multiply each quantity with its price and add them up:</p><p><strong>total = 2 * 10 + 3 * 5 + 1 * 20 = 20 + 15 + 20 = 55</strong></p><p>This is exactly a dot product. Two lists of 3 numbers go in, and one number, 55, comes out.</p><p>Now, let&#x27;s see why this is important for neural networks. A single neuron does exactly this. It takes the input numbers, multiplies each one with its own <strong>weight</strong>, and adds them up. A weight is a number that tells the neuron how important each input is. In the shop example, the prices play the role of the weights.</p><p><strong>So, one neuron = one dot product.</strong></p><h2 id="how-does-matrix-multiplication-work"><a href="#how-does-matrix-multiplication-work" aria-hidden="true" tabindex="-1"></a>How does Matrix Multiplication Work?</h2><p>Now that we have learned about the dot product, it&#x27;s time to learn about matrix multiplication.</p><p>When we multiply matrix A with matrix B, every number in the answer is one dot product: <strong>one row of A with one column of B</strong>. This is called the <strong>row-times-column rule</strong>.</p><p>The best way to learn this is by taking an example.</p><p>Let&#x27;s take matrix A of shape 2 x 3 and matrix B of shape 3 x 2:</p><div class="relative"><pre><code class="language-text code-highlight">A = [ 1  2  3 ]        B = [  7   8 ]
    [ 4  5  6 ]            [  9  10 ]
                           [ 11  12 ]
</code></pre></div><p>The answer C will have 2 rows (the rows of A) and 2 columns (the columns of B). So, C is a 2 x 2 matrix, and we need to find 4 numbers.</p><p><strong>Step 1:</strong> We find the number at row 1, column 1. We take row 1 of A, which is [1, 2, 3], and column 1 of B, which is [7, 9, 11].</p><p><strong>1 * 7 + 2 * 9 + 3 * 11 = 7 + 18 + 33 = 58</strong></p><p><strong>Step 2:</strong> We find the number at row 1, column 2. We take row 1 of A, [1, 2, 3], and column 2 of B, [8, 10, 12].</p><p><strong>1 * 8 + 2 * 10 + 3 * 12 = 8 + 20 + 36 = 64</strong></p><p><strong>Step 3:</strong> We find the number at row 2, column 1. We take row 2 of A, [4, 5, 6], and column 1 of B, [7, 9, 11].</p><p><strong>4 * 7 + 5 * 9 + 6 * 11 = 28 + 45 + 66 = 139</strong></p><p><strong>Step 4:</strong> We find the number at row 2, column 2. We take row 2 of A, [4, 5, 6], and column 2 of B, [8, 10, 12].</p><p><strong>4 * 8 + 5 * 10 + 6 * 12 = 32 + 50 + 72 = 154</strong></p><p>Let&#x27;s see Step 1 as a diagram, because every step follows the same path:</p><div class="relative"><pre><code class="language-text code-highlight">┌──────────────────────────────────────┐
│ row 1 of A = [1, 2, 3]               │
│ column 1 of B = [7, 9, 11]           │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ multiply the matching numbers        │
│ 1*7 = 7, 2*9 = 18, 3*11 = 33         │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ add them up                          │
│ 7 + 18 + 33 = 58                     │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ C at row 1, column 1 = 58            │
└──────────────────────────────────────┘
</code></pre></div><p>Here, we can see that one row and one column go in, and exactly one number comes out. Steps 2, 3, and 4 use a different row or column, but the path is the same.</p><p>So, the final answer is:</p><div class="relative"><pre><code class="language-text code-highlight">C = [  58   64 ]
    [ 139  154 ]
</code></pre></div><p>Let&#x27;s verify this with NumPy, a popular Python library for working with numbers. In NumPy, the <code>@</code> symbol means matrix multiplication. We can write the code as below:</p><div class="relative"><pre><code class="code-highlight language-python">import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

C = A @ B
print(C)
print(C.shape)
</code></pre></div><p>This will print the following:</p><div class="relative"><pre><code class="language-text code-highlight">[[ 58  64]
 [139 154]]
(2, 2)
</code></pre></div><p>Here, we can see that NumPy gives the same 4 numbers that we calculated by hand, and the shape of the answer is 2 x 2.</p><p>This is how matrix multiplication works. It is a set of dot products, one for every row of A paired with every column of B.</p><p><strong>Planning your move into AI?</strong></p><p>No matter which tech domain you work in today, with your effort, you can transition to these roles:</p><ul><li>AI Engineer</li><li>Forward Deployed Engineer</li><li>AI Agent Engineer</li><li>Harness Engineer</li><li>Voice AI Engineer</li><li>AI Architect</li><li>LLM Engineer</li><li>LLM Inference Engineer</li><li>Gen AI Engineer</li><li>MLOps Engineer</li><li>ML Engineer</li><li>Deep Learning Engineer</li></ul><p>Our <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/program/ai-and-machine-learning">AI and Machine Learning Program</a> at Outcome School takes you there.</p><h2 id="the-shape-rule"><a href="#the-shape-rule" aria-hidden="true" tabindex="-1"></a>The Shape Rule</h2><p>In the example above, a row of A had 3 numbers and a column of B also had 3 numbers. That is why the dot product worked. If the lengths did not match, some numbers would have no partner to multiply with.</p><p>This gives us the shape rule:</p><p><strong>(m x n) @ (n x p) = (m x p)</strong></p><ul><li>The number of columns of A (n) must be the same as the number of rows of B (n). These are the two &quot;inside&quot; numbers.</li><li>The answer takes its rows from A (m) and its columns from B (p). These are the two &quot;outside&quot; numbers.</li></ul><p>In our example:</p><p><strong>(2 x 3) @ (3 x 2) = (2 x 2)</strong></p><p>The inside numbers are 3 and 3. They match, so the multiplication is allowed. The outside numbers, 2 and 2, give the shape of the answer.</p><p>Now, let&#x27;s see what happens if the inside numbers do not match. Let&#x27;s try to multiply a 2 x 3 matrix with another 2 x 3 matrix:</p><div class="relative"><pre><code class="code-highlight language-python">import numpy as np

A = np.ones((2, 3))
B = np.ones((2, 3))
C = A @ B
</code></pre></div><p>NumPy stops with an error:</p><div class="relative"><pre><code class="language-text code-highlight">ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)-&gt;(n?,m?) (size 2 is different from 3)
</code></pre></div><p>Here, the last part of the error, &quot;size 2 is different from 3&quot;, tells us that the inside numbers do not match. A row of A has 3 numbers, but a column of B has only 2 numbers, so the dot product cannot be done.</p><p>This shape mismatch error is one of the most common errors while building neural networks. Most of the time, we do mistakes in the shapes, and the shape rule is how we find and fix them.</p><p><strong>A quick note for you</strong></p><p>No matter which tech domain you work in, get familiar with these topics:</p><ul><li>LLM</li><li>RAG</li><li>MCP</li><li>Agent</li><li>Fine-tuning</li><li>Quantization</li></ul><p>We put it all together in one video:</p><p><a target="_blank" rel="noopener noreferrer" href="https://www.youtube.com/watch?v=lnfWvX66FUk">AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, and Quantization</a></p><p>No need to stop reading - bookmark it and watch later when you get time. Future you will thank you.</p><p>Now, let&#x27;s get back to the topic.</p><h2 id="one-layer-of-neurons-is-one-matrix-multiplication"><a href="#one-layer-of-neurons-is-one-matrix-multiplication" aria-hidden="true" tabindex="-1"></a>One Layer of Neurons is One Matrix Multiplication</h2><p>Now, let&#x27;s connect everything to neural networks.</p><p>A <strong>layer</strong> in a neural network is a group of neurons that all look at the same input. Each neuron does the following:</p><ul><li>It takes a dot product of the input with its own weights.</li><li>It adds its own extra number called the <a target="_blank" rel="noopener noreferrer" href="bias-in-artificial-neural-network.md"><strong>bias</strong></a>. The bias lets the neuron shift its output up or down, even when the inputs are zero.</li></ul><p>After this, the neuron usually passes the result through an <a target="_blank" rel="noopener noreferrer" href="activation-functions.md">activation function</a>, a small function that helps the network learn complex patterns. That part comes after the multiplication, so we will keep it aside in this blog.</p><p>Let&#x27;s say our input has 3 numbers, and our layer has 2 neurons.</p><div class="relative"><pre><code class="language-text code-highlight">input x = [1, 2, 3]

neuron 1 weights = [2, 1, 0],  bias = 1
neuron 2 weights = [0, 1, 3],  bias = -2
</code></pre></div><p>If we compute each neuron one by one, we first take the dot product and then add the bias:</p><p><strong>neuron 1 = (1 * 2 + 2 * 1 + 3 * 0) + 1 = 4 + 1 = 5</strong></p><p><strong>neuron 2 = (1 * 0 + 2 * 1 + 3 * 3) + (-2) = 11 - 2 = 9</strong></p><p>Now, here comes the matrix multiplication into the picture. Instead of computing each neuron separately, we put the weights of each neuron as one <strong>column</strong> of a weight matrix W:</p><div class="relative"><pre><code class="language-text code-highlight">          neuron 1   neuron 2
W  =    [    2          0    ]
        [    1          1    ]
        [    0          3    ]

b  =    [    1         -2    ]
</code></pre></div><p>W has 3 rows (one for each input number) and 2 columns (one for each neuron). So, W is a 3 x 2 matrix.</p><p>Now, the whole layer becomes one line:</p><p><strong>output = x @ W + b</strong></p><p>Here, we can think of x as a matrix with 1 row and 3 columns, so x is 1 x 3. The inside numbers are 3 and 3, so the multiplication is allowed.</p><p>Let&#x27;s see the code for this:</p><div class="relative"><pre><code class="code-highlight language-python">import numpy as np

x = np.array([1, 2, 3])
W = np.array([[2, 0],
              [1, 1],
              [0, 3]])
b = np.array([1, -2])

print(x @ W + b)
</code></pre></div><p>This will print the following:</p><div class="relative"><pre><code class="language-text code-highlight">[5 9]
</code></pre></div><p>Here, we can see that we get the same 5 and 9 that we calculated neuron by neuron. The row of x meets column 1 of W to give neuron 1, and meets column 2 of W to give neuron 2.</p><p>Let&#x27;s see the shapes flowing through this layer as a diagram:</p><div class="relative"><pre><code class="language-text code-highlight">┌──────────────────────────────────────┐
│ input x = [1, 2, 3]                  │
│ shape 1 x 3                          │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ x @ W                                │
│ (1 x 3) @ (3 x 2) = (1 x 2)          │
│ result = [4, 11]                     │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ add bias b = [1, -2]                 │
│ [4, 11] + [1, -2] = [5, 9]           │
└──────────────────────────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ output = [5, 9]                      │
│ shape 1 x 2, one number per neuron   │
└──────────────────────────────────────┘
</code></pre></div><p>Here, we can see that 3 numbers go in and 2 numbers come out, because the layer has 2 neurons. The bias does not change the shape. It only shifts each number.</p><p><strong>A full layer of neurons is exactly one matrix multiplication plus a bias.</strong> A layer with 1,000 neurons is still one line of code. Only the shape of W changes.</p><p>We have a detailed blog on <a target="_blank" rel="noopener noreferrer" href="feed-forward-networks-in-llms.md">feed-forward networks in LLMs</a> that explains how two such layers work inside every Transformer layer.</p><p>The shape of W tells us everything about the layer:</p><p><strong>W shape = (number of inputs) x (number of neurons)</strong></p><p><strong>Note:</strong> Some libraries store W the other way round, as (number of neurons) x (number of inputs), and flip it (swap its rows and columns) before multiplying. The idea stays the same.</p><p>A deep neural network is many such layers, one after another. The output of one layer becomes the input of the next layer.</p><p>To learn Neural Networks, Feed Forward Neural Networks (FFNNs), and NumPy, and to build a Neural Network from scratch, check out our <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/program/ai-and-machine-learning">AI and Machine Learning Program</a> at Outcome School.</p><h2 id="processing-many-inputs-at-once"><a href="#processing-many-inputs-at-once" aria-hidden="true" tabindex="-1"></a>Processing Many Inputs at Once</h2><p>In the example above, we passed only one input. But in practice, we pass many inputs together. We call a group of inputs processed together a <strong>batch</strong>.</p><p>Let&#x27;s say we have 2 inputs instead of 1:</p><div class="relative"><pre><code class="language-text code-highlight">input 1 = [1, 2, 3]
input 2 = [4, 5, 6]
</code></pre></div><p>We could run the layer twice, once for each input. But there is a better way. We stack the inputs as rows of one matrix X:</p><div class="relative"><pre><code class="language-text code-highlight">X = [ 1  2  3 ]    &lt;- input 1
    [ 4  5  6 ]    &lt;- input 2
</code></pre></div><p>Now, X is a 2 x 3 matrix, and we do one bigger multiplication:</p><p><strong>(2 x 3) @ (3 x 2) = (2 x 2)</strong></p><p>Each row of the answer is the output of the layer for one input. Let&#x27;s see the code for this:</p><div class="relative"><pre><code class="code-highlight language-python">import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])
W = np.array([[2, 0],
              [1, 1],
              [0, 3]])
b = np.array([1, -2])

print(X @ W + b)
</code></pre></div><p>This will print the following:</p><div class="relative"><pre><code class="language-text code-highlight">[[ 5  9]
 [14 21]]
</code></pre></div><p>Here, we can see that the first row, [5, 9], is the same answer we got for input 1 before. The second row, [14, 21], is the answer for input 2. Let&#x27;s check it: (4 * 2 + 5 * 1 + 6 * 0) + 1 = 13 + 1 = 14, and (4 * 0 + 5 * 1 + 6 * 3) + (-2) = 23 - 2 = 21.</p><p>Notice that the bias b is a single row of 2 numbers, but the answer has 2 rows, one for each input. NumPy adds the same bias to every row automatically. This is called <strong>broadcasting</strong>.</p><p>We use the same W for every input. Only the number of rows in X changes. And one big multiplication is much faster than many small ones, because the hardware can work on all the rows together. We will see why in the next section.</p><p>Now, let&#x27;s take a bigger example. We have a batch of 32 inputs, each with 784 numbers. The first layer has 128 neurons, so its weight matrix W1 is 784 x 128. The second layer has 10 neurons, and it receives 128 numbers from the first layer, so its weight matrix W2 is 128 x 10. Let&#x27;s see the shapes flowing through both layers as a diagram:</p><div class="relative"><pre><code class="language-text code-highlight">┌──────────────────────────────────────────┐
│ batch X: 32 inputs, 784 numbers each     │
│ shape 32 x 784                           │
└──────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ layer 1: W1 shape 784 x 128              │
│ (32 x 784) @ (784 x 128) = (32 x 128)    │
└──────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ layer 2: W2 shape 128 x 10               │
│ (32 x 128) @ (128 x 10) = (32 x 10)      │
└──────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────┐
│ output: 32 rows, 10 numbers each         │
│ shape 32 x 10                            │
└──────────────────────────────────────────┘
</code></pre></div><p>Here, we can see that the number of rows stays 32 all the way down, because every row is one input. Only the number of columns changes, and it always becomes the number of neurons in that layer. We leave the bias out of this diagram, because it does not change the shape.</p><p><strong>Stay updated:</strong> Subscribe to our <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.substack.com/subscribe">newsletter</a> to get our latest AI and Machine Learning blogs straight to your inbox.</p><h2 id="why-is-the-gpu-perfect-for-matrix-multiplication"><a href="#why-is-the-gpu-perfect-for-matrix-multiplication" aria-hidden="true" tabindex="-1"></a>Why is the GPU Perfect for Matrix Multiplication?</h2><p>Now, the next big question is: why do we need a GPU for this?</p><p>Let&#x27;s count the work. When we multiply an (m x n) matrix with an (n x p) matrix, the answer has m * p numbers, and each of them is a dot product with n multiplications. So, we need m * n * p multiplications in total. For the first layer in the example above:</p><p><strong>32 * 784 * 128 = 3,211,264 multiplications</strong></p><p>That is more than 3 million multiplications for one layer and one small batch. Large models do billions of them for every single input.</p><p>Now, let&#x27;s look back at our worked example. To find 58, we needed only row 1 of A and column 1 of B. To find 154, we needed only row 2 of A and column 2 of B. <strong>No number in the answer depends on any other number in the answer.</strong> So, all of them can be computed at the same time.</p><p>A chip does its calculations using <strong>cores</strong>. A core is one worker inside the chip that can do calculations on its own. A <strong>CPU</strong> has a small number of very powerful cores. It is great at doing a few complex tasks one after another.</p><p>A <strong>GPU</strong> has thousands of smaller, simpler cores. Each core can do a multiply and an add very quickly. Since every number in the answer is an independent dot product, the GPU gives different dot products to different cores, and they all run in parallel.</p><p>Let&#x27;s say we have to check 10,000 exam answer sheets. One brilliant professor checking them one by one will take a long time. A thousand teachers, each checking 10 sheets at the same time, will finish much faster. The CPU is the brilliant professor, and the GPU is the thousand teachers.</p><p>Let&#x27;s see this for our 2 x 2 example as a diagram:</p><div class="relative"><pre><code class="language-text code-highlight">┌─────────────────────────────────────────────┐
│ A @ B needs 4 dot products                  │
└─────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ GPU: all 4 dot products at the same time    │
│                                             │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│ │ core 1 │ │ core 2 │ │ core 3 │ │ core 4 │ │
│ │   58   │ │   64   │ │  139   │ │  154   │ │
│ └────────┘ └────────┘ └────────┘ └────────┘ │
└─────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ C = [[58, 64], [139, 154]]                  │
└─────────────────────────────────────────────┘
</code></pre></div><p>Here, we can see that each core computes one number of C, and no core waits for another core. In a real GPU, thousands of cores do this together for much bigger matrices.</p><p>Modern GPUs also have special units built only for matrix multiplication. This is why neural networks are trained and run on GPUs.</p><p>We have a detailed blog on <a target="_blank" rel="noopener noreferrer" href="how-does-a-gpu-work-for-deep-learning.md">how a GPU works for deep learning</a> that explains this step by step.</p><p>Now we must have understood how matrix multiplication works in neural networks, from a single dot product to a full batch of inputs running through many layers on a GPU.</p><h2 id="frequently-asked-questions"><a href="#frequently-asked-questions" aria-hidden="true" tabindex="-1"></a>Frequently Asked Questions</h2><h3 id="what-does-the--symbol-mean-in-numpy"><a href="#what-does-the--symbol-mean-in-numpy" aria-hidden="true" tabindex="-1"></a>What does the @ symbol mean in NumPy?</h3><p>The @ symbol means matrix multiplication in NumPy. When we write A @ B, NumPy takes the dot product of every row of A with every column of B and gives us the answer matrix. If the inside numbers of the two shapes do not match, NumPy stops with a shape mismatch error.</p><h3 id="what-is-the-shape-of-the-weight-matrix-of-a-layer"><a href="#what-is-the-shape-of-the-weight-matrix-of-a-layer" aria-hidden="true" tabindex="-1"></a>What is the shape of the weight matrix of a layer?</h3><p>The weight matrix W of a layer has the shape (number of inputs) x (number of neurons). Each column of W holds the weights of one neuron. Some libraries store W the other way round, as (number of neurons) x (number of inputs), and flip it before multiplying, but the idea stays the same.</p><h3 id="how-many-multiplications-does-a-matrix-multiplication-need"><a href="#how-many-multiplications-does-a-matrix-multiplication-need" aria-hidden="true" tabindex="-1"></a>How many multiplications does a matrix multiplication need?</h3><p>Multiplying an (m x n) matrix with an (n x p) matrix needs m * n * p multiplications. The answer has m * p numbers, and each of them is a dot product with n multiplications. For a batch of 32 inputs with 784 numbers each going into a layer of 128 neurons, that is 3,211,264 multiplications.</p><h3 id="is-the-activation-function-part-of-the-matrix-multiplication"><a href="#is-the-activation-function-part-of-the-matrix-multiplication" aria-hidden="true" tabindex="-1"></a>Is the activation function part of the matrix multiplication?</h3><p>No. The activation function is not part of the matrix multiplication. A layer first does the matrix multiplication and adds the bias, and then the neuron usually passes the result through an activation function, a small function that helps the network learn complex patterns.</p><p>That&#x27;s it for now.</p><p>Thanks</p><p><strong>Amit Shekhar</strong><br/>Founder @ <a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com">Outcome School</a></p><p>You can connect with me on:</p><ul><li><a target="_blank" rel="noopener noreferrer" href="https://x.com/amitiitbhu">X</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/in/amit-shekhar-iitbhu">LinkedIn</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://github.com/amitshekhariitbhu">GitHub</a></li></ul><p>Follow Outcome School on:</p><ul><li><a target="_blank" rel="noopener noreferrer" href="https://x.com/outcome_school">X</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://youtube.com/@OutcomeSchool">YouTube</a></li><li><a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/company/outcomeschool">LinkedIn</a></li><li><a target="_blank" rel="noopener noreferrer" href="http://github.com/OutcomeSchool">GitHub</a></li></ul><p><a target="_blank" rel="noopener noreferrer" href="https://outcomeschool.com/blog"><strong>Read all of our high-quality blogs here.</strong></a></p>
