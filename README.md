# CNN CiFAR-10
Convolutional Neural Network trained on the CIFAR-10 dataset with analysis.

<html>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


<h1>Analysis of Custom ConvNet Architecture</h1>
<p>
    I made a convolutional neural network using PyTorch and explored different hyperparameters and architechtural settings to increase the accuracy. Here, I will show what my process and reasoning was for making certain changes to the neural network and how those affected the accuracy.
</p>
<p>
    CNNs consist of an input layer, hidden layers, and an output layer. Usually, the hidden layers are the ones that perform convolution. In these layers, the dot product (or Frobenius inner product) is computed between a convolution kernel and a segment of the image matrix. The convolution kernel is a small matrix that is chosen to extract features from the image, such as edges, ridges, or a certain segment like a line. The convolutional layers are followed by pooling layers, which basically perform downsampling so as to reduce computational cost and increase the receptive fields of neurons later in the network (receptive field of a neuron: all input neurons that can affect this output neurons value).
</p>
<p>
        The CNN input is a tensor of dimension (# imputs) x (input height) x (input width) x (channels). The layer abstracts this input to another tensor whose dimensions are now relating to the feautre map. Unlike fully-connected feedforward NNs, convolutional neural networks dont have as many neurons needed for the computation. This is ideal because if we used an FCFF neural net, we would need a massive number of neurons to process a large image. Each neuron gets its input from a number of neurons in the previous layer. In a convolution layer, each neuron gets its input from a restricted area of the previous neuron layer, the receptive field. In the traditional 2D CNN, 5x5 neurons is the size of this layer, whereas in a fully-connected network, the receptive field of a neuron <i>n</i> is all neurons in the <i>(n-1)th</i> layer. 
</p>
<h2>Architecture Overview</h2>
<table>
    <tr>
        <th>Layer</th>
        <th>Type</th>
        <th>Output Shape</th>
        <th>Parameters</th>
    </tr>
    <tr>
        <td>Conv1 + BN + ReLU + Pool</td>
        <td>Conv2D(3, 32, k=3, p=1)</td>
        <td>32 × 16 × 16</td>
        <td>896</td>
    </tr>
    <tr>
        <td>Conv2 + BN + ReLU + Pool</td>
        <td>Conv2D(32, 64, k=3, p=1)</td>
        <td>64 × 8 × 8</td>
        <td>18,624</td>
    </tr>
    <tr>
        <td>Conv3 + BN + ReLU + Pool</td>
        <td>Conv2D(64, 128, k=3, p=1)</td>
        <td>128 × 4 × 4</td>
        <td>74,112</td>
    </tr>
    <tr>
        <td>Conv4 + BN + ReLU + Pool</td>
        <td>Conv2D(128, 256, k=3, p=1)</td>
        <td>256 × 2 × 2</td>
        <td>295,680</td>
    </tr>
    <tr>
        <td>FC1 + BN + Dropout</td>
        <td>Linear(1024 → 256)</td>
        <td>256</td>
        <td>262,912</td>
    </tr>
    <tr>
        <td>FC2 + BN + Dropout</td>
        <td>Linear(256 → 128)</td>
        <td>128</td>
        <td>33,024</td>
    </tr>
    <tr>
        <td>FC3</td>
        <td>Linear(128 → 10)</td>
        <td>10</td>
        <td>1,290</td>
    </tr>
</table>

<p><strong>Total parameters:</strong> 686,730 (all trainable)</p>

<h2>Design Choices</h2>
<ul>
    <li><strong>Batch Normalization:</strong> Speeds up training and provides some regularization.</li>
    <li><strong>Dropout (0.25):</strong> Reduces overfitting by randomly zeroing activations.</li>
    <li><strong>MaxPooling(2×2):</strong> Downsamples feature maps, reducing computational cost.</li>
    <li><strong>Four convolutional stages:</strong> Increasing channels from 32 to 256 to capture complex features.</li>
</ul>

<p>
    This architecture is a balance between depth and parameter count, making it suitable for CIFAR-10 while avoiding the 
    heavy computation of very deep networks like ResNet-50.
</p>


<body>

<h2>Neural Network Training Results</h2>

<table>
    <thead>
        <tr>
            <th>No.</th>
            <th>Optimizer</th>
            <th>Loss</th>
            <th>Activation</th>
            <th>Accuracy (%)</th>
            <th>Epochs</th>
            <th>Batch Size</th>
            <th>Learning Rate</th>
            <th># Parameters</th>
            <th># Layers</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>tanh(x)</td>
            <td>51.61</td>
            <td>4</td>
            <td>4</td>
            <td>0.01</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>2</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>tanh(x)</td>
            <td>55.89</td>
            <td>4</td>
            <td>64</td>
            <td>0.01</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>tanh(x)</td>
            <td>48.05</td>
            <td>4</td>
            <td>64</td>
            <td>0.01</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>4</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>tanh(x)</td>
            <td>56.84</td>
            <td>4</td>
            <td>64</td>
            <td>0.001</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>5</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>ReLU(x)</td>
            <td>56.84</td>
            <td>4</td>
            <td>64</td>
            <td>0.001</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>6</td>
            <td>Stoch. Grad. Desc.</td>
            <td>Cross Entropy</td>
            <td>ReLU(x)</td>
            <td>57.77</td>
            <td>50</td>
            <td>64</td>
            <td>0.001</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>7</td>
            <td>Adap. Mom. Estm.</td>
            <td>Cross Entropy</td>
            <td>ReLU(x)</td>
            <td>76.34</td>
            <td>50</td>
            <td>64</td>
            <td>0.001</td>
            <td>545,546</td>
            <td>14</td>
        </tr>
        <tr>
            <td>8</td>
            <td>Adap. Mom. Estm.</td>
            <td>Cross Entropy</td>
            <td>ReLU(x)</td>
            <td>81.76</td>
            <td>60</td>
            <td>96</td>
            <td>0.001</td>
            <td>653,194</td>
            <td>16</td>
        </tr>
        <tr>
            <td>8</td>
            <td>Adap. Mom. Estm.</td>
            <td>Cross Entropy</td>
            <td>ReLU(x)</td>
            <td>85.52</td>
            <td>60</td>
            <td>96</td>
            <td>0.001</td>
            <td>686,730</td>
            <td>20</td>
        </tr>
    </tbody>
</table>

</body>
</html>
