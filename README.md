# cnncifar
Convolutional Neural Network trained on the CIFAR-10 dataset with analysis.

<html>
<head>
    <meta charset="UTF-8">
    <title>ReLU Definition</title>
    <!-- Load MathJax for LaTeX rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
</head>
<body>

<h2>ReLU Function Definition</h2>

<p>
The Rectified Linear Unit (ReLU) function is defined as:
</p>

<p>
$$
\text{ReLU}(x) = \max(0, x) =
\begin{cases}
x, & \text{if } x > 0, \\
0, & \text{otherwise}.
\end{cases}
$$
</p>

</body>

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
