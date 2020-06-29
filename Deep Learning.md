# 深度学习

## python中的广播机制

**axis用来指明将要进行的运算是沿着哪个轴执行，在numpy中，0轴是垂直的，也就是列，而1轴是水平的，也就是行。**

![img](http://www.ai-start.com/dl2017/images/bdc1b1f6f0ba18659f140e47b26bf38b.png) 

首先是**numpy**广播机制

**如果两个数组的后缘维度的轴长度相符或其中一方的轴长度为1，则认为它们是广播兼容的。广播会在缺失维度和轴长度为1的维度上进行。**

后缘维度的轴长度：`A.shape[-1]` 即矩阵维度元组中的最后一个位置的值

对于视频中卡路里计算的例子，矩阵 $A_{3,4}$ 后缘维度的轴长度是4，而矩阵 $cal_{1,4}$ 的后缘维度也是4，则他们满足后缘维度轴长度相符，可以进行广播。广播会在轴长度为1的维度进行，轴长度为1的维度对应`axis=0`，即垂直方向，矩阵 $$\text{cal}*{1,4}$$ 沿`axis=0`(垂直方向)复制成为 $$\text{cal_temp}*{3,4}$$ ，之后两者进行逐元素除法运算。

现在解释上图中的例子

矩阵 $A_{m,n}$ 和矩阵 $B_{1,n}$ 进行四则运算，后缘维度轴长度相符，可以广播，广播沿着轴长度为1的轴进行，即 $B_{1,n}$ 广播成为 ${B_{m,n}}'$ ，之后做逐元素四则运算。

矩阵 $A_{m,n}$ 和矩阵 $B_{m,1}$ 进行四则运算，后缘维度轴长度不相符，但其中一方轴长度为1，可以广播，广播沿着轴长度为1的轴进行，即 $B_{m,1}$ 广播成为 ${B_{m,n}}'$ ，之后做逐元素四则运算。

矩阵 $A_{m,1}$ 和常数$ R$ 进行四则运算，后缘维度轴长度不相符，但其中一方轴长度为1，可以广播，广播沿着缺失维度和轴长度为1的轴进行，缺失维度就是`axis=0`,轴长度为1的轴是`axis=1`，即$R$广播成为 ${B_{m,1}}'$ ，之后做逐元素四则运算。

总结一下`broadcasting`，可以看看下面的图：

![img](http://www.ai-start.com/dl2017/images/695618c70fd2922182dc89dca8eb83cc.png) 

**Python**的特性允许你使用广播（**broadcasting**）功能，这是**Python**的**numpy**程序语言库中最灵活的地方。而我认为这是程序语言的优点，也是缺点。优点的原因在于它们创造出语言的表达性，**Python**语言巨大的灵活性使得你仅仅通过一行代码就能做很多事情。但是这也是缺点，由于广播巨大的灵活性，有时候你对于广播的特点以及广播的工作原理这些细节不熟悉的话，你可能会产生很细微或者看起来很奇怪的**bug**。例如，如果你将一个列向量添加到一个行向量中，你会以为它报出维度不匹配或类型错误之类的错误，但是实际上你会得到一个行向量和列向量的求和。

在**Python**的这些奇怪的影响之中，其实是有一个内在的逻辑关系的。但是如果对**Python**不熟悉的话，我就曾经见过的一些学生非常生硬、非常艰难地去寻找**bug**。所以我在这里想做的就是分享给你们一些技巧，这些技巧对我非常有用，它们能消除或者简化我的代码中所有看起来很奇怪的**bug**。同时我也希望通过这些技巧，你也能更容易地写没有**bug**的**Python**和**numpy**代码。

为了演示**Python-numpy**的一个容易被忽略的效果，特别是怎样在**Python-numpy**中构造向量，让我来做一个快速示范。首先设置$a=np.random.randn(5)$，这样会生成存储在数组 $a$ 中的5个高斯随机数变量。之后输出 $a$，从屏幕上可以得知，此时 $a$ 的**shape**（形状）是一个$(5,)$的结构。这在**Python**中被称作**一个一维数组**。它既不是一个行向量也不是一个列向量，这也导致它有一些不是很直观的效果。举个例子，如果我输出一个转置阵，最终结果它会和$a$看起来一样，所以$a$和$a$的转置阵最终结果看起来一样。而如果我输出$a$和$a$的转置阵的内积，你可能会想：$a$乘以$a$的转置返回给你的可能会是一个矩阵。但是如果我这样做，你只会得到一个数。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/a44df591ad815cc67d74a275bd444342.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/a44df591ad815cc67d74a275bd444342.png)

所以建议你编写神经网络时，不要使用shape为 *(5,)*、*(n,)* 或者其他一维数组的数据结构。相反，如果你设置 $a$ 为$(5,1)$，那么这就将置于5行1列向量中。在先前的操作里 $a$ 和 $a$ 的转置看起来一样，而现在这样的 $a$ 变成一个新的 $a$ 的转置，并且它是一个行向量。请注意一个细微的差别，在这种数据结构中，当我们输出 $a$ 的转置时有两对方括号，而之前只有一对方括号，所以这就是1行5列的矩阵和一维数组的差别。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/264aac3c4b28d894821771308be61417.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/264aac3c4b28d894821771308be61417.png)

如果你输出 $a$ 和 $a$ 的转置的乘积，然后会返回给你一个向量的外积，是吧？所以这两个向量的外积返回给你的是一个矩阵。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/424925f28c12891f7807b24e5bb41d8b.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/424925f28c12891f7807b24e5bb41d8b.png)

就我们刚才看到的，再进一步说明。首先我们刚刚运行的命令是这个 $(a=np.random.randn(5))$，它生成了一个数据结构$a$，其中$a.shape$是$(5,)$。这被称作 $a$ 的一维数组，同时这也是一个非常有趣的数据结构。它不像行向量和列向量那样表现的很一致，这使得它带来一些不直观的影响。所以我建议，当你在编程练习或者在执行逻辑回归和神经网络时，你不需要使用这些一维数组。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5648345cf0ca282be13961bdc5b1681e.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/5648345cf0ca282be13961bdc5b1681e.png)

相反，如果你每次创建一个数组，你都得让它成为一个列向量，产生一个$(5,1)$向量或者你让它成为一个行向量，那么你的向量的行为可能会更容易被理解。所以在这种情况下，$a.shape$等同于$(5,1)$。这种表现很像 $a$，但是实际上却是一个列向量。同时这也是为什么当它是一个列向量的时候，你能认为这是矩阵$(5,1)$；同时这里 $a.shape$ 将要变成$(1,5)$，这就像行向量一样。所以当你需要一个向量时，我会说用这个或那个(**column vector or row vector**)，但绝不会是一维数组。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/2102548104812bdfbccd421f3848f900.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/2102548104812bdfbccd421f3848f900.png)

我写代码时还有一件经常做的事，那就是如果我不完全确定一个向量的维度(**dimension**)，我经常会扔进一个断言语句(**assertion statement**)。像这样，去确保在这种情况下是一个$(5,1)$向量，或者说是一个列向量。这些断言语句实际上是要去执行的，并且它们也会有助于为你的代码提供信息。所以不论你要做什么，不要犹豫直接插入断言语句。如果你不小心以一维数组来执行，你也能够重新改变数组维数 $a=reshape$，表明一个$(5,1)$数组或者一个$(1,5)$数组，以致于它表现更像列向量或行向量。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/147d3a15b0a4dae283662b2c44f91778.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/147d3a15b0a4dae283662b2c44f91778.png)

我有时候看见学生因为一维数组不直观的影响，难以定位bug而告终。通过在原先的代码里清除一维数组，我的代码变得更加简洁。而且实际上就我在代码中表现的事情而言，我从来不使用一维数组。因此，要去简化你的代码，而且不要使用一维数组。总是使用 $n \times 1$ 维矩阵（基本上是列向量），或者 $1 \times n$ 维矩阵（基本上是行向量），这样你可以减少很多**assert**语句来节省核矩阵和数组的维数的时间。另外，为了确保你的矩阵或向量所需要的维数时，不要羞于 **reshape** 操作。

## 激活函数

激活函数：使用一个神经网络时，需要决定使用哪种激活函数用隐藏层上，哪种用在输出节点上。到目前为止，之前的视频只用过**sigmoid**激活函数，但是，有时其他的激活函数效果会更好。

在神经网路的前向传播中，的$a^{[1]} = \sigma(z^{[1]})$和$a^{[2]} =\sigma(z^{[2]})$这两步会使用到**sigmoid**函数。**sigmoid**函数在这里被称为激活函数。 公式3.18： $a = \sigma(z) = \frac{1}{{1 + e}^{- z}}$

更通常的情况下，使用不同的函数$g( z^{[1]})$，$g$可以是除了**sigmoid**函数以外的非线性函数。**tanh**函数或者双曲正切函数是总体上都优于**sigmoid**函数的激活函数。

如图，$a = tan(z)$的值域是位于+1和-1之间。 公式3.19： $a= tanh(z) = \frac{e^{z} - e^{- z}}{e^{z} + e^{- z}}$

事实上，**tanh**函数是**sigmoid**的向下平移和伸缩后的结果。对它进行了变形后，穿过了$(0,0)$点，并且值域介于+1和-1之间。

结果表明，如果在隐藏层上使用函数 公式3.20： $g(z^{[1]}) = tanh(z^{[1]}) $ 效果总是优于**sigmoid**函数。因为函数值域在-1和+1的激活函数，其均值是更接近零均值的。在训练一个算法模型时，如果使用**tanh**函数代替**sigmoid**函数中心化数据，使得数据的平均值更接近0而不是0.5.

这会使下一层学习简单一点，在第二门课中会详细讲解。

在讨论优化算法时，有一点要说明：我基本已经不用**sigmoid**激活函数了，**tanh**函数在所有场合都优于**sigmoid**函数。

但有一个例外：在二分类的问题中，对于输出层，因为$y$的值是0或1，所以想让$\hat{y}$的数值介于0和1之间，而不是在-1和+1之间。所以需要使用**sigmoid**激活函数。这里的 公式3.21： $g(z^{[2]}) = \sigma(z^{[2]})$ 在这个例子里看到的是，对隐藏层使用**tanh**激活函数，输出层使用**sigmoid**函数。

所以，在不同的神经网络层中，激活函数可以不同。为了表示不同的激活函数，在不同的层中，使用方括号上标来指出$g$上标为$[1]$的激活函数，可能会跟$g$上标为$[2]$不同。方括号上标$[1]$代表隐藏层，方括号上标$[2]$表示输出层。

**sigmoid**函数和**tanh**函数两者共同的缺点是，在$z$特别大或者特别小的情况下，导数的梯度或者函数的斜率会变得特别小，最后就会接近于0，导致降低梯度下降的速度。

在机器学习另一个很流行的函数是：修正线性单元的函数（**ReLu**），**ReLu**函数图像是如下图。 公式3.22： $ a =max( 0,z) $ 所以，只要$z$是正值的情况下，导数恒等于1，当$z$是负值的时候，导数恒等于0。从实际上来说，当使用$z$的导数时，$z$=0的导数是没有定义的。但是当编程实现的时候，$z$的取值刚好等于0.00000001，这个值相当小，所以，在实践中，不需要担心这个值，$z$是等于0的时候，假设一个导数是1或者0效果都可以。

这有一些选择激活函数的经验法则：

如果输出是0、1值（二分类问题），则输出层选择**sigmoid**函数，然后其它的所有单元都选择**Relu**函数。

这是很多激活函数的默认选择，如果在隐藏层上不确定使用哪个激活函数，那么通常会使用**Relu**激活函数。有时，也会使用**tanh**激活函数，但**Relu**的一个优点是：当$z$是负值的时候，导数等于0。

这里也有另一个版本的**Relu**被称为**Leaky Relu**。

当$z$是负值时，这个函数的值不是等于0，而是轻微的倾斜，如图。

这个函数通常比**Relu**激活函数效果要好，尽管在实际中**Leaky ReLu**使用的并不多。

[![w600](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L1_week3_9.jpg)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L1_week3_9.jpg) 图3.6.1

两者的优点是：

第一，在$z$的区间变动很大的情况下，激活函数的导数或者激活函数的斜率都会远大于0，在程序实现就是一个**if-else**语句，而**sigmoid**函数需要进行浮点四则运算，在实践中，使用**ReLu**激活函数神经网络通常会比使用**sigmoid**或者**tanh**激活函数学习的更快。

第二，**sigmoid**和**tanh**函数的导数在正负饱和区的梯度都会接近于0，这会造成梯度弥散，而**Relu**和**Leaky ReLu**函数大于0部分都为常数，不会产生梯度弥散现象。(同时应该注意到的是，**Relu**进入负半区的时候，梯度为0，神经元此时不会训练，产生所谓的稀疏性，而**Leaky ReLu**不会有这问题)

$z$在**ReLu**的梯度一半都是0，但是，有足够的隐藏层使得z值大于0，所以对大多数的训练数据来说学习过程仍然可以很快。

快速概括一下不同激活函数的过程和结论。

**sigmoid**激活函数：除了输出层是一个二分类问题基本不会用它。

**tanh**激活函数：**tanh**是非常优秀的，几乎适合所有场合。

**ReLu**激活函数：最常用的默认函数，，如果不确定用哪个激活函数，就使用**ReLu**或者**Leaky ReLu**。公式3.23： $a = max( 0.01z,z)$ 为什么常数是0.01？当然，可以为学习算法选择不同的参数。

在选择自己神经网络的激活函数时，有一定的直观感受，在深度学习中的经常遇到一个问题：在编写神经网络的时候，会有很多选择：隐藏层单元的个数、激活函数的选择、初始化权值……这些选择想得到一个对比较好的指导原则是挺困难的。

鉴于以上三个原因，以及在工业界的见闻，提供一种直观的感受，哪一种工业界用的多，哪一种用的少。但是，自己的神经网络的应用，以及其特殊性，是很难提前知道选择哪些效果更好。所以通常的建议是：如果不确定哪一个激活函数效果更好，可以把它们都试试，然后在验证集或者发展集上进行评价。然后看哪一种表现的更好，就去使用它。

为自己的神经网络的应用测试这些不同的选择，会在以后检验自己的神经网络或者评估算法的时候，看到不同的效果。如果仅仅遵守使用默认的**ReLu**激活函数，而不要用其他的激励函数，那就可能在近期或者往后，每次解决问题的时候都使用相同的办法。



## 激活函数的导数

在神经网络中使用反向传播的时候，你真的需要计算激活函数的斜率或者导数。针对以下四种激活，求其导数如下：

1）**sigmoid activation function**

[![w600](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L1_week3_10.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L1_week3_10.png) 图3.8.1

其具体的求导如下： 公式3.25： $\frac{d}{dz}g(z) = {\frac{1}{1 + e^{-z}} (1-\frac{1}{1 + e^{-z}})}=g(z)(1-g(z))$

注：

当$z$ = 10或$z= -10$ ; $\frac{d}{dz}g(z)\approx0$

当$z $= 0 , $\frac{d}{dz}g(z)\text{=g(z)(1-g(z))=}{1}/{4}$

在神经网络中$a= g(z)$; $g{{(z)}^{'}}=\frac{d}{dz}g(z)=a(1-a)$

1. **Tanh activation function**

[![w600](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L1_week3_11.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L1_week3_11.png) 图3.8.2

其具体的求导如下： 公式3.26： $g(z) = tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} $

公式3.27： $\frac{d}{{d}z}g(z) = 1 - (tanh(z))^{2}$ 注：

当$z$ = 10或$z= -10$ $\frac{d}{dz}g(z)\approx0$

当$z$ = 0, $\frac{d}{dz}g(z)\text{=1-(0)=}1$

在神经网络中;

3）**Rectified Linear Unit (ReLU)**

[![w600](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L1_week3_12.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L1_week3_12.png) $g(z) =max (0,z)$

$$ g(z)^{'}= \begin{cases} 0& \text{if z < 0}\ 1& \text{if z > 0}\ undefined& \text{if z = 0} \end{cases} $$

注：通常在$z$= 0的时候给定其导数1,0；当然$z$=0的情况很少

4）**Leaky linear unit (Leaky ReLU)**

与**ReLU**类似 $$ g(z)=\max(0.01z,z) \ \ \ g(z)^{'}= \begin{cases} 0.01& \text{if z < 0}\ 1& \text{if z > 0}\ undefined& \text{if z = 0} \end{cases} $$

注：通常在$z = 0$的时候给定其导数1,0.01；当然$z=0$的情况很少。



## 神经网络的梯度下降

你的单隐层神经网络会有$W^{[1]}$，$b^{[1]}$，$W^{[2]}$，$b^{[2]}$这些参数，还有个$n_x$表示输入特征的个数，$n^{[1]}$表示隐藏单元个数，$n^{[2]}$表示输出单元个数。

在我们的例子中，我们只介绍过的这种情况，那么参数:

矩阵$W^{[1]}$的维度就是($n^{[1]}, n^{[0]}$)，$b^{[1]}$就是$n^{[1]}$维向量，可以写成$(n^{[1]}, 1)$，就是一个的列向量。 矩阵$W^{[2]}$的维度就是($n^{[2]}, n^{[1]}$)，$b^{[2]}$的维度就是$(n^{[2]},1)$维度。

你还有一个神经网络的成本函数，假设你在做二分类任务，那么你的成本函数等于：

**Cost function**: 公式： $J(W^{[1]},b^{[1]},W^{[2]},b^{[2]}) = {\frac{1}{m}}\sum_{i=1}^mL(\hat{y}, y)$ **loss function**和之前做**logistic**回归完全一样。

训练参数需要做梯度下降，在训练神经网络的时候，随机初始化参数很重要，而不是初始化成全零。当你参数初始化成某些值后，每次梯度下降都会循环计算以下预测值：

$\hat{y}^{(i)},(i=1,2,…,m)$ 公式3.28： $dW^{[1]} = \frac{dJ}{dW^{[1]}},db^{[1]} = \frac{dJ}{db^{[1]}}$ 公式3.29： ${d}W^{[2]} = \frac{{dJ}}{dW^{[2]}},{d}b^{[2]} = \frac{dJ}{db^{[2]}}$

其中

公式3.30： $W^{[1]}\implies{W^{[1]} - adW^{[1]}},b^{[1]}\implies{b^{[1]} -adb^{[1]}}$

公式3.31： $W^{[2]}\implies{W^{[2]} - \alpha{\rm d}W^{[2]}},b^{[2]}\implies{b^{[2]} - \alpha{\rm d}b^{[2]}}$ 正向传播方程如下（之前讲过）： **forward propagation**： (1) $z^{[1]} = W^{[1]}x + b^{[1]}$ (2) $a^{[1]} = \sigma(z^{[1]})$ (3) $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$ (4) $a^{[2]} = g^{[2]}(z^{[z]}) = \sigma(z^{[2]})$

反向传播方程如下:

**back propagation**： 公式3.32： $ dz^{[2]} = A^{[2]} - Y , Y = \begin{bmatrix}y^{[1]} & y^{[2]} & \cdots & y^{[m]}\ \end{bmatrix} $ 公式3.33： $ dW^{[2]} = {\frac{1}{m}}dz^{[2]}A^{[1]T} $ 公式3.34： $ {\rm d}b^{[2]} = {\frac{1}{m}}np.sum({d}z^{[2]},axis=1,keepdims=True)$ 公式3.35： $ dz^{[1]} = \underbrace{W^{[2]T}{\rm d}z^{[2]}}*{(n^{[1]},m)}\quad\*\underbrace{{g^{[1]}}^{'}}*{activation ; function ; of ; hidden ; layer}*\quad\underbrace{(z^{[1]})}*{(n^{[1]},m)} $ 公式3.36： $dW^{[1]} = {\frac{1}{m}}dz^{[1]}x^{T}$ 公式3.37： ${\underbrace{db^{[1]}}*{(n^{[1]},1)}} = {\frac{1}{m}}np.sum(dz^{[1]},axis=1,keepdims=True)$

上述是反向传播的步骤，注：这些都是针对所有样本进行过向量化，$Y$是$1×m$的矩阵；这里`np.sum`是python的numpy命令，`axis=1`表示水平相加求和，`keepdims`是防止**python**输出那些古怪的秩数$(n,)$，加上这个确保阵矩阵$db^{[2]}$这个向量输出的维度为$(n,1)$这样标准的形式。

目前为止，我们计算的都和**Logistic**回归十分相似，但当你开始计算反向传播时，你需要计算，是隐藏层函数的导数，输出在使用**sigmoid**函数进行二元分类。这里是进行逐个元素乘积，因为$W^{[2]T}dz^{[2]}$和$(z^{[1]})$这两个都为$(n^{[1]},m)$矩阵；

还有一种防止**python**输出奇怪的秩数，需要显式地调用`reshape`把`np.sum`输出结果写成矩阵形式。



## 前向传播和反向传播

先讲前向传播，输入${a}^{[l-1]}$，输出是${a}^{[l]}$，缓存为${z}^{[l]}$；从实现的角度来说我们可以缓存下${w}^{[l]}$和${b}^{[l]}$，这样更容易在不同的环节中调用函数。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/7cfc4b5fe94dcd9fe7130ac52701fed5.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/7cfc4b5fe94dcd9fe7130ac52701fed5.png)

所以前向传播的步骤可以写成： ${z}^{[l]}={W}^{[l]}\cdot{a}^{[l-1]}+{b}^{[l]}$

 ${{a}^{[l]}}={{g}^{[l]}}\left( {{z}^{[l]}}\right)$

向量化实现过程可以写成： ${z}^{[l]}={W}^{[l]}\cdot {A}^{[l-1]}+{b}^{[l]}$

 ${A}^{[l]}={g}^{[l]}({Z}^{[l]})$

前向传播需要喂入${A}^{[0]}$也就是$X$，来初始化；初始化的是第一层的输入值。${a}^{[0]}$对应于一个训练样本的输入特征，而${{A}^{[0]}}$对应于一整个训练样本的输入特征，所以这就是这条链的第一个前向函数的输入，重复这个步骤就可以从左到右计算前向传播。

下面讲反向传播的步骤：

输入为${{da}^{[l]}}$，输出为${{da}^{[l-1]}}$，${{dw}^{[l]}}$, ${{db}^{[l]}}$

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/c13d2a8fa258125a5398030c97101ee1.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/c13d2a8fa258125a5398030c97101ee1.png)

所以反向传播的步骤可以写成：

（1）$d{{z}^{[l]}}=d{{a}^{[l]}}*{{g}^{[l]}}'( {{z}^{[l]}})$

（2）$d{{w}^{[l]}}=d{{z}^{[l]}}\cdot{{a}^{[l-1]}}~$

（3）$d{{b}^{[l]}}=d{{z}^{[l]}}~~$

（4）$d{{a}^{[l-1]}}={{w}^{\left[ l \right]T}}\cdot {{dz}^{[l]}}$

（5）$d{{z}^{[l]}}={{w}^{[l+1]T}}d{{z}^{[l+1]}}\cdot \text{ }{{g}^{[l]}}'( {{z}^{[l]}})~$

式子（5）由式子（4）带入式子（1）得到，前四个式子就可实现反向函数。

向量化实现过程可以写成：

（6）$d{{Z}^{[l]}}=d{{A}^{[l]}}*{{g}^{\left[ l \right]}}'\left({{Z}^{[l]}} \right)~~$

（7）$d{{W}^{[l]}}=\frac{1}{m}\text{}d{{Z}^{[l]}}\cdot {{A}^{\left[ l-1 \right]T}}$

（8）$d{{b}^{[l]}}=\frac{1}{m}\text{ }np.sum(d{{z}^{[l]}},axis=1,keepdims=True)$

（9）$d{{A}^{[l-1]}}={{W}^{\left[ l \right]T}}.d{{Z}^{[l]}}$

总结一下：

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/53a5b4c71c0facfc8145af3b534f8583.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/53a5b4c71c0facfc8145af3b534f8583.png)

第一层你可能有一个**ReLU**激活函数，第二层为另一个**ReLU**激活函数，第三层可能是**sigmoid**函数（如果你做二分类的话），输出值为，用来计算损失；这样你就可以向后迭代进行反向传播求导来求${{dw}^{[3]}}$，${{db}^{[3]}}$ ，${{dw}^{[2]}}$ ，${{db}^{[2]}}$ ，${{dw}^{[1]}}$ ，${{db}^{[1]}}$。在计算的时候，缓存会把${{z}^{[1]}}$ ${{z}^{[2]}}$${{z}^{[3]}}$传递过来，然后回传${{da}^{[2]}}$，${{da}^{[1]}}$ ，可以用来计算${{da}^{[0]}}$，但我们不会使用它，这里讲述了一个三层网络的前向和反向传播，还有一个细节没讲就是前向递归——用输入数据来初始化，那么反向递归（使用**Logistic**回归做二分类）——对${{A}^{[l]}}$ 求导。



## drop正则化

除了$L2$正则化，还有一个非常实用的正则化方法——“**Dropout**（随机失活）”，我们来看看它的工作原理。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/97e37bf0d2893f890561cda932ba8c42.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/97e37bf0d2893f890561cda932ba8c42.png)

假设你在训练上图这样的神经网络，它存在过拟合，这就是**dropout**所要处理的，我们复制这个神经网络，**dropout**会遍历网络的每一层，并设置消除神经网络中节点的概率。假设网络中的每一层，每个节点都以抛硬币的方式设置概率，每个节点得以保留和消除的概率都是0.5，设置完节点概率，我们会消除一些节点，然后删除掉从该节点进出的连线，最后得到一个节点更少，规模更小的网络，然后用**backprop**方法进行训练。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/e45f9a948989b365650ddf16f62b097e.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/e45f9a948989b365650ddf16f62b097e.png)

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/9fa7196adeeaf88eb386fda2e9fa9909.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/9fa7196adeeaf88eb386fda2e9fa9909.png)

这是网络节点精简后的一个样本，对于其它样本，我们照旧以抛硬币的方式设置概率，保留一类节点集合，删除其它类型的节点集合。对于每个训练样本，我们都将采用一个精简后神经网络来训练它，这种方法似乎有点怪，单纯遍历节点，编码也是随机的，可它真的有效。不过可想而知，我们针对每个训练样本训练规模小得多的网络，最后你可能会认识到为什么要正则化网络，因为我们在训练规模小得多的网络。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/70b248490e496fed9b8d1d616e4d8303.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/70b248490e496fed9b8d1d616e4d8303.png)

如何实施**dropout**呢？方法有几种，接下来我要讲的是最常用的方法，即**inverted dropout**（反向随机失活），出于完整性考虑，我们用一个三层（$l=3$）网络来举例说明。编码中会有很多涉及到3的地方。我只举例说明如何在某一层中实施**dropout**。

首先要定义向量$d$，$d^{[3]}$表示网络第三层的**dropout**向量：

```
d3 = np.random.rand(a3.shape[0],a3.shape[1])
```

然后看它是否小于某数，我们称之为**keep-prob**，**keep-prob**是一个具体数字，上个示例中它是0.5，而本例中它是0.8，它表示保留某个隐藏单元的概率，此处**keep-prob**等于0.8，它意味着消除任意一个隐藏单元的概率是0.2，它的作用就是生成随机矩阵，如果对$a^{[3]}$进行因子分解，效果也是一样的。$d^{[3]}$是一个矩阵，每个样本和每个隐藏单元，其中$d^{[3]}$中的对应值为1的概率都是0.8，对应为0的概率是0.2，随机数字小于0.8。它等于1的概率是0.8，等于0的概率是0.2。

接下来要做的就是从第三层中获取激活函数，这里我们叫它$a^{[3]}$，$a^{[3]}$含有要计算的激活函数，$a^{[3]}$等于上面的$a^{[3]}$乘以$d^{[3]}$，`a3 =np.multiply(a3,d3)`，这里是元素相乘，也可写为$a3*=d3$，它的作用就是让$d^{[3]}$中所有等于0的元素（输出），而各个元素等于0的概率只有20%，乘法运算最终把$d^{\left\lbrack3 \right]}$中相应元素输出，即让$d^{[3]}$中0元素与$a^{[3]}$中相对元素归零。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/c75fe55c6bc17b60b00f5360aab180f4.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/c75fe55c6bc17b60b00f5360aab180f4.png)

如果用**python**实现该算法的话，$d^{[3]}$则是一个布尔型数组，值为**true**和**false**，而不是1和0，乘法运算依然有效，**python**会把**true**和**false**翻译为1和0，大家可以用**python**尝试一下。

最后，我们向外扩展$a^{[3]}$，用它除以0.8，或者除以**keep-prob**参数。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/6ba17ffcb3ff22a4a0ec857e66946086.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/6ba17ffcb3ff22a4a0ec857e66946086.png)

下面我解释一下为什么要这么做，为方便起见，我们假设第三隐藏层上有50个单元或50个神经元，在一维上$a^{[3]}$是50，我们通过因子分解将它拆分成$50×m$维的，保留和删除它们的概率分别为80%和20%，这意味着最后被删除或归零的单元平均有10（50×20%=10）个，现在我们看下$z^{\lbrack4]}$，$z^{[4]} = w^{[4]} a^{[3]} + b^{[4]}$，我们的预期是，$a^{[3]}$减少20%，也就是说$a^{[3]}$中有20%的元素被归零，为了不影响$z^{\lbrack4]}$的期望值，我们需要用$w^{[4]} a^{[3]}/0.8$，它将会修正或弥补我们所需的那20%，$a^{[3]}$的期望值不会变，划线部分就是所谓的**dropout**方法。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/272d902720be9993454c6d9a5f0bec49.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/272d902720be9993454c6d9a5f0bec49.png)

它的功能是，不论**keep-prop**的值是多少0.8，0.9甚至是1，如果**keep-prop**设置为1，那么就不存在**dropout**，因为它会保留所有节点。反向随机失活（**inverted dropout**）方法通过除以**keep-prob**，确保$a^{[3]}$的期望值不变。

事实证明，在测试阶段，当我们评估一个神经网络时，也就是用绿线框标注的反向随机失活方法，使测试阶段变得更容易，因为它的数据扩展问题变少，我们将在下节课讨论。

据我了解，目前实施**dropout**最常用的方法就是**Inverted dropout**，建议大家动手实践一下。**Dropout**早期的迭代版本都没有除以**keep-prob**，所以在测试阶段，平均值会变得越来越复杂，不过那些版本已经不再使用了。

现在你使用的是$d$向量，你会发现，不同的训练样本，清除不同的隐藏单元也不同。实际上，如果你通过相同训练集多次传递数据，每次训练数据的梯度不同，则随机对不同隐藏单元归零，有时却并非如此。比如，需要将相同隐藏单元归零，第一次迭代梯度下降时，把一些隐藏单元归零，第二次迭代梯度下降时，也就是第二次遍历训练集时，对不同类型的隐藏层单元归零。向量$d$或$d^{[3]}$用来决定第三层中哪些单元归零，无论用**foreprop**还是**backprop**，这里我们只介绍了**foreprob**。

如何在测试阶段训练算法，在测试阶段，我们已经给出了$x$，或是想预测的变量，用的是标准计数法。我用$a^{\lbrack0]}$，第0层的激活函数标注为测试样本$x$，我们在测试阶段不使用**dropout**函数，尤其是像下列情况：

$z^{[1]} = w^{[1]} a^{[0]} + b^{[1]}$

$a^{[1]} = g^{[1]}(z^{[1]})$

$z^{[2]} = \ w^{[2]} a^{[1]} + b^{[2]}$

$a^{[2]} = \ldots$

以此类推直到最后一层，预测值为$\hat{y}$。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/de49d4e160f29544055819c5ab1dd9c0.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/de49d4e160f29544055819c5ab1dd9c0.png)

显然在测试阶段，我们并未使用**dropout**，自然也就不用抛硬币来决定失活概率，以及要消除哪些隐藏单元了，因为在测试阶段进行预测时，我们不期望输出结果是随机的，如果测试阶段应用**dropout**函数，预测会受到干扰。理论上，你只需要多次运行预测处理过程，每一次，不同的隐藏单元会被随机归零，预测处理遍历它们，但计算效率低，得出的结果也几乎相同，与这个不同程序产生的结果极为相似。

**Inverted dropout**函数在除以**keep-prob**时可以记住上一步的操作，目的是确保即使在测试阶段不执行**dropout**来调整数值范围，激活函数的预期结果也不会发生变化，所以没必要在测试阶段额外添加尺度参数，这与训练阶段不同。

$l=keep-prob$

这就是**dropout**，大家可以通过本周的编程练习来执行这个函数，亲身实践一下。

为什么**dropout**会起作用呢？下节课我们将更加直观地了解**dropout**的具体功能。

## 理解 dropout（Understanding Dropout）

**Dropout**可以随机删除网络中的神经单元，他为什么可以通过正则化发挥如此大的作用呢？

直观上理解：不要依赖于任何一个特征，因为该单元的输入可能随时被清除，因此该单元通过这种方式传播下去，并为单元的四个输入增加一点权重，通过传播所有权重，**dropout**将产生收缩权重的平方范数的效果，和之前讲的$L2$正则化类似；实施**dropout**的结果实它会压缩权重，并完成一些预防过拟合的外层正则化；$L2$对不同权重的衰减是不同的，它取决于激活函数倍增的大小。

总结一下，**dropout**的功能类似于$L2$正则化，与$L2$正则化不同的是应用方式不同会带来一点点小变化，甚至更适用于不同的输入范围。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L2_week1_16.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L2_week1_16.png)

第二个直观认识是，我们从单个神经元入手，如图，这个单元的工作就是输入并生成一些有意义的输出。通过**dropout**，该单元的输入几乎被消除，有时这两个单元会被删除，有时会删除其它单元，就是说，我用紫色圈起来的这个单元，它不能依靠任何特征，因为特征都有可能被随机清除，或者说该单元的输入也都可能被随机清除。我不愿意把所有赌注都放在一个节点上，不愿意给任何一个输入加上太多权重，因为它可能会被删除，因此该单元将通过这种方式积极地传播开，并为单元的四个输入增加一点权重，通过传播所有权重，**dropout**将产生收缩权重的平方范数的效果，和我们之前讲过的$L2$正则化类似，实施**dropout**的结果是它会压缩权重，并完成一些预防过拟合的外层正则化。

事实证明，**dropout**被正式地作为一种正则化的替代形式，$L2$对不同权重的衰减是不同的，它取决于倍增的激活函数的大小。

总结一下，**dropout**的功能类似于$L2$正则化，与$L2$正则化不同的是，被应用的方式不同，**dropout**也会有所不同，甚至更适用于不同的输入范围。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/9b9f963a73e3ef9fcac008b179b6cf74.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/9b9f963a73e3ef9fcac008b179b6cf74.png)

实施**dropout**的另一个细节是，这是一个拥有三个输入特征的网络，其中一个要选择的参数是**keep-prob**，它代表每一层上保留单元的概率。所以不同层的**keep-prob**也可以变化。第一层，矩阵$W^{[1]}$是7×3，第二个权重矩阵$W^{[2]}$是7×7，第三个权重矩阵$W^{[3]}$是3×7，以此类推，$W^{[2]}$是最大的权重矩阵，因为$W^{[2]}$拥有最大参数集，即7×7，为了预防矩阵的过拟合，对于这一层，我认为这是第二层，它的**keep-prob**值应该相对较低，假设是0.5。对于其它层，过拟合的程度可能没那么严重，它们的**keep-prob**值可能高一些，可能是0.7，这里是0.7。如果在某一层，我们不必担心其过拟合的问题，那么**keep-prob**可以为1，为了表达清除，我用紫色线笔把它们圈出来，每层**keep-prob**的值可能不同。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5fcb1441866d6d3eff1164ed4ea38cfe.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/5fcb1441866d6d3eff1164ed4ea38cfe.png)

注意**keep-prob**的值是1，意味着保留所有单元，并且不在这一层使用**dropout**，对于有可能出现过拟合，且含有诸多参数的层，我们可以把**keep-prob**设置成比较小的值，以便应用更强大的**dropout**，有点像在处理$L2$正则化的正则化参数$\lambda$，我们尝试对某些层施行更多正则化，从技术上讲，我们也可以对输入层应用**dropout**，我们有机会删除一个或多个输入特征，虽然现实中我们通常不这么做，**keep-prob**的值为1，是非常常用的输入值，也可以用更大的值，或许是0.9。但是消除一半的输入特征是不太可能的，如果我们遵守这个准则，**keep-prob**会接近于1，即使你对输入层应用**dropout**。

总结一下，如果你担心某些层比其它层更容易发生过拟合，可以把某些层的**keep-prob**值设置得比其它层更低，缺点是为了使用交叉验证，你要搜索更多的超级参数，另一种方案是在一些层上应用**dropout**，而有些层不用**dropout**，应用**dropout**的层只含有一个超级参数，就是**keep-prob**。

结束前分享两个实施过程中的技巧，实施**dropout**，在计算机视觉领域有很多成功的第一次。计算视觉中的输入量非常大，输入太多像素，以至于没有足够的数据，所以**dropout**在计算机视觉中应用得比较频繁，有些计算机视觉研究人员非常喜欢用它，几乎成了默认的选择，但要牢记一点，**dropout**是一种正则化方法，它有助于预防过拟合，因此除非算法过拟合，不然我是不会使用**dropout**的，所以它在其它领域应用得比较少，主要存在于计算机视觉领域，因为我们通常没有足够的数据，所以一直存在过拟合，这就是有些计算机视觉研究人员如此钟情于**dropout**函数的原因。直观上我认为不能概括其它学科。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/13987e28fa537a110c6b123fc4455f7c.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/13987e28fa537a110c6b123fc4455f7c.png)

**dropout**一大缺点就是代价函数$J$不再被明确定义，每次迭代，都会随机移除一些节点，如果再三检查梯度下降的性能，实际上是很难进行复查的。定义明确的代价函数$J$每次迭代后都会下降，因为我们所优化的代价函数$J$实际上并没有明确定义，或者说在某种程度上很难计算，所以我们失去了调试工具来绘制这样的图片。我通常会关闭**dropout**函数，将**keep-prob**的值设为1，运行代码，确保J函数单调递减。然后打开**dropout**函数，希望在**dropout**过程中，代码并未引入**bug**。我觉得你也可以尝试其它方法，虽然我们并没有关于这些方法性能的数据统计，但你可以把它们与**dropout**方法一起使用

## 其他正则化方法

一.数据扩增

假设你正在拟合猫咪图片分类器，如果你想通过扩增训练数据来解决过拟合，但扩增数据代价高，而且有时候我们无法扩增数据，但我们可以通过添加这类图片来增加训练集。例如，水平翻转图片，并把它添加到训练集。所以现在训练集中有原图，还有翻转后的这张图片，所以通过水平翻转图片，训练集则可以增大一倍，因为训练集有冗余，这虽然不如我们额外收集一组新图片那么好，但这样做节省了获取更多猫咪图片的花费。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/2e2ccf4a2bb9156f876321ce07b006ca.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/2e2ccf4a2bb9156f876321ce07b006ca.png)

除了水平翻转图片，你也可以随意裁剪图片，这张图是把原图旋转并随意放大后裁剪的，仍能辨别出图片中的猫咪。

通过随意翻转和裁剪图片，我们可以增大数据集，额外生成假训练数据。和全新的，独立的猫咪图片数据相比，这些额外的假的数据无法包含像全新数据那么多的信息，但我们这么做基本没有花费，代价几乎为零，除了一些对抗性代价。以这种方式扩增算法数据，进而正则化数据集，减少过拟合比较廉价。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/dd9a0f8209e53fdab8030b98d39e11eb.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/dd9a0f8209e53fdab8030b98d39e11eb.png)

像这样人工合成数据的话，我们要通过算法验证，图片中的猫经过水平翻转之后依然是猫。大家注意，我并没有垂直翻转，因为我们不想上下颠倒图片，也可以随机选取放大后的部分图片，猫可能还在上面。

对于光学字符识别，我们还可以通过添加数字，随意旋转或扭曲数字来扩增数据，把这些数字添加到训练集，它们仍然是数字。为了方便说明，我对字符做了强变形处理，所以数字4看起来是波形的，其实不用对数字4做这么夸张的扭曲，只要轻微的变形就好，我做成这样是为了让大家看的更清楚。实际操作的时候，我们通常对字符做更轻微的变形处理。因为这几个4看起来有点扭曲。所以，数据扩增可作为正则化方法使用，实际功能上也与正则化相似。

二.**early stopping**

还有另外一种常用的方法叫作**early stopping**，运行梯度下降时，我们可以绘制训练误差，或只绘制代价函数$J$的优化过程，在训练集上用0-1记录分类误差次数。呈单调下降趋势，如图。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/69d92a8de8f62ab602d2bc022591d3c9.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/69d92a8de8f62ab602d2bc022591d3c9.png)

因为在训练过程中，我们希望训练误差，代价函数$J$都在下降，通过**early stopping**，我们不但可以绘制上面这些内容，还可以绘制验证集误差，它可以是验证集上的分类误差，或验证集上的代价函数，逻辑损失和对数损失等，你会发现，验证集误差通常会先呈下降趋势，然后在某个节点处开始上升，**early stopping**的作用是，你会说，神经网络已经在这个迭代过程中表现得很好了，我们在此停止训练吧，得到验证集误差，它是怎么发挥作用的？

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/9d0db64a9c9b050466a039c935f36f93.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/9d0db64a9c9b050466a039c935f36f93.png)

当你还未在神经网络上运行太多迭代过程的时候，参数$w$接近0，因为随机初始化$w$值时，它的值可能都是较小的随机值，所以在你长期训练神经网络之前$w$依然很小，在迭代过程和训练过程中$w$的值会变得越来越大，比如在这儿，神经网络中参数$w$的值已经非常大了，所以**early stopping**要做就是在中间点停止迭代过程，我们得到一个$w$值中等大小的弗罗贝尼乌斯范数，与$L2$正则化相似，选择参数w范数较小的神经网络，但愿你的神经网络过度拟合不严重。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/51ca931387ed9fb80313263113c56e8e.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/51ca931387ed9fb80313263113c56e8e.png)

术语**early stopping**代表提早停止训练神经网络，训练神经网络时，我有时会用到**early stopping**，但是它也有一个缺点，我们来了解一下。

我认为机器学习过程包括几个步骤，其中一步是选择一个算法来优化代价函数$J$，我们有很多种工具来解决这个问题，如梯度下降，后面我会介绍其它算法，例如**Momentum**，**RMSprop**和**Adam**等等，但是优化代价函数$J$之后，我也不想发生过拟合，也有一些工具可以解决该问题，比如正则化，扩增数据等等。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/f5fd5df8235145c54aece1a5bf7b31f6.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/f5fd5df8235145c54aece1a5bf7b31f6.png)

在机器学习中，超级参数激增，选出可行的算法也变得越来越复杂。我发现，如果我们用一组工具优化代价函数$J$，机器学习就会变得更简单，在重点优化代价函数$J$时，你只需要留意$w$和$b$，$J(w,b)$的值越小越好，你只需要想办法减小这个值，其它的不用关注。然后，预防过拟合还有其他任务，换句话说就是减少方差，这一步我们用另外一套工具来实现，这个原理有时被称为“正交化”。思路就是在一个时间做一个任务，后面课上我会具体介绍正交化，如果你还不了解这个概念，不用担心。

但对我来说**early stopping**的主要缺点就是你不能独立地处理这两个问题，因为提早停止梯度下降，也就是停止了优化代价函数$J$，因为现在你不再尝试降低代价函数$J$，所以代价函数$J$的值可能不够小，同时你又希望不出现过拟合，你没有采取不同的方式来解决这两个问题，而是用一种方法同时解决两个问题，这样做的结果是我要考虑的东西变得更复杂。

如果不用**early stopping**，另一种方法就是$L2$正则化，训练神经网络的时间就可能很长。我发现，这导致超级参数搜索空间更容易分解，也更容易搜索，但是缺点在于，你必须尝试很多正则化参数$\lambda$的值，这也导致搜索大量$\lambda$值的计算代价太高。

**Early stopping**的优点是，只运行一次梯度下降，你可以找出$w$的较小值，中间值和较大值，而无需尝试$L2$正则化超级参数$\lambda$的很多值。

如果你还不能完全理解这个概念，没关系，下节课我们会详细讲解正交化，这样会更好理解。

虽然$L2$正则化有缺点，可还是有很多人愿意用它。吴恩达老师个人更倾向于使用$L2$正则化，尝试许多不同的$\lambda$值，假设你可以负担大量计算的代价。而使用**early stopping**也能得到相似结果，还不用尝试这么多$\lambda$值。

## 归一化输入（Normalizing inputs）

训练神经网络，其中一个加速训练的方法就是归一化输入。假设一个训练集有两个特征，输入特征为2维，归一化需要两个步骤：

1. 零均值

2. 归一化方差；

   我们希望无论是训练集和测试集都是通过相同的$μ$和$σ^2$定义的数据转换，这两个是由训练集得出来的。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L2_week1_19.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L2_week1_19.png)

第一步是零均值化，$\mu = \frac{1}{m}\sum_{i =1}^{m}x^{(i)}$，它是一个向量，$x$等于每个训练数据 $x$减去$\mu$，意思是移动训练集，直到它完成零均值化。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/5e49434607f22caf087f7730177931bf.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/5e49434607f22caf087f7730177931bf.png)

第二步是归一化方差，注意特征$x_{1}$的方差比特征$x_{2}$的方差要大得多，我们要做的是给$\sigma$赋值，$\sigma^{2}= \frac{1}{m}\sum_{i =1}^{m}{({x^{(i)})}^{2}}$，这是节点$y$ 的平方，$\sigma^{2}$是一个向量，它的每个特征都有方差，注意，我们已经完成零值均化，$({x^{(i)})}^{2}$元素$y^{2}$就是方差，我们把所有数据除以向量$\sigma^{2}$，最后变成上图形式。

$x_{1}$和$x_{2}$的方差都等于1。提示一下，如果你用它来调整训练数据，那么用相同的 $μ$ 和 $\sigma^{2}$来归一化测试集。尤其是，你不希望训练集和测试集的归一化有所不同，不论$μ$的值是什么，也不论$\sigma^{2}$的值是什么，这两个公式中都会用到它们。所以你要用同样的方法调整测试集，而不是在训练集和测试集上分别预估$μ$ 和 $\sigma^{2}$。因为我们希望不论是训练数据还是测试数据，都是通过相同μ和$\sigma^{2}$定义的相同数据转换，其中$μ$和$\sigma^{2}$是由训练集数据计算得来的。

我们为什么要这么做呢？为什么我们想要归一化输入特征，回想一下右上角所定义的代价函数。

$J(w,b)=\frac{1}{m}\sum\limits_{i=1}^{m}{L({{{\hat{y}}}^{(i)}},{{y}^{(i)}})}$

如果你使用非归一化的输入特征，代价函数会像这样：

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/bc4eccfb6c9dbef6cc81636d5ce60390.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/bc4eccfb6c9dbef6cc81636d5ce60390.png)

这是一个非常细长狭窄的代价函数，你要找的最小值应该在这里。但如果特征值在不同范围，假如$x_{1}$取值范围从1到1000，特征$x_{2}$的取值范围从0到1，结果是参数$w_{1}$和$w_{2}$值的范围或比率将会非常不同，这些数据轴应该是$w_{1}$和$w_{2}$，但直观理解，我标记为$w$和$b$，代价函数就有点像狭长的碗一样，如果你能画出该函数的部分轮廓，它会是这样一个狭长的函数。

然而如果你归一化特征，代价函数平均起来看更对称，如果你在上图这样的代价函数上运行梯度下降法，你必须使用一个非常小的学习率。因为如果是在这个位置，梯度下降法可能需要多次迭代过程，直到最后找到最小值。但如果函数是一个更圆的球形轮廓，那么不论从哪个位置开始，梯度下降法都能够更直接地找到最小值，你可以在梯度下降法中使用较大步长，而不需要像在左图中那样反复执行。

当然，实际上$w$是一个高维向量，因此用二维绘制$w$并不能正确地传达并直观理解，但总地直观理解是代价函数会更圆一些，而且更容易优化，前提是特征都在相似范围内，而不是从1到1000，0到1的范围，而是在-1到1范围内或相似偏差，这使得代价函数$J$优化起来更简单快速。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/4d0c183882a140ecd205f1618243d7f8.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/4d0c183882a140ecd205f1618243d7f8.png)

实际上如果假设特征$x_{1}$范围在0-1之间，$x_{2}$的范围在-1到1之间，$x_{3}$范围在1-2之间，它们是相似范围，所以会表现得很好。

当它们在非常不同的取值范围内，如其中一个从1到1000，另一个从0到1，这对优化算法非常不利。但是仅将它们设置为均化零值，假设方差为1，就像上一张幻灯片里设定的那样，确保所有特征都在相似范围内，通常可以帮助学习算法运行得更快。

所以如果输入特征处于不同范围内，可能有些特征值从0到1，有些从1到1000，那么归一化特征值就非常重要了。如果特征值处于相似范围内，那么归一化就不是很重要了。执行这类归一化并不会产生什么危害，我通常会做归一化处理，虽然我不确定它能否提高训练或算法速度。

这就是归一化特征输入，下节课我们将继续讨论提升神经网络训练速度的方法。

## 梯度消失/梯度爆炸（Vanishing / Exploding gradients）

训练神经网络，尤其是深度神经所面临的一个问题就是梯度消失或梯度爆炸，也就是你训练神经网络的时候，导数或坡度有时会变得非常大，或者非常小，甚至于以指数方式变小，这加大了训练的难度。

这节课，你将会了解梯度消失或梯度爆炸的真正含义，以及如何更明智地选择随机初始化权重，从而避免这个问题。 假设你正在训练这样一个极深的神经网络，为了节约幻灯片上的空间，我画的神经网络每层只有两个隐藏单元，但它可能含有更多，但这个神经网络会有参数$W^{[1]}$，$W^{[2]}$，$W^{[3]}$等等，直到$W^{[l]}$，为了简单起见，假设我们使用激活函数$g(z)=z$，也就是线性激活函数，我们忽略$b$，假设$b^{[l]}$=0，如果那样的话，输出$ y=W^{[l]}W^{[L -1]}W^{[L - 2]}\ldots W^{[3]}W^{[2]}W^{[1]}x$，如果你想考验我的数学水平，$W^{[1]} x = z^{[1]}$，因为$b=0$，所以我想$z^{[1]} =W^{[1]} x$，$a^{[1]} = g(z^{[1]})$，因为我们使用了一个线性激活函数，它等于$z^{[1]}$，所以第一项$W^{[1]} x = a^{[1]}$，通过推理，你会得出$W^{[2]}W^{[1]}x =a^{[2]}$，因为$a^{[2]} = g(z^{[2]})$，还等于$g(W^{[2]}a^{[1]})$，可以用$W^{[1]}x$替换$a^{[1]}$，所以这一项就等于$a^{[2]}$，这个就是$a^{[3]}$($W^{[3]}W^{[2]}W^{[1]}x$)。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/fc03196f0b6d1c9f56fa39d0d462cfa4.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/fc03196f0b6d1c9f56fa39d0d462cfa4.png)

所有这些矩阵数据传递的协议将给出$\hat y$而不是$y$的值。

假设每个权重矩阵$W^{[l]} = \begin{bmatrix} 1.5 & 0 \\0 & 1.5 \end{bmatrix}$，从技术上来讲，最后一项有不同维度，可能它就是余下的权重矩阵，$y= W^{[1]}\begin{bmatrix} 1.5 & 0 \\ 0 & 1.5 \end{bmatrix}^{(L -1)}x$，因为我们假设所有矩阵都等于它，它是1.5倍的单位矩阵，最后的计算结果就是$\hat{y}$，$\hat{y}$也就是等于${1.5}^{(L-1)}x$。如果对于一个深度神经网络来说$L$值较大，那么$\hat{y}$的值也会非常大，实际上它呈指数级增长的，它增长的比率是${1.5}^{L}$，因此对于一个深度神经网络，$y$的值将爆炸式增长。

相反的，如果权重是0.5，$W^{[l]} = \begin{bmatrix} 0.5& 0 \ 0 & 0.5 \ \end{bmatrix}$，它比1小，这项也就变成了${0.5}^{L}$，矩阵$y= W^{[1]}\begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix}^{(L - 1)}x$，再次忽略$W^{[L]}$，因此每个矩阵都小于1，假设$x_{1}$和$x_{2}$都是1，激活函数将变成$\frac{1}{2}$，$\frac{1}{2}$，$\frac{1}{4}$，$\frac{1}{4}$，$\frac{1}{8}$，$\frac{1}{8}$等，直到最后一项变成$\frac{1}{2^{L}}$，所以作为自定义函数，激活函数的值将以指数级下降，它是与网络层数数量$L$相关的函数，在深度网络中，激活函数以指数级递减。

我希望你得到的直观理解是，权重$W$只比1略大一点，或者说只是比单位矩阵大一点，深度神经网络的激活函数将爆炸式增长，如果$W$比1略小一点，可能是$\begin{bmatrix}0.9 & 0 \ 0 & 0.9 \ \end{bmatrix}$。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/ff87a8ea1377053128cdbf5129f0203f.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/ff87a8ea1377053128cdbf5129f0203f.png)

在深度神经网络中，激活函数将以指数级递减，虽然我只是讨论了激活函数以与$L$相关的指数级数增长或下降，它也适用于与层数$L$相关的导数或梯度函数，也是呈指数级增长或呈指数递减。

对于当前的神经网络，假设$L=150$，最近**Microsoft**对152层神经网络的研究取得了很大进展，在这样一个深度神经网络中，如果激活函数或梯度函数以与$L$相关的指数增长或递减，它们的值将会变得极大或极小，从而导致训练难度上升，尤其是梯度指数小于$L$时，梯度下降算法的步长会非常非常小，梯度下降算法将花费很长时间来学习。

总结一下，我们讲了深度神经网络是如何产生梯度消失或爆炸问题的，实际上，在很长一段时间内，它曾是训练深度神经网络的阻力，虽然有一个不能彻底解决此问题的解决方案，但是已在如何选择初始化权重问题上提供了很多帮助。

## 神经网络的权重初始化

上节课，我们学习了深度神经网络如何产生梯度消失和梯度爆炸问题，最终针对该问题，我们想出了一个不完整的解决方案，虽然不能彻底解决问题，却很有用，有助于我们为神经网络更谨慎地选择随机初始化参数，为了更好地理解它，我们先举一个神经单元初始化地例子，然后再演变到整个深度网络。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/db9472c81a2cf6bb704dc398ea1cf017.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/db9472c81a2cf6bb704dc398ea1cf017.png)

我们来看看只有一个神经元的情况，然后才是深度网络。

单个神经元可能有4个输入特征，从$x_{1}$到$x_{4}$，经过$a=g(z)$处理，最终得到$\hat{y}$，稍后讲深度网络时，这些输入表示为$a^{[l]}$，暂时我们用$x$表示。

$z = w_{1}x_{1} + w_{2}x_{2} + \ldots +w_{n}x_{n}$，$b=0$，暂时忽略$b$，为了预防$z$值过大或过小，你可以看到$n$越大，你希望$w_{i}$越小，因为$z$是$w_{i}x_{i}$的和，如果你把很多此类项相加，希望每项值更小，最合理的方法就是设置$w_{i}=\frac{1}{n}$，$n$表示神经元的输入特征数量，实际上，你要做的就是设置某层权重矩阵$w^{[l]} = np.random.randn( \text{shape})*\text{np.}\text{sqrt}(\frac{1}{n^{[l-1]}})$，$n^{[l - 1]}$就是我喂给第$l$层神经单元的数量（即第$l-1$层神经元数量）。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/e4114d7dc1c6242bd96cdadb457b8959.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/e4114d7dc1c6242bd96cdadb457b8959.png)

结果，如果你是用的是**Relu**激活函数，而不是$\frac{1}{n}$，方差设置为$\frac{2}{n}$，效果会更好。你常常发现，初始化时，尤其是使用**Relu**激活函数时，$g^{[l]}(z) =Relu(z)$,它取决于你对随机变量的熟悉程度，这是高斯随机变量，然后乘以它的平方根，也就是引用这个方差$\frac{2}{n}$。这里，我用的是$n^{[l - 1]}$，因为本例中，逻辑回归的特征是不变的。但一般情况下$l$层上的每个神经元都有$n^{[l - 1]}$个输入。如果激活函数的输入特征被零均值和标准方差化，方差是1，$z$也会调整到相似范围，这就没解决问题（梯度消失和爆炸问题）。但它确实降低了梯度消失和爆炸问题，因为它给权重矩阵$w$设置了合理值，你也知道，它不能比1大很多，也不能比1小很多，所以梯度没有爆炸或消失过快。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/3decc82bf6f48903ee1fedbdb7e2c0f6.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/3decc82bf6f48903ee1fedbdb7e2c0f6.png)

我提到了其它变体函数，刚刚提到的函数是**Relu**激活函数，一篇由**Herd**等人撰写的论文曾介绍过。对于几个其它变体函数，如**tanh**激活函数，有篇论文提到，常量1比常量2的效率更高，对于**tanh**函数来说，它是$\sqrt{\frac{1}{n^{[l-1]}}}$，这里平方根的作用与这个公式作用相同($\text{np.}\text{sqrt}(\frac{1}{n^{[l-1]}})$)，它适用于**tanh**激活函数，被称为**Xavier**初始化。**Yoshua Bengio**和他的同事还提出另一种方法，你可能在一些论文中看到过，它们使用的是公式$\sqrt{\frac{2}{n^{[l-1]} + n^{\left[l\right]}}}$。其它理论已对此证明，但如果你想用**Relu**激活函数，也就是最常用的激活函数，我会用这个公式$\text{np.}\text{sqrt}(\frac{2}{n^{[l-1]}})$，如果使用**tanh**函数，可以用公式$\sqrt{\frac{1}{n^{[l-1]}}}$，有些作者也会使用这个函数。

实际上，我认为所有这些公式只是给你一个起点，它们给出初始化权重矩阵的方差的默认值，如果你想添加方差，方差参数则是另一个你需要调整的超级参数，可以给公式$\text{np.}\text{sqrt}(\frac{2}{n^{[l-1]}})$添加一个乘数参数，调优作为超级参数激增一份子的乘子参数。有时调优该超级参数效果一般，这并不是我想调优的首要超级参数，但我发现调优过程中产生的问题，虽然调优该参数能起到一定作用，但考虑到相比调优，其它超级参数的重要性，我通常把它的优先级放得比较低。

希望你现在对梯度消失或爆炸问题以及如何为权重初始化合理值已经有了一个直观认识，希望你设置的权重矩阵既不会增长过快，也不会太快下降到0，从而训练出一个权重或梯度不会增长或消失过快的深度网络。我们在训练深度网络时，这也是一个加快训练速度的技巧。

## 梯度的数值逼近

在实施**backprop**时，有一个测试叫做梯度检验，它的作用是确保**backprop**正确实施。因为有时候，你虽然写下了这些方程式，却不能100%确定，执行**backprop**的所有细节都是正确的。为了逐渐实现梯度检验，我们首先说说如何计算梯度的数值逼近，下节课，我们将讨论如何在**backprop**中执行梯度检验，以确保**backprop**正确实施。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/91cde35a0fc9c11a98f16ad2797f20a7.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/91cde35a0fc9c11a98f16ad2797f20a7.png)

我们先画出函数$f$，标记为$f\left( \theta \right)$，$f\left( \theta \right)=\theta^{3}$，先看一下$\theta$的值，假设$\theta=1$，不增大$\theta$的值，而是在$\theta$ 右侧，设置一个$\theta +\varepsilon$，在$\theta$左侧，设置$\theta -\varepsilon$。因此$\theta=1$，$\theta +\varepsilon =1.01,\theta -\varepsilon =0.99$,，跟以前一样，$\varepsilon$的值为0.01，看下这个小三角形，计算高和宽的比值，就是更准确的梯度预估，选择$f$函数在$\theta -\varepsilon$上的这个点，用这个较大三角形的高比上宽，技术上的原因我就不详细解释了，较大三角形的高宽比值更接近于$\theta$的导数，把右上角的三角形下移，好像有了两个三角形，右上角有一个，左下角有一个，我们通过这个绿色大三角形同时考虑了这两个小三角形。所以我们得到的不是一个单边公差而是一个双边公差。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/b24ad9ce9fdc4a32be4d98915d8f6ffb.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/b24ad9ce9fdc4a32be4d98915d8f6ffb.png)

我们写一下数据算式，图中绿色三角形上边的点的值是$f( \theta +\varepsilon )$，下边的点是$f( \theta-\varepsilon)$，这个三角形的高度是$f( \theta +\varepsilon)-f(\theta -\varepsilon)$，这两个宽度都是ε，所以三角形的宽度是$2\varepsilon$，高宽比值为$\frac{f(\theta + \varepsilon ) - (\theta -\varepsilon)}{2\varepsilon}$，它的期望值接近$g( \theta)$，$f( \theta)=\theta^{3}$传入参数值，$\frac {f\left( \theta + \varepsilon \right) - f(\theta -\varepsilon)}{2\varepsilon} = \frac{{(1.01)}^{3} - {(0.99)}^{3}}{2 \times0.01}$，大家可以暂停视频，用计算器算算结果，结果应该是3.0001，而前面一张幻灯片上面是，当$\theta =1$时，$g( \theta)=3\theta^{2} =3$，所以这两个$g(\theta)$值非常接近，逼近误差为0.0001，前一张幻灯片，我们只考虑了单边公差，即从$\theta $到$\theta +\varepsilon$之间的误差，$g( \theta)$的值为3.0301，逼近误差是0.03，不是0.0001，所以使用双边误差的方法更逼近导数，其结果接近于3，现在我们更加确信，$g( \theta)$可能是$f$导数的正确实现，在梯度检验和反向传播中使用该方法时，最终，它与运行两次单边公差的速度一样，实际上，我认为这种方法还是非常值得使用的，因为它的结果更准确。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/35a2e86581eab4665634593870fe5180.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/35a2e86581eab4665634593870fe5180.png)

这是一些你可能比较熟悉的微积分的理论，如果你不太明白我讲的这些理论也没关系，导数的官方定义是针对值很小的$\varepsilon$，导数的官方定义是$f^{'}\theta) = \operatorname{}\frac{f( \theta + \varepsilon) -f(\theta -\varepsilon)}{2\varepsilon}$，如果你上过微积分课，应该学过无穷尽的定义，我就不在这里讲了。

对于一个非零的$\varepsilon$，它的逼近误差可以写成$O(\varepsilon^{2})$，ε值非常小，如果$\varepsilon=0.01$，$\varepsilon^{2}=0.0001$，大写符号$O$的含义是指逼近误差其实是一些常量乘以$\varepsilon^{2}$，但它的确是很准确的逼近误差，所以大写$O$的常量有时是1。然而，如果我们用另外一个公式逼近误差就是$O(\varepsilon)$，当$\varepsilon$小于1时，实际上$\varepsilon$比$\varepsilon^{2}$大很多，所以这个公式近似值远没有左边公式的准确，所以在执行梯度检验时，我们使用双边误差，即$\frac{f\left(\theta + \varepsilon \right) - f(\theta -\varepsilon)}{2\varepsilon}$，而不使用单边公差，因为它不够准确。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/1b2617461ed10089bc61ad273c84594f.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/1b2617461ed10089bc61ad273c84594f.png)

如果你不理解上面两条结论，所有公式都在这儿，不用担心，如果你对微积分和数值逼近有所了解，这些信息已经足够多了，重点是要记住，双边误差公式的结果更准确，下节课我们做梯度检验时就会用到这个方法。

今天我们讲了如何使用双边误差来判断别人给你的函数$g( \theta)$，是否正确实现了函数$f$的偏导，现在我们可以使用这个方法来检验反向传播是否得以正确实施，如果不正确，它可能有bug需要你来解决。

## 梯度检验

梯度检验帮我们节省了很多时间，也多次帮我发现**backprop**实施过程中的bug，接下来，我们看看如何利用它来调试或检验**backprop**的实施是否正确。

假设你的网络中含有下列参数，$W^{[1]}$和$b^{[1]}$……$W^{[l]}$和$b^{[l]}$，为了执行梯度检验，首先要做的就是，把所有参数转换成一个巨大的向量数据，你要做的就是把矩阵$W$转换成一个向量，把所有$W$矩阵转换成向量之后，做连接运算，得到一个巨型向量$\theta$，该向量表示为参数$\theta$，代价函数$J$是所有$W$和$b$的函数，现在你得到了一个$\theta$的代价函数$J$（即$J(\theta)$）。接着，你得到与$W$和$b$顺序相同的数据，你同样可以把$dW^{[1]}$和${db}^{[1]}$……${dW}^{[l]}$和${db}^{[l]}$转换成一个新的向量，用它们来初始化大向量$d\theta$，它与$\theta$具有相同维度。

同样的，把$dW^{[1]}$转换成矩阵，$db^{[1]}$已经是一个向量了，直到把${dW}^{[l]}$转换成矩阵，这样所有的$dW$都已经是矩阵，注意$dW^{[1]}$与$W^{[1]}$具有相同维度，$db^{[1]}$与$b^{[1]}$具有相同维度。经过相同的转换和连接运算操作之后，你可以把所有导数转换成一个大向量$d\theta$，它与$\theta$具有相同维度，现在的问题是$d\theta$和代价函数$J$的梯度或坡度有什么关系？

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/94fb79f5694c802a5261298e54157e14.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/94fb79f5694c802a5261298e54157e14.png)

这就是实施梯度检验的过程，英语里通常简称为“**grad check**”，首先，我们要清楚$J$是超参数$\theta$的一个函数，你也可以将J函数展开为$J(\theta_{1},\theta_{2},\theta_{3},\ldots\ldots)$，不论超级参数向量$\theta$的维度是多少，为了实施梯度检验，你要做的就是循环执行，从而对每个$i$也就是对每个$\theta$组成元素计算$d\theta_{\text{approx}}[i]$的值，我使用双边误差，也就是

$d\theta_{\text{approx}}\left[i \right] = \frac{J\left( \theta_{1},\theta_{2},\ldots\theta_{i} + \varepsilon,\ldots \right) - J\left( \theta_{1},\theta_{2},\ldots\theta_{i} - \varepsilon,\ldots \right)}{2\varepsilon}$

只对$\theta_{i}$增加$\varepsilon$，其它项保持不变，因为我们使用的是双边误差，对另一边做同样的操作，只不过是减去$\varepsilon$，$\theta$其它项全都保持不变。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/f56f0cdc90730068109a6e20fcd41724.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/f56f0cdc90730068109a6e20fcd41724.png)

从上节课中我们了解到这个值（$d\theta_{\text{approx}}\left[i \right]$）应该逼近$d\theta\left[i \right]$=$\frac{\partial J}{\partial\theta_{i}}$，$d\theta\left[i \right]$是代价函数的偏导数，然后你需要对i的每个值都执行这个运算，最后得到两个向量，得到$d\theta$的逼近值$d\theta_{\text{approx}}$，它与$d\theta$具有相同维度，它们两个与$\theta$具有相同维度，你要做的就是验证这些向量是否彼此接近。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/bed27b1d469acaf24dc1a7c1a7cdc086.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/bed27b1d469acaf24dc1a7c1a7cdc086.png)

具体来说，如何定义两个向量是否真的接近彼此？我一般做下列运算，计算这两个向量的距离，$d\theta_{\text{approx}}\left[i \right] - d\theta[i]$的欧几里得范数，注意这里（${||d\theta_{\text{approx}} -d\theta||}_{2}$）没有平方，它是误差平方之和，然后求平方根，得到欧式距离，然后用向量长度归一化，使用向量长度的欧几里得范数。分母只是用于预防这些向量太小或太大，分母使得这个方程式变成比率，我们实际执行这个方程式，$\varepsilon$可能为$10^{-7}$，使用这个取值范围内的$\varepsilon$，如果你发现计算方程式得到的值为$10^{-7}$或更小，这就很好，这就意味着导数逼近很有可能是正确的，它的值非常小。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/f223102bc7d144b66ad9b2ffa4a85f00.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/f223102bc7d144b66ad9b2ffa4a85f00.png)

如果它的值在$10^{-5}$范围内，我就要小心了，也许这个值没问题，但我会再次检查这个向量的所有项，确保没有一项误差过大，可能这里有**bug**。

如果左边这个方程式结果是$10^{-3}$，我就会担心是否存在**bug**，计算结果应该比$10^{- 3}$小很多，如果比$10^{-3}$大很多，我就会很担心，担心是否存在**bug**。这时应该仔细检查所有$\theta$项，看是否有一个具体的$i$值，使得$d\theta_{\text{approx}}\left[i \right]$与$ d\theta[i]$大不相同，并用它来追踪一些求导计算是否正确，经过一些调试，最终结果会是这种非常小的值（$10^{-7}$），那么，你的实施可能是正确的。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/07fd11e7b28ef0ab7d3b6d23b19a8fdc.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/07fd11e7b28ef0ab7d3b6d23b19a8fdc.png)

在实施神经网络时，我经常需要执行**foreprop**和**backprop**，然后我可能发现这个梯度检验有一个相对较大的值，我会怀疑存在**bug**，然后开始调试，调试，调试，调试一段时间后，我得到一个很小的梯度检验值，现在我可以很自信的说，神经网络实施是正确的。

现在你已经了解了梯度检验的工作原理，它帮助我在神经网络实施中发现了很多**bug**，希望它对你也有所帮助。

## 梯度检验应用的注意事项

这节课，分享一些关于如何在神经网络实施梯度检验的实用技巧和注意事项。

[![img](https://github.com/fengdu78/deeplearning_ai_books/raw/master/images/L2_week1_24.png)](https://github.com/fengdu78/deeplearning_ai_books/blob/master/images/L2_week1_24.png)

首先，不要在训练中使用梯度检验，它只用于调试。我的意思是，计算所有$i$值的$d\theta_{\text{approx}}\left[i\right]$是一个非常漫长的计算过程，为了实施梯度下降，你必须使用$W$和$b$ **backprop**来计算$d\theta$，并使用**backprop**来计算导数，只要调试的时候，你才会计算它，来确认数值是否接近$d\theta$。完成后，你会关闭梯度检验，梯度检验的每一个迭代过程都不执行它，因为它太慢了。

第二点，如果算法的梯度检验失败，要检查所有项，检查每一项，并试着找出**bug**，也就是说，如果$d\theta_{\text{approx}}\left[i\right]$与dθ[i]的值相差很大，我们要做的就是查找不同的i值，看看是哪个导致$d\theta_{\text{approx}}\left[i\right]$与$d\theta\left[i\right]$的值相差这么多。举个例子，如果你发现，相对某些层或某层的$\theta$或$d\theta$的值相差很大，但是$\text{dw}^{[l]}$的各项非常接近，注意$\theta$的各项与$b$和$w$的各项都是一一对应的，这时，你可能会发现，在计算参数$b$的导数$db$的过程中存在**bug**。反过来也是一样，如果你发现它们的值相差很大，$d\theta_{\text{approx}}\left[i\right]$的值与$d\theta\left[i\right]$的值相差很大，你会发现所有这些项目都来自于$dw$或某层的$dw$，可能帮你定位bug的位置，虽然未必能够帮你准确定位bug的位置，但它可以帮助你估测需要在哪些地方追踪**bug**。

第三点，在实施梯度检验时，如果使用正则化，请注意正则项。如果代价函数$J(\theta) = \frac{1}{m}\sum_{}^{}{L(\hat y^{(i)},y^{(i)})} + \frac{\lambda}{2m}\sum_{}^{}{||W^{[l]}||}^{2}$，这就是代价函数$J$的定义，$d\theta$等于与$\theta$相关的$J$函数的梯度，包括这个正则项，记住一定要包括这个正则项。

第四点，梯度检验不能与**dropout**同时使用，因为每次迭代过程中，**dropout**会随机消除隐藏层单元的不同子集，难以计算**dropout**在梯度下降上的代价函数$J$。因此**dropout**可作为优化代价函数$J$的一种方法，但是代价函数J被定义为对所有指数极大的节点子集求和。而在任何迭代过程中，这些节点都有可能被消除，所以很难计算代价函数$J$。你只是对成本函数做抽样，用**dropout**，每次随机消除不同的子集，所以很难用梯度检验来双重检验**dropout**的计算，所以我一般不同时使用梯度检验和**dropout**。如果你想这样做，可以把**dropout**中的**keepprob**设置为1.0，然后打开**dropout**，并寄希望于**dropout**的实施是正确的，你还可以做点别的，比如修改节点丢失模式确定梯度检验是正确的。实际上，我一般不这么做，我建议关闭**dropout**，用梯度检验进行双重检查，在没有**dropout**的情况下，你的算法至少是正确的，然后打开**dropout**。

最后一点，也是比较微妙的一点，现实中几乎不会出现这种情况。当$w$和$b$接近0时，梯度下降的实施是正确的，在随机初始化过程中……，但是在运行梯度下降时，$w$和$b$变得更大。可能只有在$w$和$b$接近0时，**backprop**的实施才是正确的。但是当$W$和$b$变大时，它会变得越来越不准确。你需要做一件事，我不经常这么做，就是在随机初始化过程中，运行梯度检验，然后再训练网络，$w$和$b$会有一段时间远离0，如果随机初始化值比较小，反复训练网络之后，再重新运行梯度检验。