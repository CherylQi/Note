# 机器学习

一个程序被认为能从经验E中学习，解决任务T，达到性能度量值P，当且仅当，有了经验E后，经过P评判，程序在处理T时的性能有所提升。

**监督学习：** 给学习算法一个数据集，该数据集由正确答案组成。

**无监督学习：** 已知数据但是没有任何标签。 

**回归：** 推测出连续值属性。

**分类：** 推测出离散的输出值。

常用变量说明：

$m$ 代表训练集中实例的数量

$x$ 代表特征/输入变量

$y$ 代表目标变量/输出变量

$\left( x,y \right)$ 代表训练集中的实例

$({{x}^{(i)}},{{y}^{(i)}})$ 代表第$i$ 个观察实例

$h$ 代表学习算法的解决方案或函数也称为假设（**hypothesis**）

## 预测函数

$h_\theta \left( x \right)=\theta_{0} + \theta_{1}x$ 

## 目标函数

 $J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$ 

* $\frac{1}{m}$为了不依赖元素数量  
* $\frac{1}{2}$ 为了后面方便求导

![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/10ba90df2ada721cf1850ab668204dc9.png)

## 梯度下降

 随机选择一个参数的组合$\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。我们持续这么做直到找到一个局部最小值（**local minimum**），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（**global minimum**），选择不同的初始参数组合，可能会找到不同的局部最小值。

批量梯度下降（**batch gradient descent**）算法的公式为：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/7da5a5f635b1eb552618556f1b4aac1a.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/7da5a5f635b1eb552618556f1b4aac1a.png)

其中$a$是学习率（**learning rate**），它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ef4227864e3cabb9a3938386f857e938.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ef4227864e3cabb9a3938386f857e938.png)

在梯度下降算法中，还有一个更微妙的问题，梯度下降中，我们要更新${\theta_{0}}$和${\theta_{1}}$ ，当 $j=0$ 和$j=1$时，会产生更新，所以你将更新$J\left( {\theta_{0}} \right)$和$J\left( {\theta_{1}} \right)$。实现梯度下降算法的微妙之处是，在这个表达式中，如果你要更新这个等式，你需要同时更新${\theta_{0}}$和${\theta_{1}}$，我的意思是在这个等式中，我们要这样更新：

${\theta_{0}}$:= ${\theta_{0}}$ ，并更新${\theta_{1}}$:= ${\theta_{1}}$。

实现方法是：你应该计算公式右边的部分，通过那一部分计算出${\theta_{0}}$和${\theta_{1}}$的值，然后同时更新${\theta_{0}}$和${\theta_{1}}$。

让我进一步阐述这个过程：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/13176da01bb25128c91aca5476c9d464.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/13176da01bb25128c91aca5476c9d464.png)

梯度下降算法如下：

${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)$

描述：对$\theta $赋值，使得$J\left( \theta \right)$按梯度下降最快方向进行，一直迭代下去，最终得到局部最小值。其中$a$是学习率（**learning rate**），它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ee916631a9f386e43ef47efafeb65b0f.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ee916631a9f386e43ef47efafeb65b0f.png)

对于这个问题，求导的目的，基本上可以说取这个红点的切线，就是这样一条红色的直线，刚好与函数相切于这一点，让我们看看这条红色直线的斜率，就是这条刚好与函数曲线相切的这条直线，这条直线的斜率正好是这个三角形的高度除以这个水平长度，现在，这条线有一个正斜率，也就是说它有正导数，因此，我得到的新的${\theta_{1}}$，${\theta_{1}}$更新后等于${\theta_{1}}$减去一个正数乘以$a$。

让我们来看看如果$a$太小或$a$太大会出现什么情况：

如果$a$太小了，即我的学习速率太小，结果就是只能这样像小宝宝一样一点点地挪动，去努力接近最低点，这样就需要很多步才能到达最低点，所以如果$a$太小的话，可能会很慢，因为它会一点点挪动，它会需要很多步才能到达全局最低点。

如果$a$太大，那么梯度下降法可能会越过最低点，甚至可能无法收敛，下一次迭代又移动了一大步，越过一次，又越过一次，一次次越过最低点，直到你发现实际上离最低点越来越远，所以，如果$a$太大，它会导致无法收敛，甚至发散。

随着梯度下降法的运行，你移动的幅度会自动变得越来越小，直到最终移动幅度非常小，你会发现，已经收敛到局部极小值。

回顾一下，在梯度下降法中，当我们接近局部最低点时，梯度下降法会自动采取更小的幅度，这是因为当我们接近局部最低点时，很显然在局部最低时导数等于零，所以当我们接近局部最低时，导数值会自动变得越来越小，所以梯度下降将自动采取较小的幅度，这就是梯度下降的做法。所以实际上没有必要再另外减小$a$。

## 线性回归的梯度下降

梯度下降算法和线性回归算法比较如图：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5eb364cc5732428c695e2aa90138b01b.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5eb364cc5732428c695e2aa90138b01b.png)

对我们之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：

$\frac{\partial }{\partial {{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$ 时：$\frac{\partial }{\partial {{\theta }_{0}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$

$j=1$ 时：$\frac{\partial }{\partial {{\theta }_{1}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

则算法改写成：

**Repeat {**

 ${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot1}$ 

 ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

 **}** 

## 线性代数

### 矩阵和向量

如图：这个是4×2矩阵，即4行2列，如$m$为行，$n$为列，那么$m×n$即4×2

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/9fa04927c2bd15780f92a7fafb539179.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/9fa04927c2bd15780f92a7fafb539179.png)

矩阵的维数即行数×列数

矩阵元素（矩阵项）：$A=\left[ \begin{matrix} 1402 & 191 \\ 1371 & 821 \\ 949 & 1437 \\ 147 & 1448 \end{matrix} \right]$ 

$A_{ij}$指第$i$行，第$j$列的元素。

向量是一种特殊的矩阵，讲义中的向量一般都是列向量，如： $y=\left[ \begin{matrix} {460} \\ {232} \\ {315} \\ {178} \end{matrix} \right]$

为四维列向量（4×1）。

如下图为1索引向量和0索引向量，左图为1索引向量，右图为0索引向量，一般我们用1索引向量。

$y=\left[ \begin{matrix} {{y}_{1}} \\ {{y}_{2}} \\ {{y}_{3}} \\ {{y}_{4}} \end{matrix} \right]$，$y=\left[ \begin{matrix} {{y}_{0}} \\ {{y}_{1}} \\ {{y}_{2}} \\ {{y}_{3}} \end{matrix} \right]$ 

###  加法和标量乘法

矩阵的加法：行列数相等的可以加。

例：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ffddfddfdfd.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ffddfddfdfd.png)

矩阵的乘法：每个元素都要乘

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fdddddd.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/fdddddd.png)

组合算法也类似。

###  矩阵向量乘法

矩阵和向量的乘法如图：$m×n$的矩阵乘以$n×1$的向量，得到的是$m×1$的向量

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/437ae2333f00286141abe181a1b7c44a.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/437ae2333f00286141abe181a1b7c44a.png)

算法举例：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b2069e4b3e12618f5405500d058118d7.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/b2069e4b3e12618f5405500d058118d7.png)

### 矩阵乘法

矩阵乘法：

$m×n$矩阵乘以$n×o$矩阵，变成$m×o$矩阵。

如果这样说不好理解的话就举一个例子来说明一下，比如说现在有两个矩阵$A$和$B$，那么它们的乘积就可以表示为图中所示的形式。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1a9f98df1560724713f6580de27a0bde.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/1a9f98df1560724713f6580de27a0bde.jpg) [![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5ec35206e8ae22668d4b4a3c3ea7b292.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5ec35206e8ae22668d4b4a3c3ea7b292.jpg)

###  矩阵乘法的性质

矩阵乘法的性质：

矩阵的乘法不满足交换律：$A×B≠B×A$

矩阵的乘法满足结合律。即：$A×(B×C)=(A×B)×C$

单位矩阵：在矩阵的乘法中，有一种矩阵起着特殊的作用，如同数的乘法中的1,我们称这种矩阵为单位矩阵．它是个方阵，一般用 $I$ 或者 $E$ 表示，本讲义都用 $I$ 代表单位矩阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1以外全都为0。如：

$A{{A}^{-1}}={{A}^{-1}}A=I$

对于单位矩阵，有$AI=IA=A$

###  逆、转置

矩阵的逆：如矩阵$A$是一个$m×m$矩阵（方阵），如果有逆矩阵，则：$A{{A}^{-1}}={{A}^{-1}}A=I$

我们一般在**OCTAVE**或者**MATLAB**中进行计算矩阵的逆矩阵。

矩阵的转置：设$A$为$m×n$阶矩阵（即$m$行$n$列），第$i $行$j $列的元素是$a(i,j)$，即：$A=a(i,j)$

定义$A$的转置为这样一个$n×m$阶矩阵$B$，满足$B=a(j,i)$，即 $b (i,j)=a(j,i)$（$B$的第$i$行第$j$列元素是$A$的第$j$行第$i$列元素），记${{A}^{T}}=B$。(有些书记为A'=B）

直观来看，将$A$的所有元素绕着一条从第1行第1列元素出发的右下方45度的射线作镜面反转，即得到$A$的转置。

例：

${{\left| \begin{matrix} a& b \\ c& d \\ e& f \end{matrix} \right|}^{T}}=\left|\begin{matrix} a& c & e \\ b& d & f \end{matrix} \right|$ 

矩阵的转置基本性质:

$ {{\left( A\pm B \right)}^{T}}={{A}^{T}}\pm {{B}^{T}} $ ${{\left( A\times B \right)}^{T}}={{B}^{T}}\times {{A}^{T}}$ ${{\left( {{A}^{T}} \right)}^{T}}=A $ ${{\left( KA \right)}^{T}}=K{{A}^{T}} $

**matlab**中矩阵转置：直接打一撇，`x=y'`。

## 多变量梯度下降

与单变量线性回归类似，在多变量线性回归中，我们也构建一个代价函数，则这个代价函数是所有建模误差的平方和，即：$J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ ，

其中：$h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$ ，

我们的目标和单变量线性回归问题中一样，是要找出使得代价函数最小的一系列参数。 多变量线性回归的批量梯度下降算法为：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/41797ceb7293b838a3125ba945624cf6.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/41797ceb7293b838a3125ba945624cf6.png)

即：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/6bdaff07783e37fcbb1f8765ca06b01b.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/6bdaff07783e37fcbb1f8765ca06b01b.png)

求导数后得到：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/dd33179ceccbd8b0b59a5ae698847049.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/dd33179ceccbd8b0b59a5ae698847049.png)

当$n>=1$时， ${{\theta }_{0}}:={{\theta }_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)}$

${{\theta }_{1}}:={{\theta }_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{1}^{(i)}$

${{\theta }_{2}}:={{\theta }_{2}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{2}^{(i)}$

我们开始随机选择一系列的参数值，计算所有的预测结果后，再给所有的参数一个新的值，如此循环直到收敛。

代码示例：

计算代价函数 $J\left( \theta \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {h_{\theta}}\left( {x^{(i)}} \right)-{y^{(i)}} \right)}^{2}}}$ 其中：${h_{\theta}}\left( x \right)={\theta^{T}}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

**Python** 代码：

```python
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
```

## 梯度下降-特征缩放

在我们面对多维特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。

以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/966e5a9b00687678374b8221fdd33475.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/966e5a9b00687678374b8221fdd33475.jpg) 

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。如图：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b8167ff0926046e112acf789dba98057.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/b8167ff0926046e112acf789dba98057.png)   



最简单的方法是令：${{x}*{n}}=\frac{{{x}*{n}}-{{\mu}*{n}}}{{{s}*{n}}}$，其中 ${\mu_{n}}$是平均值，${s_{n}}$是标准差。

## 梯度下降-学习率

梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，我们不能提前预知，我们可以绘制迭代次数和代价函数的图表来观测算法在何时趋于收敛。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/cd4e3df45c34f6a8e2bb7cd3a2849e6c.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/cd4e3df45c34f6a8e2bb7cd3a2849e6c.jpg)

也有一些自动测试是否收敛的方法，例如将代价函数的变化值与某个阀值（例如0.001）进行比较，但通常看上面这样的图表更好。

梯度下降算法的每次迭代受到学习率的影响，如果学习率$a$过小，则达到收敛所需的迭代次数会非常高；如果学习率$a$过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。

通常可以考虑尝试些学习率：

$\alpha=0.01，0.03，0.1，0.3，1，3，10$

## 正规方程

对于某些线性回归问题，正规方程方法是更好的解决方案。如：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/a47ec797d8a9c331e02ed90bca48a24b.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/a47ec797d8a9c331e02ed90bca48a24b.png)

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0$ 。 假设我们的训练集特征矩阵为 $X$（包含了 ${{x}_{0}}=1$）并且我们的训练集结果为向量 $y$，则利用正规方程解出向量 $\theta ={{\left( {X^T}X \right)}^{-1}}{X^{T}}y$ 。 上标**T**代表矩阵转置，上标-1 代表矩阵的逆。设矩阵$A={X^{T}}X$，则：${{\left( {X^T}X \right)}^{-1}}={A^{-1}}$ 以下表示数据为例：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/261a11d6bce6690121f26ee369b9e9d1.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/261a11d6bce6690121f26ee369b9e9d1.png)

即：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c8eedc42ed9feb21fac64e4de8d39a06.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/c8eedc42ed9feb21fac64e4de8d39a06.png)

运用正规方程方法求解参数：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b62d24a1f709496a6d7c65f87464e911.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/b62d24a1f709496a6d7c65f87464e911.jpg)

注：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。

梯度下降与正规方程的比较：

| 梯度下降                      | 正规方程                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| 需要选择学习率$\alpha$        | 不需要                                                       |
| 需要多次迭代                  | 一次运算得出                                                 |
| 当特征数量$n$大时也能较好适用 | 需要计算${{\left( {{X}^{T}}X \right)}^{-1}}$ 如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为$O\left( {{n}^{3}} \right)$，通常来说当$n$小于10000 时还是可以接受的 |
| 适用于各种类型的模型          | 只适用于线性模型，不适合逻辑回归模型等其他模型               |

总结一下，只要特征变量的数目并不大，标准方程是一个很好的计算参数$\theta $的替代方法。具体地说，只要特征变量数量小于一万，我通常使用标准方程法，而不使用梯度下降法。

正规方程的**python**实现：

```python
import numpy as np
    
 def normalEqn(X, y):
    
   theta = np.linalg.inv(X.T@X)@X.T@y #X.T@X等价于X.T.dot(X)
    
   return theta
```



# 逻辑回归

预测的变量为离散的值，实际上用于分类。算法的输出值永远在0~1之间。

逻辑回归模型的假设是： $h_\theta \left( x \right)=g\left(\theta^{T}X \right)$ 其中： $X$ 代表特征向量 $g$ 代表逻辑函数（**logistic function**)是一个常用的逻辑函数为**S**形函数（**Sigmoid function**），公式为： $g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。

**python**代码实现：

```
import numpy as np
    
def sigmoid(z):
    
   return 1 / (1 + np.exp(-z))
```

该函数的图像为：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1073efb17b0d053b4f9218d4393246cc.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/1073efb17b0d053b4f9218d4393246cc.jpg)

合起来，我们得到逻辑回归模型的假设：

对模型的理解： $g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。

$h_\theta \left( x \right)$的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性（**estimated probablity**）即$h_\theta \left( x \right)=P\left( y=1|x;\theta \right)$ 例如，如果对于给定的$x$，通过已经确定的参数计算得出$h_\theta \left( x \right)=0.7$，则表示有70%的几率$y$为正向类，相应地$y$为负向类的几率为1-0.7=0.3。

## 代价函数

将${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数（**non-convexfunction**）。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8b94e47b7630ac2b0bcb10d204513810.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/8b94e47b7630ac2b0bcb10d204513810.jpg)

这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。

线性回归的代价函数为：$J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ 。 我们重新定义逻辑回归的代价函数为：$J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}$，其中

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/54249cb51f0086fa6a805291bf2639f1.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/54249cb51f0086fa6a805291bf2639f1.png)

**log可以把指数变为线性，保证凸函数。**  

${h_\theta}\left( x \right)$与 $Cost\left( {h_\theta}\left( x \right),y \right)$之间的关系如下图所示：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ffa56adcc217800d71afdc3e0df88378.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ffa56adcc217800d71afdc3e0df88378.jpg)       横轴为预测值，纵轴为代价

这样构建的$Cost\left( {h_\theta}\left( x \right),y \right)$函数的特点是：当实际的 $y=1$ 且${h_\theta}\left( x \right)$也为 1 时误差为 0，当 $y=1$ 但${h_\theta}\left( x \right)$不为1时误差随着${h_\theta}\left( x \right)$变小而变大；当实际的 $y=0$ 且${h_\theta}\left( x \right)$也为 0 时代价为 0，当$y=0$ 但${h_\theta}\left( x \right)$不为 0时误差随着 ${h_\theta}\left( x \right)$的变大而变大。 将构建的 $Cost\left( {h_\theta}\left( x \right),y \right)$简化如下： $Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$ 带入代价函数得到： $J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$ 即：$J\left( \theta \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

**Python**代码实现：

```
import numpy as np
    
def cost(theta, X, y):
    
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```

在得到这样一个代价函数以后，我们便可以用梯度下降算法来求得能使代价函数最小的参数了。算法为：

**Repeat** { $\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)$ (**simultaneously update all** ) }

求导后得到：

**Repeat** { $\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}$ **(simultaneously update all** ) }

在这个视频中，我们定义了单训练样本的代价函数，凸性分析的内容是超出这门课的范围的，但是可以证明我们所选的代价值函数会给我们一个凸优化问题。代价函数$J(\theta)$会是一个凸函数，并且没有局部最优值。

推导过程：

$J\left( \theta \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$ 考虑： ${h_\theta}\left( {{x}^{(i)}} \right)=\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}}$ 则： ${{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)$ $={{y}^{(i)}}\log \left( \frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)$ $=-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^T}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^T}{{x}^{(i)}}}} \right)$

所以： $\frac{\partial }{\partial {\theta_{j}}}J\left( \theta \right)=\frac{\partial }{\partial {\theta_{j}}}[-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^{T}}{{x}^{(i)}}}} \right)]}]$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\frac{-x_{j}^{(i)}{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}{1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}]$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{{y}^{(i)}}\frac{x_j^{(i)}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}]$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}x_j^{(i)}-x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}+{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}\left( 1\text{+}{{e}^{{\theta^T}{{x}^{(i)}}}} \right)-{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}x_j^{(i)}}$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$ $=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}-{h_\theta}\left( {{x}^{(i)}} \right)]x_j^{(i)}}$ $=\frac{1}{m}\sum\limits_{i=1}^{m}{[{h_\theta}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_j^{(i)}}$

注：虽然得到的梯度下降算法表面上看上去与线性回归的梯度下降算法一样，但是这里的${h_\theta}\left( x \right)=g\left( {\theta^T}X \right)$与线性回归中不同，所以实际上是不一样的。另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的。特征缩放的方法，也适用于逻辑回归。如果你的特征范围差距很大的话，那么应用特征缩放的方法，同样也可以让逻辑回归中，梯度下降收敛更快。

一些梯度下降算法之外的选择： 除了梯度下降算法以外，还有一些常被用来令代价函数最小的算法，这些算法更加复杂和优越，而且通常不需要人工选择学习率，通常比梯度下降算法要更加快速。这些算法有：**共轭梯度**（**Conjugate Gradient**），**局部优化法**(**Broyden fletcher goldfarb shann,BFGS**)和**有限内存局部优化法**(**LBFGS**) ，**fminunc**是 **matlab**和**octave** 中都带的一个最小值优化函数，使用时我们需要提供代价函数和每个参数的求导。



## 多类别分类：一对多

为了能实现这样的转变，我们将多个类中的一个类标记为正向类（$y=1$），然后将其他所有类都标记为负向类，这个模型记作$h_\theta^{\left( 1 \right)}\left( x \right)$。接着，类似地第我们选择另一个类标记为正向类（$y=2$），再将其它类都标记为负向类，将这个模型记作 $h_\theta^{\left( 2 \right)}\left( x \right)$,依此类推。 最后我们得到一系列的模型简记为： $h_\theta^{\left( i \right)}\left( x \right)=p\left( y=i|x;\theta \right)$其中：$i=\left( 1,2,3....k \right)$

最后，在我们需要做预测时，我们将所有的分类机都运行一遍，然后对每一个输入变量，都选择最高可能性的输出变量。

总之，我们已经把要做的做完了，现在要做的就是训练这个逻辑回归分类器：$h_\theta^{\left( i \right)}\left( x \right)$， 其中 $i$ 对应每一个可能的 $y=i$，最后，为了做出预测，我们给出输入一个新的 $x$ 值，用这个做预测。我们要做的就是在我们三个分类器里面输入 $x$，然后我们选择一个让 $h_\theta^{\left( i \right)}\left( x \right)$ 最大的$ i$，即$\mathop{\max}\limits_i,h_\theta^{\left( i \right)}\left( x \right)$。



## 正则化

解决过拟合：

1. 丢弃一些不能帮助我们正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来帮忙（例如**PCA**）
2. 正则化。 保留所有的特征，但是减少参数的大小（**magnitude**）。

正则化线性回归的代价函数为：

$J\left( \theta \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{[({{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta _{j}^{2}})]}$

如果我们要使用梯度下降法令这个代价函数最小化，因为我们未对$\theta_0$进行正则化，所以梯度下降算法将分两种情形：

$Repeat$ $until$ $convergence${

 ${\theta_0}:={\theta_0}-a\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{0}^{(i)}})$

 ${\theta_j}:={\theta_j}-a[\frac{1}{m}\sum\limits_{i=1}^{m}{(({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}+\frac{\lambda }{m}{\theta_j}]$

 $for$ $j=1,2,...n$

 }

对上面的算法中$ j=1,2,...,n$ 时的更新式子进行调整可得：

${\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}$ 可以看出，正则化线性回归的梯度下降算法的变化在于，每次都在原有算法更新规则的基础上令$\theta $值减少了一个额外的值。

我们同样也可以利用正规方程来求解正则化线性回归模型，方法如下所示：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/71d723ddb5863c943fcd4e6951114ee3.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/71d723ddb5863c943fcd4e6951114ee3.png)

图中的矩阵尺寸为 $(n+1)*(n+1)$。

**Python**代码：

```python
import numpy as np

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```



# 神经网络

神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫激活单元，**activation unit**）采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被成为权重（**weight**）。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c2233cd74605a9f8fe69fd59547d3853.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/c2233cd74605a9f8fe69fd59547d3853.jpg)

我们设计出了类似于神经元的神经网络，效果如下：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fbb4ffb48b64468c384647d45f7b86b5.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/fbb4ffb48b64468c384647d45f7b86b5.png)

其中$x_1$, $x_2$, $x_3$是输入单元（**input units**），我们将原始数据输入给它们。 $a_1$, $a_2$, $a_3$是中间单元，它们负责将数据进行处理，然后呈递到下一层。 最后是输出单元，它负责计算${h_\theta}\left( x \right)$。

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为输入层（**Input Layer**），最后一层称为输出层（**Output Layer**），中间一层成为隐藏层（**Hidden Layers**）。我们为每一层都增加一个偏差单位（**bias unit**）：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8293711e1d23414d0a03f6878f5a2d91.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/8293711e1d23414d0a03f6878f5a2d91.jpg)

下面引入一些标记法来帮助描述模型： $a_{i}^{\left( j \right)}$ 代表第$j$ 层的第 $i$ 个激活单元。${{\theta }^{\left( j \right)}}$代表从第 $j$ 层映射到第$ j+1$ 层时的权重的矩阵，例如${{\theta }^{\left( 1 \right)}}$代表从第一层映射到第二层的权重的矩阵。其尺寸为：以第 $j+1$层的激活单元数量为行数，以第 $j$ 层的激活单元数加一为列数的矩阵。例如：上图所示的神经网络中${{\theta }^{\left( 1 \right)}}$的尺寸为 3*4。

对于上图所示的模型，激活单元和输出分别表达为：

$a_{1}^{(2)}=g(\Theta _{10}^{(1)}{{x}_{0}}+\Theta _{11}^{(1)}{{x}_{1}}+\Theta _{12}^{(1)}{{x}_{2}}+\Theta _{13}^{(1)}{{x}_{3}})$

 $a_{2}^{(2)}=g(\Theta _{20}^{(1)}{{x}_{0}}+\Theta _{21}^{(1)}{{x}_{1}}+\Theta _{22}^{(1)}{{x}_{2}}+\Theta _{23}^{(1)}{{x}_{3}})$ 

$a_{3}^{(2)}=g(\Theta _{30}^{(1)}{{x}_{0}}+\Theta _{31}^{(1)}{{x}_{1}}+\Theta _{32}^{(1)}{{x}_{2}}+\Theta_{33}^{(1)}{{x}_{3}})$ 

${{h}_{\Theta }}(x)=g(\Theta _{10}^{(2)}a_{0}^{(2)}+\Theta _{11}^{(2)}a_{1}^{(2)}+\Theta _{12}^{(2)}a_{2}^{(2)}+\Theta _{13}^{(2)}a_{3}^{(2)})$

为了更好了了解**Neuron Networks**的工作原理，我们先把左半部分遮住：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/6167ad04e696c400cb9e1b7dc1e58d8a.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/6167ad04e696c400cb9e1b7dc1e58d8a.png)

右半部分其实就是以$a_0, a_1, a_2, a_3$, 按照**Logistic Regression**的方式输出$h_\theta(x)$：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/10342b472803c339a9e3bc339188c5b8.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/10342b472803c339a9e3bc339188c5b8.png)

其实神经网络就像是**logistic regression**，只不过我们把**logistic regression**中的输入向量$\left[ x_1\sim {x_3} \right]$ 变成了中间层的$\left[ a_1^{(2)}\sim a_3^{(2)} \right]$, 即: $h_\theta(x)=g\left( \Theta_0^{\left( 2 \right)}a_0^{\left( 2 \right)}+\Theta_1^{\left( 2 \right)}a_1^{\left( 2 \right)}+\Theta_{2}^{\left( 2 \right)}a_{2}^{\left( 2 \right)}+\Theta_{3}^{\left( 2 \right)}a_{3}^{\left( 2 \right)} \right)$ 我们可以把$a_0, a_1, a_2, a_3$看成更为高级的特征值，也就是$x_0, x_1, x_2, x_3$的进化体，并且它们是由 $x$与$\theta$决定的，因为是梯度下降的，所以$a$是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅将 $x$次方厉害，也能更好的预测新数据。 这就是神经网络相比于逻辑回归和线性回归的优势。

我们可以利用神经元来组合成更为复杂的神经网络以实现更复杂的运算。例如我们要实现**XNOR** 功能（输入的两个值必须一样，均为1或均为0），即 $\text{XNOR}=( \text{x}_1, \text{AND}, \text{x}_2 ), \text{OR} \left( \left( \text{NOT}, \text{x}_1 \right) \text{AND} \left( \text{NOT}, \text{x}_2 \right) \right)$ 首先构造一个能表达$\left( \text{NOT}, \text{x}_1 \right) \text{AND} \left( \text{NOT}, \text{x}_2 \right)$部分的神经元：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/4c44e69a12b48efdff2fe92a0a698768.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/4c44e69a12b48efdff2fe92a0a698768.png)

然后将表示 **AND** 的神经元和表示$\left( \text{NOT}, \text{x}_1 \right) \text{AND} \left( \text{NOT}, \text{x}_2 \right)$的神经元以及表示 OR 的神经元进行组合：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/432c906875baca78031bd337fe0c8682.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/432c906875baca78031bd337fe0c8682.png)

我们就得到了一个能实现 $\text{XNOR}$ 运算符功能的神经网络。

按这种方法我们可以逐渐构造出越来越复杂的函数，也能得到更加厉害的特征值。

这就是神经网络的厉害之处。

## 多分类

当我们有不止两种分类时（也就是$y=1,2,3….$），比如以下这种情况，该怎么办？如果我们要训练一个神经网络算法来识别路人、汽车、摩托车和卡车，在输出层我们应该有4个值。例如，第一个值为1或0用于预测是否是行人，第二个值用于判断是否为汽车。

输入向量$x$有三个维度，两个中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现${{\left[ a\text{ }b\text{ }c\text{ }d \right]}^{T}}$，且$a,b,c,d$中仅有一个为1，表示当前类。下面是该神经网络的可能结构示例：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f3236b14640fa053e62c73177b3474ed.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/f3236b14640fa053e62c73177b3474ed.jpg)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/685180bf1774f7edd2b0856a8aae3498.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/685180bf1774f7edd2b0856a8aae3498.png)

神经网络算法的输出结果为四种可能情形之一：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5e1a39d165f272b7f145c68ef78a3e13.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5e1a39d165f272b7f145c68ef78a3e13.png)



## 代价函数

假设神经网络的训练样本有$m$个，每个包含一组输入$x$和一组输出信号$y$，$L$表示神经网络层数，$S_I$表示每层的**neuron**个数($S_l$表示输出层神经元个数)，$S_L$代表最后一层中处理单元的个数。

神经网络中的代价函数为：

$\newcommand{\subk}[1]{ #1_k }$ $$h_\theta\left(x\right)\in \mathbb{R}^{K}$$ $${\left({h_\theta}\left(x\right)\right)}_{i}={i}^{th} \text{output}$$

$J(\Theta) = -\frac{1}{m} \left[ \sum\limits_{i=1}^{m} \sum\limits_{k=1}^{k} {y_k}^{(i)} \log \subk{(h_\Theta(x^{(i)}))} + \left( 1 - y_k^{(i)} \right) \log \left( 1- \subk{\left( h_\Theta \left( x^{(i)} \right) \right)} \right) \right] + \frac{\lambda}{2m} \sum\limits_{l=1}^{L-1} \sum\limits_{i=1}^{s_l} \sum\limits_{j=1}^{s_{l+1}} \left( \Theta_{ji}^{(l)} \right)^2$ 

正则化的那一项只是排除了每一层$\theta_0$后，每一层的$\theta$ 矩阵的和。最里层的循环$j$循环所有的行（由$s_{l+1}$ 层的激活单元数决定），循环$i$则循环所有的列，由该层（$s_l$层）的激活单元数所决定。即：$h_\theta(x)$与真实值之间的距离为每个样本-每个类输出的加和，对参数进行**regularization**的**bias**项处理所有参数的平方和。

## 反向传播算法

为了计算代价函数的偏导数$\frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right)$，我们需要采用一种反向传播算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。

推导https://blog.csdn.net/qq_29762941/article/details/80343185

假设我们的训练集只有一个样本$\left({x}^{(1)},{y}^{(1)}\right)$，我们的神经网络是一个四层的神经网络，其中$K=4，S_{L}=4，L=4$：

前向传播算法：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2ea8f5ce4c3df931ee49cf8d987ef25d.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/2ea8f5ce4c3df931ee49cf8d987ef25d.png)

下面的公式推导过程见：https://blog.csdn.net/qq_29762941/article/details/80343185

我们从最后一层的误差开始计算，误差是激活单元的预测（${a^{(4)}}$）与实际值（$y^k$）之间的误差，（$k=1:k$）。 我们用$\delta$来表示误差，则：$\delta^{(4)}=a^{(4)}-y$ 我们利用这个误差值来计算前一层的误差：$\delta^{(3)}=\left({\Theta^{(3)}}\right)^{T}\delta^{(4)}\ast g'\left(z^{(3)}\right)$ 其中 $g'(z^{(3)})$是 $S$ 形函数的导数，$g'(z^{(3)})=a^{(3)}\ast(1-a^{(3)})$。而$(θ^{(3)})^{T}\delta^{(4)}$则是权重导致的误差的和。下一步是继续计算第二层的误差： $ \delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}\ast g'(z^{(2)})$ 因为第一层是输入变量，不存在误差。我们有了所有的误差的表达式后，便可以计算代价函数的偏导数了，假设$λ=0$，即我们不做任何正则化处理时有： $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_{j}^{(l)} \delta_{i}^{l+1}$

重要的是清楚地知道上面式子中上下标的含义：

$l$ 代表目前所计算的是第几层。

$j$ 代表目前计算层中的激活单元的下标，也将是下一层的第$j$个输入变量的下标。

$i$ 代表下一层中误差单元的下标，是受到权重矩阵中第$i$行影响的下一层中的误差单元的下标。

如果我们考虑正则化处理，并且我们的训练集是一个特征矩阵而非向量。在上面的特殊情况中，我们需要计算每一层的误差单元来计算代价函数的偏导数。在更为一般的情况中，我们同样需要计算每一层的误差单元，但是我们需要为整个训练集计算误差单元，此时的误差单元也是一个矩阵，我们用$\Delta^{(l)}_{ij}$来表示这个误差矩阵。第 $l$ 层的第 $i$ 个激活单元受到第 $j$ 个参数影响而导致的误差。

我们的算法表示为：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5514df14ebd508fd597e552fbadcf053.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5514df14ebd508fd597e552fbadcf053.jpg)

即首先用正向传播方法计算出每一层的激活单元，利用训练集的结果与神经网络预测的结果求出最后一层的误差，然后利用该误差运用反向传播法计算出直至第二层的所有误差。

在求出了$\Delta_{ij}^{(l)}$之后，我们便可以计算代价函数的偏导数了，计算方法如下： $ D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}+\lambda\Theta_{ij}^{(l)}$ ${if}; j \neq 0$

$ D_{ij}^{(l)} :=\frac{1}{m}\Delta_{ij}^{(l)}$ ${if}j = 0$

前向传播算法：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5778e97c411b23487881a87cfca781bb.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5778e97c411b23487881a87cfca781bb.png)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/63a0e4aef6d47ba7fa6e07088b61ae68.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/63a0e4aef6d47ba7fa6e07088b61ae68.png)

反向传播算法做的是：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/57aabbf26290e2082a00c5114ae1c5dc.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/57aabbf26290e2082a00c5114ae1c5dc.png)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1542307ad9033e39093e7f28d0c7146c.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/1542307ad9033e39093e7f28d0c7146c.png)

**感悟**：上图中的 $\delta^{(l)}_{j}="error" \ of cost \ for \ a^{(l)}_{j} \ (unit \ j \ in \ layer \ l)$ 理解如下：

$\delta^{(l)}_{j}$ 相当于是第 $l$ 层的第 $j$ 单元中得到的激活项的“误差”，即”正确“的 $a^{(l)}_{j}$ 与计算得到的 $a^{(l)}_{j}$ 的差。

而 $a^{(l)}_{j}=g(z^{(l)})$ ，（g为sigmoid函数）。我们可以想象 $\delta^{(l)}_{j}$ 为函数求导时迈出的那一丁点微分，所以更准确的说 $\delta^{(l)}_{j}=\frac{\partial}{\partial z^{(l)}_{j}}cost(i)$   

## 梯度检验

当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。

为了避免这样的问题，我们采取一种叫做梯度的数值检验（**Numerical Gradient Checking**）方法。这种方法的思想是通过估计梯度值来检验我们计算的导数值是否真的是我们要求的。

对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的平均值用以估计梯度。即对于某个特定的 $\theta$，我们计算出在 $\theta$-$\varepsilon $ 处和 $\theta$+$\varepsilon $ 的代价值（$\varepsilon $是一个非常小的值，通常选取 0.001），然后求两个代价的平均，用以估计在 $\theta$ 处的代价值。

当$\theta$是一个向量时，我们则需要对偏导数进行检验。因为代价函数的偏导数检验只针对一个参数的改变进行检验，下面是一个只针对$\theta_1$进行检验的示例： $$ \frac{\partial}{\partial\theta_1}=\frac{J\left(\theta_1+\varepsilon_1,\theta_2,\theta_3...\theta_n \right)-J \left( \theta_1-\varepsilon_1,\theta_2,\theta_3...\theta_n \right)}{2\varepsilon} $$

最后我们还需要对通过反向传播方法计算出的偏导数进行检验。

根据上面的算法，计算出的偏导数存储在矩阵 $D_{ij}^{(l)}$ 中。检验时，我们要将该矩阵展开成为向量，同时我们也将 $\theta$ 矩阵展开为向量，我们针对每一个 $\theta$ 都计算一个近似的梯度值，将这些值存储于一个近似梯度矩阵中，最终将得出的这个矩阵同 $D_{ij}^{(l)}$ 进行比较。

## 随机初始化

任何优化算法都需要一些初始的参数。到目前为止我们都是初始所有参数为0，这样的初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。如果我们令所有的初始参数都为0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我们初始所有的参数都为一个非0的数，结果也是一样的。



## 总结

网络结构：第一件要做的事是选择网络结构，即决定选择多少层以及决定每层分别有多少个单元。

第一层的单元数即我们训练集的特征数量。

最后一层的单元数是我们训练集的结果的类的数量。

如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。

我们真正要决定的是隐藏层的层数和每个中间层的单元数。

训练神经网络：

1. 参数的随机初始化
2. 利用正向传播方法计算所有的$h_{\theta}(x)$
3. 编写计算代价函数 $J$ 的代码
4. 利用反向传播方法计算所有偏导数
5. 利用数值检验方法检验这些偏导数
6. 使用优化算法来最小化代价函数



# 应用机器学习的建议

当我们运用训练好了的模型来预测未知数据的时候发现有较大的误差，我们下一步可以做什么？

1. 获得更多的训练样本——通常是有效的，但代价较大，下面的方法也可能有效，可考虑先采用下面的几种方法。
2. 尝试减少特征的数量
3. 尝试获得更多的特征
4. 尝试增加多项式特征
5. 尝试减少正则化程度$\lambda$
6. 尝试增加正则化程度$\lambda$ 

这些方法，它们也被称为"机器学习诊断法"。“诊断法”的意思是：这是一种测试法，你通过执行这种测试，能够深入了解某种算法到底是否有用。这通常也能够告诉你，要想改进一种算法的效果，什么样的尝试，才是有意义的。



## 评估假设

当我们确定学习算法的参数的时候，我们考虑的是选择参量来使训练误差最小化，有人认为得到一个非常小的训练误差一定是一件好事，但我们已经知道，仅仅是因为这个假设具有很小的训练误差，并不能说明它就一定是一个好的假设函数。

为了检验算法是否过拟合，我们将数据分成训练集和测试集，通常用70%的数据作为训练集，用剩下30%的数据作为测试集。很重要的一点是训练集和测试集均要含有各种类型的数据，通常我们要对数据进行“洗牌”，然后再分成训练集和测试集。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/9c769fd59c8a9c9f92200f538d1ab29c.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/9c769fd59c8a9c9f92200f538d1ab29c.png)

测试集评估在通过训练集让我们的模型学习得出其参数后，对测试集运用该模型，我们有两种方式计算误差：

1. 对于线性回归模型，我们利用测试集数据计算代价函数$J$
2. 对于逻辑回归模型，我们除了可以利用测试数据集来计算代价函数外：

$$ J_{test}{(\theta)} = -\frac{1}{m_test}\sum\limits_{i=1}^{m_{test}}\log{h_{\theta}(x^{(i)}_{test})}+(1-{y^{(i)}_{test}})\log{h_{\theta}(x^{(i)}_{test})}$$ 

误分类的比率，对于每一个测试集样本，计算：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/751e868bebf4c0bf139db173d25e8ec4.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/751e868bebf4c0bf139db173d25e8ec4.png)

然后对计算结果求平均。



## 模型选择与交叉验证集

假设我们要在10个不同次数的二项式模型之间进行选择：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1b908480ad78ee54ba7129945015f87f.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/1b908480ad78ee54ba7129945015f87f.jpg)

显然越高次数的多项式模型越能够适应我们的训练数据集，但是适应训练数据集并不代表着能推广至一般情况，我们应该选择一个更能适应一般情况的模型。我们需要使用交叉验证集来帮助选择模型。

即：使用60%的数据作为训练集，使用 20%的数据作为交叉验证集，使用20%的数据作为测试集

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/7cf1cd9c123a72ca4137ca515871689d.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/7cf1cd9c123a72ca4137ca515871689d.png)

模型选择的方法为：

1. 使用训练集训练出10个模型

2. 用10个模型分别对交叉验证集计算得出交叉验证误差（代价函数的值）

3. 选取代价函数值最小的模型

4. 用步骤3中选出的模型对测试集计算得出推广误差（代价函数的值）

   ***Train/validation/test error***

   **Training error:**

$J_{train}(\theta) = \frac{1}{2m}\sum_\limits{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$

**Cross Validation error:**

$J_{cv}(\theta) = \frac{1}{2m_{cv}}\sum_\limits{i=1}^{m}(h_{\theta}(x^{(i)}*{cv})-y^{(i)}*{cv})^2$

**Test error:**

$J_{test}(\theta)=\frac{1}{2m_{test}}\sum_\limits{i=1}^{m_{test}}(h_{\theta}(x^{(i)}*{cv})-y^{(i)}*{cv})^2$

## 诊断偏差和方差

当你运行一个学习算法时，如果这个算法的表现不理想，那么多半是出现两种情况：要么是偏差比较大，要么是方差比较大。换句话说，出现的情况要么是欠拟合，要么是过拟合问题。

我们通常会通过将训练集和交叉验证集的代价函数误差与多项式的次数绘制在同一张图表上来帮助分析：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/bca6906add60245bbc24d71e22f8b836.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/bca6906add60245bbc24d71e22f8b836.png)                           $d$ 表示多项式的次数

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/64ad47693447761bd005243ae7db0cca.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/64ad47693447761bd005243ae7db0cca.png) 

对于训练集，当 $d$ 较小时，模型拟合程度更低，误差较大；随着 $d$ 的增长，拟合程度提高，误差减小。

对于交叉验证集，当 $d$ 较小时，模型拟合程度低，误差较大；但是随着 $d$ 的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

如果我们的交叉验证集误差较大，我们如何判断是方差还是偏差呢？根据上面的图表，我们知道:

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/25597f0f88208a7e74a3ca028e971852.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/25597f0f88208a7e74a3ca028e971852.png)

训练集误差和交叉验证集误差近似时：偏差/欠拟合         (偏差是距离目标较远)

交叉验证集误差远大于训练集误差时：方差/过拟合	     (方差是较分散，训练集数量小于参数数量则可能过拟合)

## 正则化和偏差方差

在我们在训练模型的过程中，一般会使用一些正则化方法来防止过拟合。但是我们可能会正则化的程度太高或太小了，即我们在选择λ的值时也需要思考与刚才选择多项式模型次数类似的问题。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2ba317c326547f5b5313489a3f0d66ce.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/2ba317c326547f5b5313489a3f0d66ce.png)

我们选择一系列的想要测试的 $\lambda$ 值，通常是 0-10之间的呈现2倍关系的值（如：$0,0.01,0.02,0.04,0.08,0.15,0.32,0.64,1.28,2.56,5.12,10$共12个）。 我们同样把数据分为训练集、交叉验证集和测试集。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8f557105250853e1602a78c99b2ef95b.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/8f557105250853e1602a78c99b2ef95b.png)

选择$\lambda$的方法为：

1. 使用训练集训练出12个不同程度正则化的模型
2. 用12个模型分别对交叉验证集计算的出交叉验证误差
3. 选择得出交叉验证误差**最小**的模型
4. 运用步骤3中选出模型对测试集计算得出推广误差，我们也可以同时将训练集和交叉验证集模型的代价函数误差与λ的值绘制在一张图表上：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/38eed7de718f44f6bb23727c5a88bf5d.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/38eed7de718f44f6bb23727c5a88bf5d.png)

• 当 $\lambda$ 较小时，训练集误差较小（过拟合）而交叉验证集误差较大

• 随着 $\lambda$ 的增加，训练集误差不断增加（欠拟合），而交叉验证集误差则是先减小后增加



## 学习曲线

学习曲线就是一种很好的工具，我经常使用学习曲线来判断某一个学习算法是否处于偏差、方差问题。学习曲线是学习算法的一个很好的**合理检验**（**sanity check**）。学习曲线是将训练集误差和交叉验证集误差作为训练集样本数量（$m$）的函数绘制的图表。

即，如果我们有100行数据，我们从1行数据开始，逐渐学习更多行的数据。思想是：当训练较少行数据的时候，训练的模型将能够非常完美地适应较少的训练数据，但是训练出来的模型却不能很好地适应交叉验证集数据或测试集数据。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/969281bc9b07e92a0052b17288fb2c52.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/969281bc9b07e92a0052b17288fb2c52.png)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/973216c7b01c910cfa1454da936391c6.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/973216c7b01c910cfa1454da936391c6.png)

如何利用学习曲线识别高偏差/欠拟合：作为例子，我们尝试用一条直线来适应下面的数据，可以看出，无论训练集有多么大误差都不会有太大改观：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/4a5099b9f4b6aac5785cb0ad05289335.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/4a5099b9f4b6aac5785cb0ad05289335.jpg)

也就是说在高偏差/欠拟合的情况下，增加数据到训练集不一定能有帮助。

如何利用学习曲线识别高方差/过拟合：假设我们使用一个非常高次的多项式模型，并且正则化非常小，可以看出，当交叉验证集误差远大于训练集误差时，往训练集增加更多数据可以提高模型的效果。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2977243994d8d28d5ff300680988ec34.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/2977243994d8d28d5ff300680988ec34.jpg)

也就是说在高方差/过拟合的情况下，增加更多数据到训练集可能可以提高算法效果。

## 解决方法

1. 获得更多的训练样本——解决高方差
2. 尝试减少特征的数量——解决高方差
3. 尝试获得更多的特征——解决高偏差
4. 尝试增加多项式特征——解决高偏差
5. 尝试减少正则化程度λ——解决高偏差
6. 尝试增加正则化程度λ——解决高方差

神经网络的方差和偏差： [![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c5cd6fa2eb9aea9c581b2d78f2f4ea57.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/c5cd6fa2eb9aea9c581b2d78f2f4ea57.png)

使用较小的神经网络，类似于参数较少的情况，容易导致高偏差和欠拟合，但计算代价较小使用较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算代价比较大，但是可以通过正则化手段来调整而更加适应数据。

**通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好。**

对于神经网络中的隐藏层的层数的选择，通常从一层开始逐渐增加层数，为了更好地作选择，可以把数据分为训练集、交叉验证集和测试集，针对不同隐藏层层数的神经网络训练神经网络， 然后选择交叉验证集代价最小的神经网络。

## 误差分析

每当我研究机器学习的问题时，我最多只会花一天的时间，就是字面意义上的24小时，来试图很快的把结果搞出来，即便效果不好。坦白的说，就是根本没有用复杂的系统，但是只是很快的得到的结果。即便运行得不完美，但是也把它运行一遍，最后通过交叉验证来检验数据。一旦做完，你可以画出学习曲线，通过画出学习曲线，以及检验误差，来找出你的算法是否有高偏差和高方差的问题，或者别的问题。在这样分析之后，再来决定用更多的数据训练，或者加入更多的特征变量是否有用。

这么做的原因是：这在你刚接触机器学习问题时是一个很好的方法，你并不能提前知道你是否需要复杂的特征变量，或者你是否需要更多的数据，还是别的什么。提前知道你应该做什么，是非常难的，因为你缺少证据，缺少学习曲线。因此，你很难知道你应该把时间花在什么地方来提高算法的表现。但是当你实践一个非常简单即便不完美的方法时，你可以通过画出学习曲线来做出进一步的选择。你可以用这种方式来避免一种电脑编程里的过早优化问题，这种理念是：我们必须用证据来领导我们的决策，怎样分配自己的时间来优化算法，而不是仅仅凭直觉，凭直觉得出的东西一般总是错误的。

除了画出学习曲线之外，一件非常有用的事是误差分析，我的意思是说：当我们在构造垃圾邮件分类器时，我会看一看我的交叉验证数据集，然后亲自看一看哪些邮件被算法错误地分类。因此，通过这些被算法错误分类的垃圾邮件与非垃圾邮件，你可以发现某些系统性的规律：什么类型的邮件总是被错误分类。经常地这样做之后，这个过程能启发你构造新的特征变量，或者告诉你：现在这个系统的短处，然后启发你如何去提高它。

误差分析并不总能帮助我们判断应该采取怎样的行动。有时我们需要尝试不同的模型，然后进行比较，在模型比较时，用数值来判断哪一个模型更好更有效，通常我们是看**交叉验证集**的误差。

## 类偏斜的误差度量

类偏斜情况表现为我们的训练集中有非常多的同一种类的样本，只有很少或没有其他类的样本。

例如我们希望用算法来预测癌症是否是恶性的，在我们的训练集中，只有0.5%的实例是恶性肿瘤。假设我们编写一个非学习而来的算法，在所有情况下都预测肿瘤是良性的，那么误差只有0.5%。然而我们通过训练而得到的神经网络算法却有1%的误差。这时，误差的大小是不能视为评判算法效果的依据的。

**查准率**（**Precision**）和**查全率**（**Recall**） 我们将算法预测的结果分成四种情况：

1. **正确肯定**（**True Positive,TP**）：预测为真，实际为真

2.**正确否定**（**True Negative,TN**）：预测为假，实际为假

3.**错误肯定**（**False Positive,FP**）：预测为真，实际为假

4.**错误否定**（**False Negative,FN**）：预测为假，实际为真

则：查准率=**TP/(TP+FP)**。例，在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿瘤的病人的百分比，越高越好。

查全率=**TP/(TP+FN)**。例，在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤的病人的百分比，越高越好。

这样，对于我们刚才那个总是预测病人肿瘤为良性的算法，其查全率是0。

|            |              | **预测值**   |             |
| ---------- | ------------ | ------------ | ----------- |
|            |              | **Positive** | **Negtive** |
| **实际值** | **Positive** | **TP**       | **FN**      |
|            | **Negtive**  | **FP**       | **TN**      |



## 查全率和查准率之间的权衡

假使，我们的算法输出的结果在0-1 之间，我们使用阀值0.5 来预测真和假。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ad00c2043ab31f32deb2a1eb456b7246.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ad00c2043ab31f32deb2a1eb456b7246.png)

查准率**(Precision)=TP/(TP+FP)** 例，在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿瘤的病人的百分比，越高越好。

查全率**(Recall)=TP/(TP+FN)**例，在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤的病人的百分比，越高越好。

如果我们希望只在非常确信的情况下预测为真（肿瘤为恶性），即我们希望更高的查准率，我们可以使用比0.5更大的阀值，如0.7，0.9。这样做我们会减少错误预测病人为恶性肿瘤的情况，同时却会增加未能成功预测肿瘤为恶性的情况。

如果我们希望提高查全率，尽可能地让所有有可能是恶性肿瘤的病人都得到进一步地检查、诊断，我们可以使用比0.5更小的阀值，如0.3。

我们可以将不同阀值情况下，查全率与查准率的关系绘制成图表，曲线的形状根据数据的不同而不同：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/84067e23f2ab0423679379afc6ed6caf.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/84067e23f2ab0423679379afc6ed6caf.png)

我们希望有一个帮助我们选择这个阀值的方法。一种方法是计算**F1 值**（**F1 Score**），其计算公式为：

${{F}_{1}}Score:2\frac{PR}{P+R}$

我们选择使得**F1**值最高的阀值。

# 支持向量机

构建支持向量机。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/59541ab1fda4f92d6f1b508c8e29ab1c.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/59541ab1fda4f92d6f1b508c8e29ab1c.png)

这是我们在逻辑回归中使用代价函数$J(\theta)$。也许这个方程看起来不是非常熟悉。这是因为之前有个负号在方程外面，但是，这里我所做的是，将负号移到了表达式的里面，这样做使得方程看起来有些不同。对于支持向量机而言，实质上我们要将这替换为${\cos}t_1{(z)}$，也就是${\cos}t_1{(\theta^Tx)}$，同样地，我也将这一项替换为${\cos}t_0{(z)}$，也就是代价${\cos}t_0{(\theta^Tx)}$。这里的代价函数${\cos}t_1$，就是之前所提到的那条线。此外，代价函数${\cos}t_0$，也是上面所介绍过的那条线。因此，对于支持向量机，我们得到了这里的最小化问题，即:

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/4ac1ca54cb0f2c465ab81339baaf9186.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/4ac1ca54cb0f2c465ab81339baaf9186.png)

然后，再加上正则化参数。现在，按照支持向量机的惯例，事实上，我们的书写会稍微有些不同，代价函数的参数表示也会稍微有些不同。

首先，我们要除去$1/m$这一项，当然，这仅仅是由于人们使用支持向量机时，对比于逻辑回归而言，不同的习惯所致，但这里我所说的意思是：你知道，我将要做的是仅仅除去$1/m$这一项，但是，这也会得出同样的 ${{\theta }}$ 最优值，好的，因为$1/m$ 仅是个常量，因此，你知道在这个最小化问题中，无论前面是否有$1/m$ 这一项，最终我所得到的最优值${{\theta }}$都是一样的。这里我的意思是，先给你举一个样本，假定有一最小化问题：即要求当$(u-5)^2+1$取得最小值时的$u$值，这时最小值为：当$u=5$时取得最小值。

现在，如果我们想要将这个目标函数乘上常数10，这里我的最小化问题就变成了：求使得$10×(u-5)^2+10$最小的值$u$，然而，使得这里最小的$u$值仍为5。因此将一些常数乘以你的最小化项，这并不会改变最小化该方程时得到$u$值。因此，这里我所做的是删去常量$m$。也相同的，我将目标函数乘上一个常量$m$，并不会改变取得最小值时的${{\theta }}$值。

第二点概念上的变化，我们只是指在使用支持向量机时，一些如下的标准惯例，而不是逻辑回归。因此，对于逻辑回归，在目标函数中，我们有两项：第一个是训练样本的代价，第二个是我们的正则化项，我们不得不去用这一项来平衡。这就相当于我们想要最小化$A$加上正则化参数$\lambda$，然后乘以其他项$B$对吧？这里的$A$表示这里的第一项，同时我用**B**表示第二项，但不包括$\lambda$，我们不是优化这里的$A+\lambda\times B$。我们所做的是通过设置不同正则参数$\lambda$达到优化目的。这样，我们就能够权衡对应的项，是使得训练样本拟合的更好。即最小化$A$。还是保证正则参数足够小，也即是对于**B**项而言，但对于支持向量机，按照惯例，我们将使用一个不同的参数替换这里使用的$\lambda$来权衡这两项。你知道，就是第一项和第二项我们依照惯例使用一个不同的参数称为$C$，同时改为优化目标，$C×A+B$。 因此，在逻辑回归中，如果给定$\lambda$，一个非常大的值，意味着给予$B$更大的权重。而这里，就对应于将$C$ 设定为非常小的值，那么，相应的将会给$B$比给$A$更大的权重。因此，这只是一种不同的方式来控制这种权衡或者一种不同的方法，即用参数来决定是更关心第一项的优化，还是更关心第二项的优化。当然你也可以把这里的参数$C$ 考虑成$1/\lambda$，同 $1/\lambda$所扮演的角色相同，并且这两个方程或这两个表达式并不相同，因为$C=1/\lambda$，但是也并不全是这样，如果当$C=1/\lambda$时，这两个优化目标应当得到相同的值，相同的最优值 ${{\theta }}$。因此，就用它们来代替。那么，我现在删掉这里的$\lambda$，并且用常数$C$来代替。因此，这就得到了在支持向量机中我们的整个优化目标函数。然后最小化这个目标函数，得到**SVM** 学习到的参数$C$。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5a63e35db410fdb57c76de97ea888278.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/5a63e35db410fdb57c76de97ea888278.png)

最后有别于逻辑回归输出的概率。在这里，我们的代价函数，当最小化代价函数，获得参数${{\theta }}$时，支持向量机所做的是它来直接预测$y$的值等于1，还是等于0。因此，这个假设函数会预测1。当$\theta^Tx$大于或者等于0时，或者等于0时，所以学习参数${{\theta }}$就是支持向量机假设函数的形式。那么，这就是支持向量机数学上的定义。

## 大边界的直观理解

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/cc66af7cbd88183efc07c8ddf09cbc73.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/cc66af7cbd88183efc07c8ddf09cbc73.png)

这是我的支持向量机模型的代价函数，在左边这里我画出了关于$z$的代价函数${\cos}t_1{(z)}$，此函数用于正样本，而在右边这里我画出了关于$z$的代价函数${\cos}t_0{(z)}$，横轴表示$z$，现在让我们考虑一下，最小化这些代价函数的必要条件是什么。如果你有一个正样本，$y=1$，则只有在$z>=1$时，代价函数${\cos}t_1{(z)}$才等于0。

事实上，支持向量机现在要比这个大间距分类器所体现得更成熟，尤其是当你使用大间距分类器的时候，你的学习算法会受异常点(outlier) 的影响。比如我们加入一个额外的正样本。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b8fbe2f6ac48897cf40497a2d034c691.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/b8fbe2f6ac48897cf40497a2d034c691.png)

在这里，如果你加了这个样本，为了将样本用最大间距分开，也许我最终会得到一条类似这样的决策界，对么？就是这条粉色的线，仅仅基于一个异常值，仅仅基于一个样本，就将我的决策界从这条黑线变到这条粉线，这实在是不明智的。而如果正则化参数$C$，设置的非常大，这事实上正是支持向量机将会做的。它将决策界，从黑线变到了粉线，但是如果$C$ 设置的小一点，**如果你将C设置的不要太大，则你最终会得到这条黑线，**当然数据如果不是线性可分的，如果你在这里有一些正样本或者你在这里有一些负样本，则支持向量机也会将它们恰当分开。因此，大间距分类器的描述，仅仅是从直观上给出了正则化参数$C$非常大的情形，同时，要提醒你$C$的作用类似于$1/\lambda$，$\lambda$是我们之前使用过的正则化参数。这只是$C$非常大的情形，或者等价地 $\lambda$ 非常小的情形。你最终会得到类似粉线这样的决策界，但是实际上应用支持向量机的时候，**当$C$不是非常非常大的时候，它可以忽略掉一些异常点的影响，得到更好的决策界。**甚至当你的数据不是线性可分的时候，支持向量机也可以给出好的结果。

回顾 $C=1/\lambda$，因此：

$C$ 较大时，相当于 $\lambda$ 较小，可能会导致过拟合，高方差。

$C$ 较小时，相当于$\lambda$较大，可能会导致低拟合，高偏差。



## 核函数

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/529b6dbc07c9f39f5266bd0b3f628545.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/529b6dbc07c9f39f5266bd0b3f628545.png)

为了获得上图所示的判定边界，我们的模型可能是${{\theta }*{0}}+{{\theta }*{1}}{{x}*{1}}+{{\theta }*{2}}{{x}*{2}}+{{\theta }*{3}}{{x}*{1}}{{x}*{2}}+{{\theta }*{4}}x*{1}^{2}+{{\theta }*{5}}x*{2}^{2}+\cdots $的形式。

我们可以用一系列的新的特征$f$来替换模型中的每一项。例如令： ${{f}*{1}}={{x}*{1}},{{f}*{2}}={{x}*{2}},{{f}*{3}}={{x}*{1}}{{x}*{2}},{{f}*{4}}=x_{1}^{2},{{f}*{5}}=x*{2}^{2}$

...得到$h_θ(x)={{\theta }*{1}}f_1+{{\theta }*{2}}f_2+...+{{\theta }_{n}}f_n$。然而，除了对原有的特征进行组合以外，有没有更好的方法来构造$f_1,f_2,f_3$？我们可以利用核函数来计算出新的特征。

给定一个训练样本$x$，我们利用$x$的各个特征与我们预先选定的**地标**(**landmarks**)$l^{(1)},l^{(2)},l^{(3)}$的近似程度来选取新的特征$f_1,f_2,f_3$。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2516821097bda5dfaf0b94e55de851e0.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/2516821097bda5dfaf0b94e55de851e0.png)

例如：${{f}_{1}}=similarity(x,{{l}^{(1)}})=e(-\frac{{{\left| x-{{l}^{(1)}} \right|}^{2}}}{2{{\sigma }^{2}}})$

其中：${{\left| x-{{l}^{(1)}} \right|}^{2}}=\sum{*{j=1}^{n}}{{({{x}*{j}}-l_{j}^{(1)})}^{2}}$，为实例$x$中所有特征与地标$l^{(1)}$之间的距离的和。上例中的$similarity(x,{{l}^{(1)}})$就是核函数，具体而言，这里是一个**高斯核函数**(**Gaussian Kernel**)。 **注：这个函数与正态分布没什么实际上的关系，只是看上去像而已。**

这些地标的作用是什么？如果一个训练样本$x$与地标$l$之间的距离近似于0，则新特征 $f$近似于$e^{-0}=1$，如果训练样本$x$与地标$l$之间距离较远，则$f$近似于$e^{-(一个较大的数)}=0$。

假设我们的训练样本含有两个特征[$x_{1}$ $x{_2}$]，给定地标$l^{(1)}$与不同的$\sigma$值，见下图：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b9acfc507a54f5ca13a3d50379972535.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/b9acfc507a54f5ca13a3d50379972535.jpg)

图中水平面的坐标为 $x_{1}$，$x_{2}$而垂直坐标轴代表$f$。可以看出，只有当$x$与$l^{(1)}$重合时$f$才具有最大值。随着$x$的改变$f$值改变的速率受到$\sigma^2$的控制。

在下图中，当样本处于洋红色的点位置处，因为其离$l^{(1)}$更近，但是离$l^{(2)}$和$l^{(3)}$较远，因此$f_1$接近1，而$f_2$,$f_3$接近0。因此$h_θ(x)=θ_0+θ_1f_1+θ_2f_2+θ_1f_3>0$，因此预测$y=1$。同理可以求出，对于离$l^{(2)}$较近的绿色点，也预测$y=1$，但是对于蓝绿色的点，因为其离三个地标都较远，预测$y=0$。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/3d8959d0d12fe9914dc827d5a074b564.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/3d8959d0d12fe9914dc827d5a074b564.jpg)

这样，图中红色的封闭曲线所表示的范围，便是我们依据一个单一的训练样本和我们选取的地标所得出的判定边界，在预测时，我们采用的特征不是训练样本本身的特征，而是通过核函数计算出的新特征$f_1,f_2,f_3$。

如何选择地标？

我们通常是根据训练集的数量选择地标的数量，即如果训练集中有$m$个样本，则我们选取$m$个地标，并且令:$l^{(1)}=x^{(1)},l^{(2)}=x^{(2)},.....,l^{(m)}=x^{(m)}$。这样做的好处在于：现在我们得到的新特征是建立在原有特征与训练集中所有其他特征之间距离的基础之上的，即：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/eca2571849cc36748c26c68708a7a5bd.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/eca2571849cc36748c26c68708a7a5bd.png)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ea31af620b0a0132fe494ebb4a362465.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ea31af620b0a0132fe494ebb4a362465.png)

下面我们将核函数运用到支持向量机中，修改我们的支持向量机假设为：

• 给定$x$，计算新特征$f$，当$θ^Tf>=0$ 时，预测 $y=1$，否则反之。

相应地修改代价函数为：$\sum{_{j=1}^{n=m}}\theta _{j}^{2}={{\theta}^{T}}\theta $，

$min C\sum\limits_{i=1}^{m}{[{{y}^{(i)}}cos {{t}*{1}}}( {{\theta }^{T}}{{f}^{(i)}})+(1-{{y}^{(i)}})cos {{t}*{0}}( {{\theta }^{T}}{{f}^{(i)}})]+\frac{1}{2}\sum\limits_{j=1}^{n=m}{\theta *{j}^{2}}$ 在具体实施过程中，我们还需要对最后的正则化项进行些微调整，在计算$\sum{*{j=1}^{n=m}}\theta _{j}^{2}={{\theta}^{T}}\theta $时，我们用$θ^TMθ$代替$θ^Tθ$，其中$M$是根据我们选择的核函数而不同的一个矩阵。这样做的原因是为了简化计算。

理论上讲，我们也可以在逻辑回归中使用核函数，但是上面使用 $M$来简化计算的方法不适用与逻辑回归，因此计算将非常耗费时间。

在此，我们不介绍最小化支持向量机的代价函数的方法，你可以使用现有的软件包（如**liblinear**,**libsvm**等）。在使用这些软件包最小化我们的代价函数之前，我们通常需要编写核函数，并且如果我们使用高斯核函数，那么在使用之前进行特征缩放是非常必要的。

另外，支持向量机也可以不使用核函数，不使用核函数又称为**线性核函数**(**linear kernel**)，当我们不采用非常复杂的函数，或者我们的训练集特征非常多而样本非常少的时候，可以采用这种不带核函数的支持向量机。

下面是支持向量机的两个参数$C$和$\sigma$的影响：

$C=1/\lambda$

$C$ 较大时，相当于$\lambda$较小，可能会导致过拟合，高方差；

$C$ 较小时，相当于$\lambda$较大，可能会导致低拟合，高偏差；

$\sigma$较大时，可能会导致低方差，高偏差；

$\sigma$较小时，可能会导致低偏差，高方差。

# 聚类

## 无监督学习

在非监督学习中，我们需要将一系列无标签的训练数据，输入到一个算法中，然后我们告诉这个算法，快去为我们找找这个数据的内在结构给定数据。我们可能需要某种算法帮助我们寻找一种结构。图上的数据看起来可以分成两个分开的点集（称为簇），一个能够找到我圈出的这些点集的算法，就被称为聚类算法。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/6709f5ca3cd2240d4e95dcc3d3e808d5.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/6709f5ca3cd2240d4e95dcc3d3e808d5.png)

这将是我们介绍的第一个非监督学习算法。当然，此后我们还将提到其他类型的非监督学习算法，它们可以为我们找到其他类型的结构或者其他的一些模式，而不只是簇。

## K-均值算法

**K-均值**是最普及的聚类算法，算法接受一个未标记的数据集，然后将数据聚类成不同的组。

**K-均值**是一个迭代算法，假设我们想要将数据聚类成n个组，其方法为:

首先选择$K$个随机的点，称为**聚类中心**（**cluster centroids**）；

对于数据集中的每一个数据，按照距离$K$个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。

计算每一个组的平均值，将该组所关联的中心点移动到平均值的位置。

重复步骤2-4直至中心点不再变化。

用$μ^1$,$μ^2$,...,$μ^k$ 来表示聚类中心，用$c^{(1)}$,$c^{(2)}$,...,$c^{(m)}$来存储与第$i$个实例数据最近的聚类中心的索引，**K-均值**算法的伪代码如下：

```
Repeat {

for i = 1 to m

c(i) := index (form 1 to K) of cluster centroid closest to x(i)

for k = 1 to K

μk := average (mean) of points assigned to cluster k

}
```

算法分为两个步骤，第一个**for**循环是赋值步骤，即：对于每一个样例$i$，计算其应该属于的类。第二个**for**循环是聚类中心的移动，即：对于每一个类$K$，重新计算该类的质心。

**K-均值**算法也可以很便利地用于将数据分为许多不同组，即使在没有非常明显区分的组群的情况下也可以。下图所示的数据集包含身高和体重两项特征构成的，利用**K-均值**算法将数据分为三类，用于帮助确定将要生产的T-恤衫的三种尺寸。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fed50a4e482cf3aae38afeb368141a97.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/fed50a4e482cf3aae38afeb368141a97.png)

## 优化目标

K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称**畸变函数** **Distortion function**）为：

$$J(c^{(1)},...,c^{(m)},μ_1,...,μ_K)=\dfrac {1}{m}\sum^{m}*{i=1}\left| X^{\left( i\right) }-\mu*{c^{(i)}}\right| ^{2}$$

其中${{\mu }_{{{c}^{(i)}}}}$代表与${{x}^{(i)}}$最近的聚类中心点。 我们的的优化目标便是找出使得代价函数最小的 $c^{(1)}$,$c^{(2)}$,...,$c^{(m)}$和$μ^1$,$μ^2$,...,$μ^k$： [![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8605f0826623078a156d30a7782dfc3c.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/8605f0826623078a156d30a7782dfc3c.png)

回顾刚才给出的: **K-均值**迭代算法，我们知道，第一个循环是用于减小$c^{(i)}$引起的代价，而第二个循环则是用于减小${{\mu }_{i}}$引起的代价。迭代的过程一定会是每一次迭代都在减小代价函数，不然便是出现了错误。

## 随机初始化

在运行K-均值算法的之前，我们首先要随机初始化所有的聚类中心点，下面介绍怎样做：

1. 我们应该选择$K<m$，即聚类中心点的个数要小于所有训练集实例的数量
2. 随机选择$K$个训练实例，然后令$K$个聚类中心分别与这$K$个训练实例相等

**K-均值**的一个问题在于，它有可能会停留在一个局部最小值处，而这取决于初始化的情况。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/d4d2c3edbdd8915f4e9d254d2a47d9c7.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/d4d2c3edbdd8915f4e9d254d2a47d9c7.png)

为了解决这个问题，我们通常需要多次运行**K-均值**算法，每一次都重新进行随机初始化，最后再比较多次运行**K-均值**的结果，选择代价函数最小的结果。这种方法在$K$较小的时候（2--10）还是可行的，但是如果$K$较大，这么做也可能不会有明显地改善。

## 聚类数的选择

当人们在讨论，选择聚类数目的方法时，有一个可能会谈及的方法叫作“肘部法则”。关于“肘部法则”，我们所需要做的是改变$K$值，也就是聚类类别数目的总数。我们用一个聚类来运行**K均值**聚类方法。这就意味着，所有的数据都会分到一个聚类里，然后计算成本函数或者计算畸变函数$J$。$K$代表聚类数字。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f3ddc6d751cab7aba7a6f8f44794e975.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/f3ddc6d751cab7aba7a6f8f44794e975.png)

我们可能会得到一条类似于这样的曲线。像一个人的肘部。这就是“肘部法则”所做的，让我们来看这样一个图，看起来就好像有一个很清楚的肘在那儿。好像人的手臂，如果你伸出你的胳膊，那么这就是你的肩关节、肘关节、手。这就是“肘部法则”。你会发现这种模式，它的畸变值会迅速下降，从1到2，从2到3之后，你会在3的时候达到一个肘点。在此之后，畸变值就下降的非常慢，看起来就像使用3个聚类来进行聚类是正确的，这是因为那个点是曲线的肘点，畸变值下降得很快，$K=3$之后就下降得很慢，那么我们就选$K=3$。当你应用“肘部法则”的时候，如果你得到了一个像上面这样的图，那么这将是一种用来选择聚类个数的合理方法。

例如，我们的 T-恤制造例子中，我们要将用户按照身材聚类，我们可以分成3个尺寸:$S,M,L$，也可以分成5个尺寸$XS,S,M,L,XL$，这样的选择是建立在回答“聚类后我们制造的T-恤是否能较好地适合我们的客户”这个问题的基础上作出的。

**参考**

1.相似度/距离计算方法总结

(1). 闵可夫斯基距离**Minkowski**/（其中欧式距离：$p=2$)

$dist(X,Y)={{\left( {{\sum\limits_{i=1}^{n}{\left| {{x}_{i}}-{{y}_{i}} \right|}}^{p}} \right)}^{\frac{1}{p}}}$

(2). 杰卡德相似系数(**Jaccard**)：

$J(A,B)=\frac{\left| A\cap B \right|}{\left|A\cup B \right|}$

(3). 余弦相似度(**cosine similarity**)：

$n$维向量$x$和$y$的夹角记做$\theta$，根据余弦定理，其余弦值为：

$cos (\theta )=\frac{{{x}^{T}}y}{\left|x \right|\cdot \left| y \right|}=\frac{\sum\limits_{i=1}^{n}{{{x}_{i}}{{y}_{i}}}}{\sqrt{\sum\limits_{i=1}^{n}{{{x}_{i}}^{2}}}\sqrt{\sum\limits_{i=1}^{n}{{{y}_{i}}^{2}}}}$ (4). Pearson皮尔逊相关系数： ${{\rho }*{XY}}=\frac{\operatorname{cov}(X,Y)}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{E[(X-{{\mu }_{X}})(Y-{{\mu }_{Y}})]}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{\sum\limits_{i=1}^{n}{(x-{{\mu }*{X}})(y-{{\mu }_{Y}})}}{\sqrt{\sum\limits_{i=1}^{n}{{{(x-{{\mu }_{X}})}^{2}}}}\sqrt{\sum\limits_{i=1}^{n}{{{(y-{{\mu }_{Y}})}^{2}}}}}$

Pearson相关系数即将$x$、$y$坐标向量各自平移到原点后的夹角余弦。

2.聚类的衡量指标

(1). 均一性：$p$

类似于精确率，一个簇中只包含一个类别的样本，则满足均一性。其实也可以认为就是正确率(每个 聚簇中正确分类的样本数占该聚簇总样本数的比例和)

(2). 完整性：$r$

类似于召回率，同类别样本被归类到相同簇中，则满足完整性;每个聚簇中正确分类的样本数占该 类型的总样本数比例的和

(3). **V-measure**:

均一性和完整性的加权平均

$V = \frac{(1+\beta^2)*pr}{\beta^2*p+r}$

(4). 轮廓系数

样本$i$的轮廓系数：$s(i)$

簇内不相似度:计算样本$i$到同簇其它样本的平均距离为$a(i)$，应尽可能小。

簇间不相似度:计算样本$i$到其它簇$C_j$的所有样本的平均距离$b_{ij}$，应尽可能大。

轮廓系数：$s(i)$值越接近1表示样本$i$聚类越合理，越接近-1，表示样本$i$应该分类到 另外的簇中，近似为0，表示样本$i$应该在边界上;所有样本的$s(i)$的均值被成为聚类结果的轮廓系数。

$s(i) = \frac{b(i)-a(i)}{max{a(i),b(i)}}$

(5). **ARI**

数据集$S$共有$N$个元素， 两个聚类结果分别是：

$X={{{X}*{1}},{{X}*{2}},...,{{X}*{r}}},Y={{{Y}*{1}},{{Y}*{2}},...,{{Y}*{s}}}$

$X$和$Y$的元素个数为：

$a={{{a}*{1}},{{a}*{2}},...,{{a}*{r}}},b={{{b}*{1}},{{b}*{2}},...,{{b}*{s}}}$

[![ri1](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/Ari11.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/Ari11.png)

记：${{n}*{ij}}=\left| {{X}*{i}}\cap {{Y}_{i}} \right|$

$ARI=\frac{\sum\limits_{i,j}{C_{{{n}_{ij}}}^{2}}-\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)\cdot \left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]/C_{n}^{2}}{\frac{1}{2}\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)+\left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]-\left[ \left( \sum\limits_{i}{C_{{{a}_{i}}}^{2}} \right)\cdot \left( \sum\limits_{i}{C_{{{b}_{i}}}^{2}} \right) \right]/C_{n}^{2}}$

# 降维

如果你想测量——如果你想做，你知道，做一个调查或做这些不同飞行员的测试——你可能有一个特征：$x_1$，这也许是他们的技能（直升机飞行员），也许$x_2$可能是飞行员的爱好。这是表示他们是否喜欢飞行，也许这两个特征将高度相关。你真正关心的可能是这条红线的方向，不同的特征，决定飞行员的能力。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8274f0c29314742e9b4f15071ea7624a.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/8274f0c29314742e9b4f15071ea7624a.png)

将数据从三维降至二维： 这个例子中我们要将一个三维的特征向量降至一个二维的特征向量。过程是与上面类似的，我们将三维向量投射到一个二维的平面上，强迫使得所有的数据都在同一个平面上，降至二维的特征向量。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/67e2a9d760300d33ac5e12ad2bd5523c.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/67e2a9d760300d33ac5e12ad2bd5523c.jpg)

这样的处理过程可以被用于把任何维度的数据降到任何想要的维度，例如将1000维的特征降至100维。

## 主成分分析

主成分分析(**PCA**)是最常见的降维算法。

在**PCA**中，我们要做的是找到一个方向向量（**Vector direction**），当我们把所有的数据都投射到该向量上时，我们希望投射平均均方误差能尽可能地小。方向向量是一个经过原点的向量，而投射误差是从特征向量向该方向向量作垂线的长度。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/a93213474b35ce393320428996aeecd9.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/a93213474b35ce393320428996aeecd9.jpg)

下面给出主成分分析问题的描述：

问题是要将$n$维数据降至$k$维，目标是找到向量$u^{(1)}$,$u^{(2)}$,...,$u^{(k)}$使得总的投射误差最小。主成分分析与线性回顾的比较：

主成分分析与线性回归是两种不同的算法。主成分分析最小化的是投射误差（**Projected Error**），而线性回归尝试的是最小化预测误差。线性回归的目的是预测结果，而主成分分析不作任何预测。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/7e1389918ab9358d1432d20ed20f8142.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/7e1389918ab9358d1432d20ed20f8142.png)

上图中，左边的是线性回归的误差（垂直于横轴投影），右边则是主要成分分析的误差（垂直于红线投影）。

**PCA**将$n$个特征降维到$k$个，可以用来进行数据压缩，如果100维的向量最后可以用10维来表示，那么压缩率为90%。同样图像处理领域的**KL变换**使用**PCA**做图像压缩。但**PCA** 要保证降维后，还要保证数据的特性损失最小。

**PCA**技术的一大好处是对数据进行降维的处理。我们可以对新求出的“主元”向量的重要性进行排序，根据需要取前面最重要的部分，将后面的维数省去，可以达到降维从而简化模型或是对数据进行压缩的效果。同时最大程度的保持了原有数据的信息。

**PCA**技术的一个很大的优点是，它是完全无参数限制的。在**PCA**的计算过程中完全不需要人为的设定参数或是根据任何经验模型对计算进行干预，最后的结果只与数据相关，与用户是独立的。

但是，这一点同时也可以看作是缺点。如果用户对观测对象有一定的先验知识，掌握了数据的一些特征，却无法通过参数化等方法对处理过程进行干预，可能会得不到预期的效果，效率也不高。

**算法**

**PCA** 减少$n$维到$k$维：

第一步是均值归一化。我们需要计算出所有特征的均值，然后令 $x_j= x_j-μ_j$。如果特征是在不同的数量级上，我们还需要将其除以标准差 $σ^2$。

第二步是计算**协方差矩阵**（**covariance matrix**）$Σ$： $\sum=\dfrac {1}{m}\sum^{n}_{i=1}\left( x^{(i)}\right) \left( x^{(i)}\right) ^{T}$

第三步是计算协方差矩阵$Σ$的**特征向量**（**eigenvectors**）:

在 **Octave** 里我们可以利用**奇异值分解**（**singular value decomposition**）来求解，`[U, S, V]= svd(sigma)`。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0918b38594709705723ed34bb74928ba.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/0918b38594709705723ed34bb74928ba.png) $$Sigma=\dfrac {1}{m}\sum^{n}_{i=1}\left( x^{(i)}\right) \left( x^{(i)}\right) ^{T}$$

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/01e1c4a2f29a626b5980a27fc7d6a693.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/01e1c4a2f29a626b5980a27fc7d6a693.png)

对于一个 $n×n$维度的矩阵，上式中的$U$是一个具有与数据之间最小投射误差的方向向量构成的矩阵。如果我们希望将数据从$n$维降至$k$维，我们只需要从$U$中选取前$k$个向量，获得一个$n×k$维度的矩阵，我们用$U_{reduce}$表示，然后通过如下计算获得要求的新特征向量$z^{(i)}$: $$z^{(i)}=U^{T}_{reduce}*x^{(i)}$$

其中$x$是$n×1$维的，因此结果为$k×1$维度。注，我们不对方差特征进行处理。

## 选择主成分的数量

主要成分分析是减少投射的平均均方误差：

训练集的方差为：$\dfrac {1}{m}\sum^{m}_{i=1}\left| x^{\left( i\right) }\right| ^{2}$

我们希望在平均均方误差与训练集方差的比例尽可能小的情况下选择尽可能小的$k$值。

如果我们希望这个比例小于1%，就意味着原本数据的偏差有99%都保留下来了，如果我们选择保留95%的偏差，便能非常显著地降低模型中特征的维度了。

我们可以先令$k=1$，然后进行主要成分分析，获得$U_{reduce}$和$z$，然后计算比例是否小于1%。如果不是的话再令$k=2$，如此类推，直到找到可以使得比例小于1%的最小$k$ 值（原因是各个特征之间通常情况存在某种相关性）。

还有一些更好的方式来选择$k$，当我们在**Octave**中调用“**svd**”函数的时候，我们获得三个参数：`[U, S, V] = svd(sigma)`。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/a4477d787f876ae4e72cb416a2cb0b8a.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/a4477d787f876ae4e72cb416a2cb0b8a.jpg)

其中的$S$是一个$n×n$的矩阵，只有对角线上有值，而其它单元都是0，我们可以使用这个矩阵来计算平均均方误差与训练集方差的比例： $$\dfrac {\dfrac {1}{m}\sum^{m}*{i=1}\left| x^{\left( i\right) }-x^{\left( i\right) }*{approx}\right| ^{2}}{\dfrac {1}{m}\sum^{m}*{i=1}\left| x^{(i)}\right| ^{2}}=1-\dfrac {\Sigma^{k}*{i=1}S_{ii}}{\Sigma^{m}*{i=1}S*{ii}}\leq 1%$$

也就是：$$\frac {\Sigma^{k}*{i=1}s*{ii}}{\Sigma^{n}*{i=1}s*{ii}}\geq0.99$$

在压缩过数据后，我们可以采用如下方法来近似地获得原有的特征：$$x^{\left( i\right) }*{approx}=U*{reduce}z^{(i)}$$

## 重建的压缩表示

如果这是一个压缩算法，应该能回到这个压缩表示，回到你原有的高维数据的一种近似。

所以，给定的$z^{(i)}$，这可能100维，怎么回到你原来的表示$x^{(i)}$，这可能是1000维的数组？

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0a4edcb9c0d0a3812a50b3e95ef3912a.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/0a4edcb9c0d0a3812a50b3e95ef3912a.png)

**PCA**算法，我们可能有一个这样的样本。如图中样本$x^{(1)}$,$x^{(2)}$。我们做的是，我们把这些样本投射到图中这个一维平面。然后现在我们需要只使用一个实数，比如$z^{(1)}$，指定这些点的位置后他们被投射到这一个三维曲面。给定一个点$z^{(1)}$，我们怎么能回去这个原始的二维空间呢？$x$为2维，$z$为1维，$z=U^{T}*{reduce}x$，相反的方程为：$x*{appox}=U_{reduce}\cdot z$,$x_{appox}\approx x$。如图：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/66544d8fa1c1639d80948006f7f4a8ff.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/66544d8fa1c1639d80948006f7f4a8ff.png)

如你所知，这是一个漂亮的与原始数据相当相似。所以，这就是你从低维表示$z$回到未压缩的表示。我们得到的数据的一个之间你的原始数据 $x$，我们也把这个过程称为重建原始数据。

当我们认为试图重建从压缩表示 $x$ 的初始值。所以，给定未标记的数据集，您现在知道如何应用**PCA**，你的带高维特征$x$和映射到这的低维表示$z$。这个视频，希望你现在也知道如何采取这些低维表示$z$，映射到备份到一个近似你原有的高维数据。

现在你知道如何实施应用**PCA**，我们将要做的事是谈论一些技术在实际使用**PCA**很好，特别是，在接下来的视频中，我想谈一谈关于如何选择$k$。

## 主成分分析的应用建议

假使我们正在针对一张 100×100像素的图片进行某个计算机视觉的机器学习，即总共有10000 个特征。

1. 第一步是运用主要成分分析将数据压缩至1000个特征
2. 然后对训练集运行学习算法
   1. 在预测时，采用之前学习而来的$U_{reduce}$将输入的特征$x$转换成特征向量$z$，然后再进行预测

注：如果我们有交叉验证集合测试集，也采用对训练集学习而来的$U_{reduce}$。

错误的主要成分分析情况：一个常见错误使用主要成分分析的情况是，将其用于减少过拟合（减少了特征的数量）。这样做非常不好，不如尝试正则化处理。原因在于主要成分分析只是近似地丢弃掉一些特征，它并不考虑任何与结果变量有关的信息，因此可能会丢失非常重要的特征。然而当我们进行正则化处理时，会考虑到结果变量，不会丢掉重要的数据。

另一个常见的错误是，默认地将主要成分分析作为学习过程中的一部分，这虽然很多时候有效果，最好还是从所有原始特征开始，只在有必要的时候（算法运行太慢或者占用太多内存）才考虑采用主要成分分析。

# 异常检测

所谓的异常检测问题就是：我们希望知道这个新的飞机引擎是否有某种异常，或者说，我们希望判断这个引擎是否需要进一步测试。因为，如果它看起来像一个正常的引擎，那么我们可以直接将它运送到客户那里，而不需要进一步的测试。

给定数据集 $x^{(1)},x^{(2)},..,x^{(m)}$，我们假使数据集是正常的，我们希望知道新的数据 $x_{test}$ 是不是异常的，即这个测试数据不属于该组数据的几率如何。我们所构建的模型应该能根据该测试数据的位置告诉我们其属于一组数据的可能性 $p(x)$。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/65afdea865d50cba12d4f7674d599de5.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/65afdea865d50cba12d4f7674d599de5.png)

上图中，在蓝色圈内的数据属于该组数据的可能性较高，而越是偏远的数据，其属于该组数据的可能性就越低。

这种方法称为密度估计，表达如下：

$$ if \quad p(x) \begin{cases} < \varepsilon & anomaly \

> =\varepsilon & normal \end{cases} $$

欺诈检测：

$x^{(i)} = {用户的第i个活动特征}$

模型$p(x)$ 为我们其属于一组数据的可能性，通过$p(x) < \varepsilon$检测非正常用户。

再一个例子是检测一个数据中心，特征可能包含：内存使用情况，被访问的磁盘数量，**CPU**的负载，网络的通信量等。根据这些特征可以构建一个模型，用来判断某些计算机是不是有可能出错了。

## 高斯分布

通常如果我们认为变量 $x$ 符合高斯分布 $x \sim N(\mu, \sigma^2)$则其概率密度函数为： $p(x,\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ 我们可以利用已有的数据来预测总体中的$μ$和$σ^2$的计算方法如下： $\mu=\frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}$

$\sigma^2=\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}-\mu)^2$

高斯分布样例：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fcb35433507a56631dde2b4e543743ee.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/fcb35433507a56631dde2b4e543743ee.png)

注：机器学习中对于方差我们通常只除以$m$而非统计学中的$(m-1)$。这里顺便提一下，在实际使用中，到底是选择使用$1/m$还是$1/(m-1)$其实区别很小，只要你有一个还算大的训练集，在机器学习领域大部分人更习惯使用$1/m$这个版本的公式。这两个版本的公式在理论特性和数学特性上稍有不同，但是在实际使用中，他们的区别甚小，几乎可以忽略不计。

异常检测算法：

对于给定的数据集 $x^{(1)},x^{(2)},...,x^{(m)}$，我们要针对每一个特征计算 $\mu$ 和 $\sigma^2$ 的估计值。

$\mu_j=\frac{1}{m}\sum\limits_{i=1}^{m}x_j^{(i)}$

$\sigma_j^2=\frac{1}{m}\sum\limits_{i=1}^m(x_j^{(i)}-\mu_j)^2$

一旦我们获得了平均值和方差的估计值，给定新的一个训练实例，根据模型计算 $p(x)$：

$p(x)=\prod\limits_{j=1}^np(x_j;\mu_j,\sigma_j^2)=\prod\limits_{j=1}^1\frac{1}{\sqrt{2\pi}\sigma_j}exp(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})$

当$p(x) < \varepsilon$时，为异常。

下图是一个由两个特征的训练集，以及特征的分布情况：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ba47767a11ba39a23898b9f1a5a57cc5.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/ba47767a11ba39a23898b9f1a5a57cc5.png)

下面的三维图表表示的是密度估计函数，$z$轴为根据两个特征的值所估计$p(x)$值：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/82b90f56570c05966da116c3afe6fc91.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/82b90f56570c05966da116c3afe6fc91.jpg)

我们选择一个$\varepsilon$，将$p(x) = \varepsilon$作为我们的判定边界，当$p(x) > \varepsilon$时预测数据为正常数据，否则为异常。

在这段视频中，我们介绍了如何拟合$p(x)$，也就是 $x$的概率值，以开发出一种异常检测算法。同时，在这节课中，我们也给出了通过给出的数据集拟合参数，进行参数估计，得到参数 $\mu$ 和 $\sigma$，然后检测新的样本，确定新样本是否是异常。

## 开发与评价异常检测系统

异常检测算法是一个非监督学习算法，意味着我们无法根据结果变量 $ y$ 的值来告诉我们数据是否真的是异常的。我们需要另一种方法来帮助检验算法是否有效。当我们开发一个异常检测系统时，我们从带标记（异常或正常）的数据着手，我们从其中选择一部分正常数据用于构建训练集，然后用剩下的正常数据和异常数据混合的数据构成交叉检验集和测试集。

例如：我们有10000台正常引擎的数据，有20台异常引擎的数据。 我们这样分配数据：

6000台正常引擎的数据作为训练集

2000台正常引擎和10台异常引擎的数据作为交叉检验集

2000台正常引擎和10台异常引擎的数据作为测试集

具体的评价方法如下：

1. 根据测试集数据，我们估计特征的平均值和方差并构建$p(x)$函数
2. 对交叉检验集，我们尝试使用不同的$\varepsilon$值作为阀值，并预测数据是否异常，根据$F1$值或者查准率与查全率的比例来选择 $\varepsilon$
3. 选出 $\varepsilon$ 后，针对测试集进行预测，计算异常检验系统的$F1$值，或者查准率与查全率之比

## 异常检测与监督学习的对比

| 异常检测                                                     | 监督学习                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 非常少量的正向类（异常数据 $y=1$）, 大量的负向类（$y=0$）    | 同时有大量的正向类和负向类                                   |
| 许多不同种类的异常，非常难。根据非常 少量的正向类数据来训练算法。 | 有足够多的正向类实例，足够用于训练 算法，未来遇到的正向类实例可能与训练集中的非常近似。 |
| 未来遇到的异常可能与已掌握的异常、非常的不同。               |                                                              |
| 例如： 欺诈行为检测 生产（例如飞机引擎）检测数据中心的计算机运行状况 | 例如：邮件过滤器 天气预报 肿瘤分类                           |

希望这节课能让你明白一个学习问题的什么样的特征，能让你把这个问题当做是一个异常检测，或者是一个监督学习的问题。另外，对于很多技术公司可能会遇到的一些问题，通常来说，正样本的数量很少，甚至有时候是0，也就是说，出现了太多没见过的不同的异常类型，那么对于这些问题，通常应该使用的算法就是异常检测算法。

## 特征选择

异常检测假设特征符合高斯分布，如果数据的分布不是高斯分布，异常检测算法也能够工作，但是最好还是将数据转换成高斯分布，例如使用对数函数：$x= log(x+c)$，其中 $c$ 为非负常数； 或者 $x=x^c$，$c$为 0-1 之间的一个分数，等方法。(编者注：在**python**中，通常用`np.log1p()`函数，$log1p$就是 $log(x+1)$，可以避免出现负数结果，反向函数就是`np.expm1()`)

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0990d6b7a5ab3c0036f42083fe2718c6.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/0990d6b7a5ab3c0036f42083fe2718c6.jpg)

误差分析：

一个常见的问题是一些异常的数据可能也会有较高的$p(x)$值，因而被算法认为是正常的。这种情况下误差分析能够帮助我们，我们可以分析那些被算法错误预测为正常的数据，观察能否找出一些问题。我们可能能从问题中发现我们需要增加一些新的特征，增加这些新特征后获得的新算法能够帮助我们更好地进行异常检测。

异常检测误差分析：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f406bc738e5e032be79e52b6facfa48e.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/f406bc738e5e032be79e52b6facfa48e.png)

我们通常可以通过将一些相关的特征进行组合，来获得一些新的更好的特征（异常数据的该特征值异常地大或小），例如，在检测数据中心的计算机状况的例子中，我们可以用**CPU**负载与网络通信量的比例作为一个新的特征，如果该值异常地大，便有可能意味着该服务器是陷入了一些问题中。

# 推荐系统

假使我们是一个电影供应商，我们有 5 部电影和 4 个用户，我们要求用户为电影打分。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c2822f2c28b343d7e6ade5bd40f3a1fc.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/c2822f2c28b343d7e6ade5bd40f3a1fc.png)

前三部电影是爱情片，后两部则是动作片，我们可以看出**Alice**和**Bob**似乎更倾向与爱情片， 而 **Carol** 和 **Dave** 似乎更倾向与动作片。并且没有一个用户给所有的电影都打过分。我们希望构建一个算法来预测他们每个人可能会给他们没看过的电影打多少分，并以此作为推荐的依据。

下面引入一些标记：

$n_u$ 代表用户的数量

$n_m$ 代表电影的数量

$r(i, j)$ 如果用户j给电影 $i$ 评过分则 $r(i,j)=1$

$y^{(i, j)}$ 代表用户 $j$ 给电影$i$的评分

$m_j$代表用户 $j$ 评过分的电影的总数

假设每部电影都有两个特征，如$x_1$代表电影的浪漫程度，$x_2$ 代表电影的动作程度。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/747c1fd6bff694c6034da1911aa3314b.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/747c1fd6bff694c6034da1911aa3314b.png)

则每部电影都有一个特征向量，如$x^{(1)}$是第一部电影的特征向量为[0.9 0]。

下面我们要基于这些特征来构建一个推荐系统算法。 假设我们采用线性回归模型，我们可以针对每一个用户都训练一个线性回归模型，如${{\theta }^{(1)}}$是第一个用户的模型的参数。 于是，我们有：

$\theta^{(j)}$用户 $j$ 的参数向量

$x^{(i)}$电影 $i$ 的特征向量

对于用户 $j$ 和电影 $i$，我们预测评分为：$(\theta^{(j)})^T x^{(i)}$

代价函数

针对用户 $j$，该线性回归模型的代价为预测误差的平方和，加上正则化项： $$ \min_{\theta (j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\left(\theta_{k}^{(j)}\right)^2 $$

其中 $i:r(i,j)$表示我们只计算那些用户 $j$ 评过分的电影。在一般的线性回归模型中，误差项和正则项应该都是乘以$1/2m$，在这里我们将$m$去掉。并且我们不对方差项$\theta_0$进行正则化处理。

上面的代价函数只是针对一个用户的，为了学习所有用户，我们将所有用户的代价函数求和： $$ \min_{\theta^{(1)},...,\theta^{(n_u)}} \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 $$ 如果我们要用梯度下降法来求解最优解，我们计算代价函数的偏导数后得到梯度下降的更新公式为：

$$ \theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)} \quad (\text{for} , k = 0) $$

$$ \theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}+\lambda\theta_k^{(j)}\right) \quad (\text{for} , k\neq 0) $$

## 协同过滤

在之前的基于内容的推荐系统中，对于每一部电影，我们都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。相反地，如果我们拥有用户的参数，我们可以学习得出电影的特征。

$$ \mathop{min}\limits_{x^{(1)},...,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j{r(i,j)=1}}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2 $$ 但是如果我们既没有用户的参数，也没有电影的特征，这两种方法都不可行了。协同过滤算法可以同时学习这两者。

我们的优化目标便改为同时针对$x$和$\theta$进行。 $$ J(x^{(1)},...x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 $$

对代价函数求偏导数的结果如下：

$$ x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}+\lambda x_k^{(i)}\right) $$

$$ \theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}+\lambda \theta_k^{(j)}\right) $$

注：在协同过滤从算法中，我们通常不使用方差项，如果需要的话，算法会自动学得。 协同过滤算法使用步骤如下：

1. 初始 $x^{(1)},x^{(1)},...x^{(nm)},\ \theta^{(1)},\theta^{(2)},...,\theta^{(n_u)}$为一些随机小值
2. 使用梯度下降算法最小化代价函数
3. 在训练完算法后，我们预测$(\theta^{(j)})^Tx^{(i)}$为用户 $j$ 给电影 $i$ 的评分

通过这个学习过程获得的特征矩阵包含了有关电影的重要数据，这些数据不总是人能读懂的，但是我们可以用这些数据作为给用户推荐电影的依据。

例如，如果一位用户正在观看电影 $x^{(i)}$，我们可以寻找另一部电影$x^{(j)}$，依据两部电影的特征向量之间的距离$\left| {{x}^{(i)}}-{{x}^{(j)}} \right|$的大小。

## 协同过滤算法

协同过滤优化目标：

给定$x^{(1)},...,x^{(n_m)}$，估计$\theta^{(1)},...,\theta^{(n_u)}$： $$ \min_{\theta^{(1)},...,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 $$

给定$\theta^{(1)},...,\theta^{(n_u)}$，估计$x^{(1)},...,x^{(n_m)}$：

同时最小化$x^{(1)},...,x^{(n_m)}$和$\theta^{(1)},...,\theta^{(n_u)}$： $$ J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 $$

$$ \min_{x^{(1)},...,x^{(n_m)} \\ \theta^{(1)},...,\theta^{(n_u)}}J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}) $$

## 向量化：低秩矩阵分解

举例子：

1. 当给出一件产品时，你能否找到与之相关的其它产品。
2. 一位用户最近看上一件产品，有没有其它相关的产品，你可以推荐给他。

我将要做的是：实现一种选择的方法，写出协同过滤算法的预测情况。

我们有关于五部电影的数据集，我将要做的是，将这些用户的电影评分，进行分组并存到一个矩阵中。

我们有五部电影，以及四位用户，那么 这个矩阵 $Y$ 就是一个5行4列的矩阵，它将这些电影的用户评分数据都存在矩阵里：

| **Movie**            | **Alice (1)** | **Bob (2)** | **Carol (3)** | **Dave (4)** |
| -------------------- | ------------- | ----------- | ------------- | ------------ |
| Love at last         | 5             | 5           | 0             | 0            |
| Romance forever      | 5             | ?           | ?             | 0            |
| Cute puppies of love | ?             | 4           | 0             | ?            |
| Nonstop car chases   | 0             | 0           | 5             | 4            |
| Swords vs. karate    | 0             | 0           | 5             | ?            |

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/42a92e07b32b593bb826f8f6bc4d9eb3.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/42a92e07b32b593bb826f8f6bc4d9eb3.png)

推出评分：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c905a6f02e201a4767d869b3791e8aeb.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/c905a6f02e201a4767d869b3791e8aeb.png)

找到相关影片：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0a8b49da1ab852f2996a02afcaca2322.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/0a8b49da1ab852f2996a02afcaca2322.png)

现在既然你已经对特征参数向量进行了学习，那么我们就会有一个很方便的方法来度量两部电影之间的相似性。例如说：电影 $i$ 有一个特征向量$x^{(i)}$，你是否能找到一部不同的电影 $j$，保证两部电影的特征向量之间的距离$x^{(i)}$和$x^{(j)}$很小，那就能很有力地表明电影$i$和电影 $j$ 在某种程度上有相似，至少在某种意义上，某些人喜欢电影 $i$，或许更有可能也对电影 $j$ 感兴趣。总结一下，当用户在看某部电影 $i$ 的时候，如果你想找5部与电影非常相似的电影，为了能给用户推荐5部新电影，你需要做的是找出电影 $j$，在这些不同的电影中与我们要找的电影 $i$ 的距离最小，这样你就能给你的用户推荐几部不同的电影了。

## 均值归一化

让我们来看下面的用户评分数据：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/54b1f7c3131aed24f9834d62a6835642.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/54b1f7c3131aed24f9834d62a6835642.png)

如果我们新增一个用户 **Eve**，并且 **Eve** 没有为任何电影评分，那么我们以什么为依据为**Eve**推荐电影呢？

我们首先需要对结果 $Y $矩阵进行均值归一化处理，将每一个用户对某一部电影的评分减去所有用户对该电影评分的平均值：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/9ec5cb55e14bd1462183e104f8e02b80.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/9ec5cb55e14bd1462183e104f8e02b80.png)

然后我们利用这个新的 $Y$ 矩阵来训练算法。 如果我们要用新训练出的算法来预测评分，则需要将平均值重新加回去，预测$(\theta^{(j)})^T x^{(i)}+\mu_i$，对于**Eve**，我们的新模型会认为她给每部电影的评分都是该电影的平均分。

# 大规模机器学习

如果我们有一个低方差的模型，增加数据集的规模可以帮助你获得更好的结果。我们应该怎样应对一个有100万条记录的训练集？

以线性回归模型为例，每一次梯度下降迭代，我们都需要计算训练集的误差的平方和，如果我们的学习算法需要有20次迭代，这便已经是非常大的计算代价。

首先应该做的事是去检查一个这么大规模的训练集是否真的必要，也许我们只用1000个训练集也能获得较好的效果，我们可以绘制学习曲线来帮助判断。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/bdf069136b4b661dd14158496d1d1419.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/bdf069136b4b661dd14158496d1d1419.png)	

## 随机梯度下讲法

如果我们一定需要一个大规模的训练集，我们可以尝试使用随机梯度下降法来代替批量梯度下降法。

在随机梯度下降法中，我们定义代价函数为一个单一训练实例的代价：

 $$cost\left( \theta, \left( {x}^{(i)} , {y}^{(i)} \right) \right) = \frac{1}{2}\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{{(i)}} \right)^{2}$$

**随机**梯度下降算法为：首先对训练集随机“洗牌”，然后： Repeat (usually anywhere between1-10){

**for** $i = 1:m${

 $\theta:={\theta}*{j}-\alpha\left( {h}*{\theta}\left({x}^{(i)}\right)-{y}^{(i)} \right){{x}_{j}}^{(i)}$

 (**for** $j=0:n$)

 } }

随机梯度下降算法在每一次计算之后便更新参数 ${{\theta }}$ ，而不需要首先将所有的训练集求和，在梯度下降算法还没有完成一次迭代时，随机梯度下降算法便已经走出了很远。但是这样的算法存在的问题是，不是每一步都是朝着”正确”的方向迈出的。因此算法虽然会逐渐走向全局最小值的位置，但是可能无法站到那个最小值的那一点，而是在最小值点附近徘徊。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/9710a69ba509a9dcbca351fccc6e7aae.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/9710a69ba509a9dcbca351fccc6e7aae.jpg)

## 小批量梯度下降

小批量梯度下降算法是介于批量梯度下降算法和随机梯度下降算法之间的算法，每计算常数$b$次训练实例，便更新一次参数 ${{\theta }}$ 。 **Repeat** {

**for** $i = 1:m${

 $\theta:={\theta}_{j}-\alpha\frac{1}{b}\sum\limits{k=i}^{i+b-1}\left( {h}_{\theta}\left({x}^{(k)}\right)-{y}^{(k)} \right){{x}_{j}}^{(k)}$

 (**for** $j=0:n$)

 $ i +=10 $

 } }

通常我们会令 $b$ 在 2-100 之间。这样做的好处在于，我们可以用向量化的方式来循环 $b$个训练实例，如果我们用的线性代数函数库比较好，能够支持平行处理，那么算法的总体表现将不受影响（与随机梯度下降相同）。

## 随机梯度下降收敛

现在我们介绍随机梯度下降算法的调试，以及学习率 $α$ 的选取。

在批量梯度下降中，我们可以令代价函数$J$为迭代次数的函数，绘制图表，根据图表来判断梯度下降是否收敛。但是，在大规模的训练集的情况下，这是不现实的，因为计算代价太大了。

在随机梯度下降中，我们在每一次更新 ${{\theta }}$ 之前都计算一次代价，然后每$x$次迭代后，求出这$x$次对训练实例计算代价的平均值，然后绘制这些平均值与$x$次迭代的次数之间的函数图表。

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/76fb1df50bdf951f4b880fa66489e367.png)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/76fb1df50bdf951f4b880fa66489e367.png)

当我们绘制这样的图表时，可能会得到一个颠簸不平但是不会明显减少的函数图像（如上面左下图中蓝线所示）。我们可以增加$α$来使得函数更加平缓，也许便能看出下降的趋势了（如上面左下图中红线所示）；或者可能函数图表仍然是颠簸不平且不下降的（如洋红色线所示），那么我们的模型本身可能存在一些错误。

如果我们得到的曲线如上面右下方所示，不断地上升，那么我们可能会需要选择一个较小的学习率$α$。

我们也可以令学习率随着迭代次数的增加而减小，例如令：

 $$\alpha = \frac{const1}{iterationNumber + const2}$$

随着我们不断地靠近全局最小值，通过减小学习率，我们迫使算法收敛而非在最小值附近徘徊。 但是通常我们不需要这样做便能有非常好的效果了，对$α$进行调整所耗费的计算通常不值得

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f703f371dbb80d22fd5e4aec48aa9fd4.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/f703f371dbb80d22fd5e4aec48aa9fd4.jpg)

总结下，这段视频中，我们介绍了一种方法，近似地监测出随机梯度下降算法在最优化代价函数中的表现，这种方法不需要定时地扫描整个训练集，来算出整个样本集的代价函数，而是只需要每次对最后1000个，或者多少个样本，求一下平均值。应用这种方法，你既可以保证随机梯度下降法正在正常运转和收敛，也可以用它来调整学习速率$α$的大小。

## 在线学习

今天，许多大型网站或者许多大型网络公司，使用不同版本的在线学习机制算法，从大批的涌入又离开网站的用户身上进行学习。特别要提及的是，如果你有一个由连续的用户流引发的连续的数据流，进入你的网站，你能做的是使用一个在线学习机制，从数据流中学习用户的偏好，然后使用这些信息来优化一些关于网站的决策。

假定你有一个提供运输服务的公司，用户们来向你询问把包裹从**A**地运到**B**地的服务，同时假定你有一个网站，让用户们可多次登陆，然后他们告诉你，他们想从哪里寄出包裹，以及包裹要寄到哪里去，也就是出发地与目的地，然后你的网站开出运输包裹的的服务价格。比如，我会收取$50来运输你的包裹，我会收取$20之类的，然后根据你开给用户的这个价格，用户有时会接受这个运输服务，那么这就是个正样本，有时他们会走掉，然后他们拒绝购买你的运输服务，所以，让我们假定我们想要一个学习算法来帮助我们，优化我们想给用户开出的价格。

一个算法来从中学习的时候来模型化问题在线学习算法指的是对数据流而非离线的静态数据集的学习。许多在线网站都有持续不断的用户流，对于每一个用户，网站希望能在不将数据存储到数据库中便顺利地进行算法学习。

假使我们正在经营一家物流公司，每当一个用户询问从地点A至地点B的快递费用时，我们给用户一个报价，该用户可能选择接受（$y=1$）或不接受（$y=0$）。

现在，我们希望构建一个模型，来预测用户接受报价使用我们的物流服务的可能性。因此报价 是我们的一个特征，其他特征为距离，起始地点，目标地点以及特定的用户数据。模型的输出是:$p(y=1)$。

在线学习的算法与随机梯度下降算法有些类似，我们对单一的实例进行学习，而非对一个提前定义的训练集进行循环。 Repeat forever (as long as the website is running) { Get $\left(x,y\right)$ corresponding to the current user $\theta:={\theta}*{j}-\alpha\left( {h}*{\theta}\left({x}\right)-{y} \right){{x}_{j}}$ (**for** $j=0:n$) }

一旦对一个数据的学习完成了，我们便可以丢弃该数据，不需要再存储它了。这种方式的好处在于，我们的算法可以很好的适应用户的倾向性，算法可以针对用户的当前行为不断地更新模型以适应该用户。

每次交互事件并不只产生一个数据集，例如，我们一次给用户提供3个物流选项，用户选择2项，我们实际上可以获得3个新的训练实例，因而我们的算法可以一次从3个实例中学习并更新模型。

这些问题中的任何一个都可以被归类到标准的，拥有一个固定的样本集的机器学习问题中。或许，你可以运行一个你自己的网站，尝试运行几天，然后保存一个数据集，一个固定的数据集，然后对其运行一个学习算法。但是这些是实际的问题，在这些问题里，你会看到大公司会获取如此多的数据，真的没有必要来保存一个固定的数据集，取而代之的是你可以使用一个在线学习算法来连续的学习，从这些用户不断产生的数据中来学习。这就是在线学习机制，然后就像我们所看到的，我们所使用的这个算法与随机梯度下降算法非常类似，唯一的区别的是，我们不会使用一个固定的数据集，我们会做的是获取一个用户样本，从那个样本中学习，然后丢弃那个样本并继续下去，而且如果你对某一种应用有一个连续的数据流，这样的算法可能会非常值得考虑。当然，在线学习的一个优点就是，如果你有一个变化的用户群，又或者你在尝试预测的事情，在缓慢变化，就像你的用户的品味在缓慢变化，这个在线学习算法，可以慢慢地调试你所学习到的假设，将其调节更新到最新的用户行为。

## 映射化简和数据并行

映射化简和数据并行对于大规模机器学习问题而言是非常重要的概念。之前提到，如果我们用批量梯度下降算法来求解大规模数据集的最优解，我们需要对整个训练集进行循环，计算偏导数和代价，再求和，计算代价非常大。如果我们能够将我们的数据集分配给不多台计算机，让每一台计算机处理数据集的一个子集，然后我们将计所的结果汇总在求和。这样的方法叫做映射简化。

具体而言，如果任何学习算法能够表达为，对训练集的函数的求和，那么便能将这个任务分配给多台计算机（或者同一台计算机的不同**CPU** 核心），以达到加速处理的目的。

例如，我们有400个训练实例，我们可以将批量梯度下降的求和任务分配给4台计算机进行处理：

[![img](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/919eabe903ef585ec7d08f2895551a1f.jpg)](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/blob/master/images/919eabe903ef585ec7d08f2895551a1f.jpg)

很多高级的线性代数函数库已经能够利用多核**CPU**的多个核心来并行地处理矩阵运算，这也是算法的向量化实现如此重要的缘故（比调用循环快）。