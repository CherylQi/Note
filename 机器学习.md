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