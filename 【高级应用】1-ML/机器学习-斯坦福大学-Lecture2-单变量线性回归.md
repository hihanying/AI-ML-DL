机器学习-斯坦福大学-Lecture2-单变量线性回归
---
[TOC]

---
### 2.1模型表示
>例子：这个例子是预测住房价格的，使用一个数据集，数据集包含俄勒冈州波特兰市的住房价格。我们要根据不同房屋尺寸所售出的价格，画出我的数据集。比方说，如果有个朋友的房子是 1250 平方尺大小，要告诉他们这房子能卖多少钱。那么，我们可以做的一件事就是构建一个模型，也许是条直线。这就是监督学习算法的一个例子。
>![这里写图片描述](http://img.blog.csdn.net/20170117105601651?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>**这是一个监督学习问题，因为对于每个数据我们都给出了正确答案。具体来说，这是一个回归问题。**

假使我们回归问题的训练集（ Training Set）如下表所示：
![这里写图片描述](http://img.blog.csdn.net/20170117110126831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
我们将要用来描述的这个回归问题标记如下：
>m 代表训练集中实例的数量
x 代表特征/输入变量
y 代表目标变量/输出变量
(x,y) 代表训练集中的实例
($ x^i,y^i $ ) 代表第 i 个观察实例
h 代表学习算法的解决方案或函数也称为假设（ hypothesis）

这是一个监督学习算法的工作方式。
我们将训练集里的房屋价格给学习算法，学习算法工作后输出一个函数通常用h（h是一个从x到y的函数映射）表示，我们用这个函数预测房屋的价格。
而h我们可能用公式![这里写图片描述](http://img.blog.csdn.net/20170117111942867?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)表示。
因为只含有一个输入变量，所以我们称这样的问题为单变量线性回归问题。
![这里写图片描述](http://img.blog.csdn.net/20170117111225347?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

---
### 2.2-2.4代价函数
![这里写图片描述](http://img.blog.csdn.net/20170117121038551?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)我们现在要做的便是为我们的模型选择合适的参数（ parameters） θ0 和 θ1，我们选择的参数决定了我们得到的直线相对于我们的训练集的准确程度，模型所预测的值与训练集中实际值之间的差距就是建模误差（ modeling error ）。我们的目标便是选择出可以使得建模误差的平方和能够最小的模型参数。 即使得代价函数最小。
我们绘制一个三维图，三个坐标分别为 θ0 和 θ1 和 J(θ0,θ1)，可以看出在三维空间中存在一个使得 J(θ0,θ1)最小的点。
![这里写图片描述](http://img.blog.csdn.net/20170117121219219?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
代价函数也被称作平方误差函数/平方误差代价函数。解决回归问题还有其他的代价函数，平方误差代价函数是解决回归问题最常用的手段。
我们常用等高线图（三维图的俯视图）描述代价函数，通过下图我们可以清楚地看出代价函数与假设函数的关系。![这里写图片描述](http://img.blog.csdn.net/20170117121304210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
### 2.5-2.6梯度下降算法
是什么：
>用来求函数最小值的算法

为什么：
>开始时我们随机选择一个参数的组合（ θ0,θ1,...,θn），计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。我们持续这么做直到到到一个局部最小值（ local minimum） ， 因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（ global minimum），选择不同的初始参数组合，可能会找到不同的局部最小值。

怎么做：
>![这里写图片描述](http://img.blog.csdn.net/20170117123805981?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>其中α是**学习速率**（ learning rate） ， 它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大。
>在批量梯度下降中，我们每一次都**同时**让所有的参数减去学习速率乘以代价函数的导数。即**同时更新（如上图左侧算法表达式）**。

过程模拟：
![这里写图片描述](http://img.blog.csdn.net/20170117125418705?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
让我们来看看如果 α 太小或 α 太大会出现什么情况：
![这里写图片描述](http://img.blog.csdn.net/20170117124540458?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170117124548469?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
- If α is too small, gradient descent can be slow.
- If α is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

分析：
>随着逼近最低点，下降速度变得越来越小，由于**下降的减量**是 **α** 和 **J对θ的偏微分** 的**乘积**，J对θ的偏微分在下降的过程中不断减小。
>在梯度下降法中， 当我们接近局部最低点时， **梯度下降法会自动采取更小的幅度**， 这是因为当我们接近局部最低点时， 很显然在局部最低时导数等于零， 所以当我们接近局部最低时， 导数值会自动变得越来越小， 所以梯度下降将自动采取较小的幅度， 这就是梯度下降的做法。

### 2.7梯度下降的线性回归
梯度下降算法和线性回归算法：
![这里写图片描述](http://img.blog.csdn.net/20170117125908351?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
对我们之前运用的线性回归问题运用梯度下降算法，关键在于对代价函数求偏导：
![这里写图片描述](http://img.blog.csdn.net/20170117130151758?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20170117130227665?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
则算法改写为：
![这里写图片描述](http://img.blog.csdn.net/20170117130406212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)---------
