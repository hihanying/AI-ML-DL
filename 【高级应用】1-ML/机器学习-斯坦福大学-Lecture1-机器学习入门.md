机器学习-斯坦福大学-Lecture1-机器学习入门
---
在网上搜索过很多资料，最终确定以Coursera上斯坦福大学的公开课[ 机器学习(Andrew Ng)][1]作为入门资料

摘要：

[TOC]

-------------------
### 1.1欢迎

#### 应用场景举例
- 你打开谷歌、必应搜索到你需要的内容，正是因为他们有良好的学习算法
- 你用 Facebook 或苹果的图片分类程序他能认出你朋友的照片，这也是机器学习
- 你每次阅读你的电子邮件垃圾邮件筛选器，可以帮你过滤大量的垃圾邮件这也是一种学习算法
- 每次你去亚马逊或 Netflix 或 iTunes Genius，它都会给出其他电影或产品或音乐的建议，这是一种学习算法

#### 为什么机器学习如此受欢迎呢？
机器学习不只是用于人工智能领域
- Database mining 数据挖掘Large datasets from growth of automation/web. E.g., Web click data（Web点击数据）, medical records, biology（计算生物学）, engineering
- Applications can’t program by hand.E.g., Autonomous helicopter（自动驾驶直升机）, handwriting recognition（手写识别）, most of Natural Language Processing (自然语言处理), Computer Vision（计算机视觉）. 
- Self-customizing programs（私人定制）E.g., Amazon, Netflix product recommendations（产品推荐）
- Understanding human learning (学习算法被用来理解人类的学习和了解大脑).

---
### 1.2机器学习是什么

#### 机器学习的定义
- Arthur Samuel：在进行特定编程的情况下，给予计算机学习能力的领域。
- Tom Mitchell：一个程序被认为能从**经验 E** 中学习，解决**任务 T**，达到**性能度量值 P**，当且仅当，有了经验 E 后，经过 P 评判，程序在处理 T 时的性能有所提升。

#### 练习题
Suppose your email program watches which emails you do or do not mark as spam, and based on that learns how to better filter spam. What is the task T in this setting? - Classifying emails as spam or not spam. **(task T)**
- Watching you label emails as spam or not spam. **(experience E)**
- The number (or fraction) of emails correctly classified as spam/not spam. **(performance measure P)**
- None of the above—this is not a machine learning problem.

#### 学习算法的分类
- Machine learning algorithms:    
 - Supervised learning监督学习    
 - Unsupervised learning无监督学习
- Others: 
 - Reinforcement learning强化学习    
 - recommender systems推荐系统

---

### 1.3监督学习

#### 基本思想
我们数据集中的每个样本都有相应的“正确答案”，再根据这些样本作出预测，预测的结果可能是一个连续输出，也可能是一组离散的结果。

#### 两类问题
- 回归问题：推测出这一系列连续值得属性
Eg：预测房价：如下图，横轴表示房子的面积，单位是平方英尺，纵轴表示房价，单位是千美元。那基于这组数据，假如你有一个朋友，他有一套 750 平方英尺房子，现在他希望把房子卖掉，他想知道这房子能卖多少钱。![这里写图片描述](http://img.blog.csdn.net/20170116113718413?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
 - 可以应用学习算法，在这组数据中拟合一条直线，根据这条线我们可以推测出，这套房子可能卖$150, 000    
 - 可以应用学习算法，用二次方程去拟合可能效果会更好，根据二次方程的曲线，我们可以从这个点推测出，这套房子能卖接近$200, 000。
>**总结**：监督学习指的就是我们**给学习算法一个数据集**。这个**数据集由“正确答案”组成**。在房价的例子中，我们给了一系列房子的数据，我们给定数据集中每个样本的正确价格，即它们实际的售价。**然后运用学习算法**，算出更多的正确答案。比如你朋友那个新房子的价格。用术语来讲，这叫做**回归问题**。我们试着**推测出一个连续值的结果**，即房子的价格。一般房子的价格会记到美分，所以房价实际上是一系列离散的值， 但是我们通常又把房价看成实数，看成是标量，所以又把它看成一个连续的数值。

- 分类问题：推测出离散的输出值
Eg：假设说你想通过查看病历来推测乳腺癌良性与否，在这个数据集中，横轴表示肿瘤的大小，纵轴上，我标出 1 和 0 表示是或者不是恶性肿瘤。我们之前见过的肿瘤，如果是恶性则记为 1 ，不是恶性，或者说良性记为 0。![这里写图片描述](http://img.blog.csdn.net/20170116114837624?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
机器学习的问题就在于，你能否**估算出肿瘤是恶性的或是良性的概率**。用术语来讲，这是一个**分类问题**。分类指的是，我们试着**推测出离散的输出值**： 0 或 1 良性或恶性，而事实上在分类问题中，输出可能不止两个值。比如说可能有三种乳腺癌，所以你希望预测离散输出 0、 1、 2、3。 0 代表良性， 1 表示第一类乳腺癌， 2 表示第二类癌症， 3 表示第三类，但这也是分类问题。
> 用不同的符号来表示这些数据（良性与否），将结果放在**一维向量**里，作为判断病情结果一个特征，之后我们还会遇到不只有一种特征，我们有一个算法，叫**支持向量机**，里面有一个巧妙地数学技巧，可以让计算机处理无限多个特征。
> ![这里写图片描述](http://img.blog.csdn.net/20170116115711079?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 练习题
You’re running a company, and you want to develop learning algorithms to address each of two problems.
- Problem 1: You have a large inventory of identical items. You want to predict how many of these items will sell over the next 3 months.**（回归问题）**
- Problem 2: You’d like software to examine individual customer accounts, and for each account decide if it has been hacked/compromised. **（分类问题）**

Should you treat these as **classification** or as **regression** problems?

---

### 1.4无监督学习

#### 是什么

无监督学习，它是学习策略，交给算法大量的数据，并让算法为我们从数据中找出某种结构。即**无监督学习中没有任何的标签或者是有相同的标签或者就是没标签**。针对数据集，**无监督学习就能判断出数据有两个不同的聚集簇**，这叫做聚类算法。
#### 聚类算法应用：
- **谷歌新闻**：谷歌新闻每天都在，收集非常多，非常多的网络的新闻内容。它再将这些新闻分组，组成有关联的新闻。所以谷歌新闻做的就是搜索非常多的新闻事件，自动地把它们聚类到一起。所以，这些新闻事件全是同一主题的，所以显示到一起。
- **基因学 DNA 微观数据**：基本思想是输入一组不同个体，对其中的每个个体，你要分析出它们是否有一个特定的基因。技术上，你要分析多少特定基因已经表达。你能做的就是运行一个聚类算法，把个体聚类到不同的类或不同类型的组（人）……
> 因为我们没有给算法正确答案来回应数据集中的数据，所以这就是无监督学习。
- **组织大型计算机集群**：在大数据中心工作，那里有大型的计算机集群，要解决什么样的机器易于协同地工作，如果你能够让那些机器协同工作，你就能让你的数据中心工作得更高效。
- **市场分割**：许多公司有大型的数据库，存储消费者信息。所以，你能检索这些顾客数据集，自动地发现市场分类，并自动地把顾客划分到不同的细分市场中，你才能自动并更有效地销售或不同的细分市场一起进行销售
- **用于天文数据分析**

#### 鸡尾酒宴问题
在一个这样的鸡尾酒宴中的两个人，他俩同时都在说话，假设现在是在个有些小的鸡尾酒宴中。我们放两个麦克风在房间中，因为这些麦克风在两个地方，离说话人的距离不同每个麦克风记录下不同的声音，虽然是同样的两个说话人。听起来像是两份录音被叠加到一起，或是被归结到一起，产生了我们现在的这些录音。
聚类算法会区分出两个音频资源，并输出。从**音频中分离出音频**可以只用一行代码来完成：
```octave [W,s,v] =svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x'); ```

使用[ **Octave 编程环境**][2]。Octave,是免费的开源软件，使用一个像 Octave 或 Matlab的工具，许多学习算法变得只有几行代码就可实现。
对大量机器学习算法，第一步就是建原型，在 Octave 建软件原型，因为软件在 Octave 中可以 令人难以置信地、快速地实现这些学习算法，在你已经让它工作后，你才移植它到 C++或 Java 或别的语言。

>### 附：
> - [Octave 编程环境安装（英文）][3]
> - [Octave 编程环境安装（中文）][4]

---------
[1]: https://www.coursera.org/learn/machine-learning
[2]: http://www.gnu.org/software/octave/
[3]:http://blog.csdn.net/yinlili2010/article/details/39971353
[4]:http://blog.sina.com.cn/s/blog_96d2e7190101acw5.html
