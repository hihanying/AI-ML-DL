����ѧϰ-˹̹����ѧ-Lecture2-���������Իع�
---
[TOC]

---
### 2.1ģ�ͱ�ʾ
>���ӣ����������Ԥ��ס���۸�ģ�ʹ��һ�����ݼ������ݼ��������ո��ݲ������е�ס���۸�����Ҫ���ݲ�ͬ���ݳߴ����۳��ļ۸񣬻����ҵ����ݼ����ȷ�˵������и����ѵķ����� 1250 ƽ���ߴ�С��Ҫ���������ⷿ����������Ǯ����ô�����ǿ�������һ���¾��ǹ���һ��ģ�ͣ�Ҳ������ֱ�ߡ�����Ǽලѧϰ�㷨��һ�����ӡ�
>![����дͼƬ����](http://img.blog.csdn.net/20170117105601651?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>**����һ���ලѧϰ���⣬��Ϊ����ÿ���������Ƕ���������ȷ�𰸡�������˵������һ���ع����⡣**

��ʹ���ǻع������ѵ������ Training Set�����±���ʾ��
![����дͼƬ����](http://img.blog.csdn.net/20170117110126831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
���ǽ�Ҫ��������������ع����������£�
>m ����ѵ������ʵ��������
x ��������/�������
y ����Ŀ�����/�������
(x,y) ����ѵ�����е�ʵ��
($ x^i,y^i $ ) ����� i ���۲�ʵ��
h ����ѧϰ�㷨�Ľ����������Ҳ��Ϊ���裨 hypothesis��

����һ���ලѧϰ�㷨�Ĺ�����ʽ��
���ǽ�ѵ������ķ��ݼ۸��ѧϰ�㷨��ѧϰ�㷨���������һ������ͨ����h��h��һ����x��y�ĺ���ӳ�䣩��ʾ���������������Ԥ�ⷿ�ݵļ۸�
��h���ǿ����ù�ʽ![����дͼƬ����](http://img.blog.csdn.net/20170117111942867?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)��ʾ��
��Ϊֻ����һ������������������ǳ�����������Ϊ���������Իع����⡣
![����дͼƬ����](http://img.blog.csdn.net/20170117111225347?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

---
### 2.2-2.4���ۺ���
![����дͼƬ����](http://img.blog.csdn.net/20170117121038551?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)��������Ҫ���ı���Ϊ���ǵ�ģ��ѡ����ʵĲ����� parameters�� ��0 �� ��1������ѡ��Ĳ������������ǵõ���ֱ����������ǵ�ѵ������׼ȷ�̶ȣ�ģ����Ԥ���ֵ��ѵ������ʵ��ֵ֮��Ĳ����ǽ�ģ�� modeling error �������ǵ�Ŀ�����ѡ�������ʹ�ý�ģ����ƽ�����ܹ���С��ģ�Ͳ����� ��ʹ�ô��ۺ�����С��
���ǻ���һ����άͼ����������ֱ�Ϊ ��0 �� ��1 �� J(��0,��1)�����Կ�������ά�ռ��д���һ��ʹ�� J(��0,��1)��С�ĵ㡣
![����дͼƬ����](http://img.blog.csdn.net/20170117121219219?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
���ۺ���Ҳ������ƽ������/ƽ�������ۺ���������ع����⻹�������Ĵ��ۺ�����ƽ�������ۺ����ǽ���ع�������õ��ֶΡ�
���ǳ��õȸ���ͼ����άͼ�ĸ���ͼ���������ۺ�����ͨ����ͼ���ǿ�������ؿ������ۺ�������躯���Ĺ�ϵ��![����дͼƬ����](http://img.blog.csdn.net/20170117121304210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
### 2.5-2.6�ݶ��½��㷨
��ʲô��
>����������Сֵ���㷨

Ϊʲô��
>��ʼʱ�������ѡ��һ����������ϣ� ��0,��1,...,��n����������ۺ�����Ȼ������Ѱ����һ�����ô��ۺ���ֵ�½����Ĳ�����ϡ����ǳ�����ô��ֱ������һ���ֲ���Сֵ�� local minimum�� �� ��Ϊ���ǲ�û�г��������еĲ�����ϣ����Բ���ȷ�����ǵõ��ľֲ���Сֵ�Ƿ����ȫ����Сֵ�� global minimum����ѡ��ͬ�ĳ�ʼ������ϣ����ܻ��ҵ���ͬ�ľֲ���Сֵ��

��ô����
>![����дͼƬ����](http://img.blog.csdn.net/20170117123805981?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>
>���Ц���**ѧϰ����**�� learning rate�� �� �������������������ô��ۺ����½��̶����ķ������������Ĳ����ж��
>�������ݶ��½��У�����ÿһ�ζ�**ͬʱ**�����еĲ�����ȥѧϰ���ʳ��Դ��ۺ����ĵ�������**ͬʱ���£�����ͼ����㷨���ʽ��**��

����ģ�⣺
![����дͼƬ����](http://img.blog.csdn.net/20170117125418705?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
��������������� �� ̫С�� �� ̫������ʲô�����
![����дͼƬ����](http://img.blog.csdn.net/20170117124540458?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![����дͼƬ����](http://img.blog.csdn.net/20170117124548469?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
- If �� is too small, gradient descent can be slow.
- If �� is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

������
>���űƽ���͵㣬�½��ٶȱ��Խ��ԽС������**�½��ļ���**�� **��** �� **J�Ԧȵ�ƫ΢��** ��**�˻�**��J�Ԧȵ�ƫ΢�����½��Ĺ����в��ϼ�С��
>���ݶ��½����У� �����ǽӽ��ֲ���͵�ʱ�� **�ݶ��½������Զ���ȡ��С�ķ���**�� ������Ϊ�����ǽӽ��ֲ���͵�ʱ�� ����Ȼ�ھֲ����ʱ���������㣬 ���Ե����ǽӽ��ֲ����ʱ�� ����ֵ���Զ����Խ��ԽС�� �����ݶ��½����Զ���ȡ��С�ķ��ȣ� ������ݶ��½���������

### 2.7�ݶ��½������Իع�
�ݶ��½��㷨�����Իع��㷨��
![����дͼƬ����](http://img.blog.csdn.net/20170117125908351?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
������֮ǰ���õ����Իع����������ݶ��½��㷨���ؼ����ڶԴ��ۺ�����ƫ����
![����дͼƬ����](http://img.blog.csdn.net/20170117130151758?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![����дͼƬ����](http://img.blog.csdn.net/20170117130227665?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
���㷨��дΪ��
![����дͼƬ����](http://img.blog.csdn.net/20170117130406212?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvTmlnaHRtYXJlX2h5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)---------
