# python-
这是一个基于暗通道先的图像去雾算法。 思路：暗通道先验+自适应天空分割+区域修正因子+引导滤波
自适应天空分割：
1.预处理；输入：灰度图；输出：边缘图像。天空图像灰度图（保留亮度信息）+高斯滤波降噪+canny边缘（高低阈值根据图像灰度中位数动态设定，适应不同图像）+膨胀边缘图（让零散边缘连成小区域，便于统计局部边缘密度）
2.天空先验：输入：灰度图；输出：天空阈值+粗糙的天空区域。基于亮度的迭代阈值（先取全图的亮度均值作为初始值；大于该亮度的判断为天空，小于就是前景；再取两者平均值作为新阈值，两个阈值相差大小小于规定值就收敛）
3.逐列扫描自适应确定天空边界：输入：灰度图，边缘图，先验图；输出：最终天空图。先设定stop_y=-1，每一列最多扫描高度0.85y，对每一列第一个开始判断是否为天空先验，再到灰度图和边缘图里面取窗口，判断窗口里面的边缘密度、灰度方差、灰度变化是否符合天空特性。如果超过阈值就会将stop_y设定为y-1，跳出循环。那么y之上的部分就是天空。最后加上掩模图片，用红色标记出天空区域。在暗通道该区域计算大气光值A
区域修正因子：
原始透射率在高亮度地区估算偏小以及边缘突变区域透射率计算偏大，所以加入修正系数：将亮度因子与边缘因子加权线性组合k(x)=α⋅I_gray(x)−β⋅G_norm(x)。修正透射率:高亮度或强边缘区域的透射率适度调整t_mod(x)=t(x)⋅(1+0.5⋅k(x))。

参考资料：
https://blog.csdn.net/caobin_cumt/article/details/134256376?fromshare=blogdetail&sharetype=blogdetail&sharerId=134256376&sharerefer=PC&sharesource=2301_78055143&sharefrom=from_link

图像处理基础（二）暗通道先验去雾 - 山与水你和我的文章 - 知乎https://zhuanlan.zhihu.com/p/440903916

He K M, Sun J, Tang X. Guided image filtering[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2012, 35(6): 1397-1409.

He K M, Sun J, Tang X O. Single image haze removal using dark channel prior ［J］. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011, 33(12): 2341-2353.
