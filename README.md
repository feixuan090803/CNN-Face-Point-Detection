## CNN-Face-Point-Detection
This project is about the utilization of CNN to detect human face point. Trained with the dataset LFW and images from Internet.

----------------
## 2015/9/23  增补

这个CNN人脸配准系统是基于C++写的，核心模块都是自个实现的，代码参考的是一个开源的手写体识别CNN程序：
         http://www.codeproject.com/Articles/16650/Neural-Network-for-Recognition-of-Handwritten-Digi 
论文参考的是：
         http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm

在调试多线程模块的时候一直有错，暂时就把多线程实现的部分注释掉了。故实际训练时是串行训练的，速度可想而知。
在博客 http://blog.csdn.net/reporter521/article/details/45567555 里面也说了，纯属抱着学习的心态。 :)
代码中的注释写的也比较完整，适合对CNN底层实现机制感兴趣的小伙伴。
 
 ---
## 系统实现的模块大致分为：
1、卷积网络各基础部件模块，包括网络层类，神经元类，网络连接类等，各类别定义了各自的成员变量与成员函数，
   详见：NeuralNetwork.h/cpp；
   
2、网络参数配置模块，对网络训练参数初始化配置，比如学习速率，收敛停止条件等，参数设置以配置文件形式进行读取与修改，
   详见：Preferences.h/cpp；
   
3、网络构建模块，包括对九层网络的逐层构建，层与层之间的连接，参数的初始化等，详见：CCreateNetwork.h/cpp；

4、网络前向/后向模块，前向计算进行网络计算，后向传播进行网络训练，在训练过程中经过一定次数时将网络权值自动保存为本地文件，
  详见：CCalculateNetwork.h/cpp；
  
5、系统逻辑模块，实现命令行的交互实现，若选择进行网络预测，询问是否需要载入已训练好的网络；选择网络训练，询问是重头开始训练还   是载入已有些参数权重，接着进行训练，详见：main.cpp。

   
