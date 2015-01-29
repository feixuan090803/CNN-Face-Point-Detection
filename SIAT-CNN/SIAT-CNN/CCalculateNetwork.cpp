#include "CCalculateNetwork.h"
#include "StdAfx.h"

using namespace std;

CCalculateNetwork::CCalculateNetwork(void)
{
	m_NeuronOutputs.clear();


}


CCalculateNetwork::~CCalculateNetwork(void)
{
}


//完成数据的准备工作，即将图片按序输入神经网络进行检测
void CCalculateNetwork::CalculateNetwork(void)
{
	// runs the current character image through the neural net
	
	// 当用户选择预测默认目录下的人脸图片的特征点时，逐个读入图片输入网络，保存预测结果
	CString strNum;

	// now get image data
	unsigned char grayArray[ g_cImageSize * g_cImageSize] ={0};

	double arrFeaturePoint[10]={0};
	// 本函数将图片像素值保存至 grayAray中,index为对图片的索引值
	//int index=.....;   // 获得当前图片的索引，同时获得特征点的标定位置，五个坐标点位置保存在结构体数组中

	BOOL bTraining = 、、、、、;  // 标记当前是训练还是测试过程
	
	if ( bTraining != FALSE )
	{

		//////////////////////////////////////
		//	  待完成！！！！！！
		//////////////////////////////////////
		GetTrainImageArrayValues(index,grayArray,&arrFeaturePoint);
	}
	else
	{

		//////////////////////////////////////
		//	  待完成！！！！！！
		//////////////////////////////////////
		GetTestImageArrayValues(index,grayArray,&arrFeaturePoint);
	}

	//开始为测试过程计时
	DWORD tick = ::GetTickCount();

	double inputVector[1521];  	 //存储为灰度值归一化的像素数组

	for ( ii=0; ii<g_cImageSize; ++ii )
	{
		for ( jj=0; jj<g_cImageSize; ++jj )
		{
			inputVector[jj + 39*(ii) ] = (double)((int)(unsigned char)grayArray[ jj + g_cImageSize*ii ])/128.0 - 1.0;  // one is white, -one is black
		}
	}

	// get state of "Distort input 
	BOOL bDistort = false;  // 是否需要对图像进行形变 ，默认不需要

	double outputVector[10] = {0.0};
	double targetOutputVector[10] = {0.0};

	// initialize target output vector (i.e., desired values)
	// 保存当前图片的10个特征点标记位置
	for(ii=0; ii<10; ii++)
	{
		targetOutputVector[ii]=arrFeaturePoint;
	}

	/////////////////////////////////////////////////////////
	//
	//
	//  将输入向量输入到神经网络中进行计算，返回计算结果
	//
	/////////////////////////////////////////////////////////
	CalculateNeuralNet( inputVector, 1521, outputVector, 10, &m_NeuronOutputs, bDistort );



}


////////////////////////////////////////////////////////////////////////////
//	引入乱序机制，接着调用m_NN.Calculate(....),
//	m_NN.Calculate(...)以inputvector作为神经网络输入，结果保存到outputVector
//
////////////////////////////////////////////////////////////////////////////
void CCalculateNetwork::CalculateNeuralNet(double *inputVector, int count, 
								   double* outputVector /* =NULL */, int oCount /* =0 */,
								   std::vector< std::vector< double > >* pNeuronOutputs /* =NULL */,
								   BOOL bDistort /* =FALSE */ )
{
	// wrapper function for neural net's Calculate() function, needed because the NN is a protected member
	// waits on the neural net mutex (using the CAutoMutex object, which automatically releases the
	// mutex when it goes out of scope) so as to restrict access to one thread at a time
	
	CAutoMutex tlo( m_utxNeuralNet );
	
	if ( bDistort != FALSE )
	{	
		//引入乱序输入机制
		GenerateDistortionMap();
		ApplyDistortionMap( inputVector );
	}
	
	////////////////////////////////////////////////
	//
	//   调用NeuralNetwork类的神经网络前向计算函数
	//	 m_NN.Calculate ：forward propagation
	////////////////////////////////////////////////
	m_NN.Calculate( inputVector, count, outputVector, oCount, pNeuronOutputs );
	
}
