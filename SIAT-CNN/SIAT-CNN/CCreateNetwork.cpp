
#include "StdAfx.h"
#include "CCreateNetwork.h"
#include "Preferences.h"
#include <afxwin.h>
#include <highgui.h>
#include <imgproc\imgproc.hpp>
#include <cv.h>
#include <afx.h>

#include "SHLWAPI.H"	// for the path functions
#pragma comment( lib, "shlwapi.lib" )

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )
#define UNIFORM_ZERO_THRU_ONE ( (double)(rand())/(RAND_MAX + 1 ) )

using namespace std;

CCreateNetwork::CCreateNetwork(void)
{
	// TODO: add one-time construction code here
	m_cBackprops = 0;
	m_nAfterEveryNBackprops = 1;

	
	m_iNextTestingPattern = 0;
	m_iNextTrainingPattern = 0;
	
	// allocate memory to store the distortion maps
	m_cCols = 39;
	m_cRows = 39;
	
	m_cCount = m_cCols * m_cRows;

	m_DispH = new double[ m_cCount ];
	m_DispV = new double[ m_cCount ];

	// clear m_NeuronOutputs
	m_NeuronOutputs.clear();	

	//训练与测试图片的默认路径
	m_TrainFilePath="E:\\人脸识别\\face point detection\\trainingImage\\lfw_5590n\\";
	m_TestFilePath="E:\\人脸识别\\face point detection\\trainingImage\\lfw_5590n\\";

	m_TrainInfoPath="E:\\人脸识别\\face point detection\\trainingImage\\trainInfo\\*.txt";

	m_sWeightSavePath= "E:\\人脸识别\\face point detection\\weight";

}

CCreateNetwork::~CCreateNetwork(void)
{
}

vector<CString> CCreateNetwork::m_InfoFileNames;  
vector<CString> CCreateNetwork::m_InfoFilePaths;
vector<CString> CCreateNetwork::m_ImageFilePaths;


// 构建用于人脸特征点检测的九层网络
BOOL CCreateNetwork::InitNetwork(bool bLoadWeightFile)
{
	// grab the mutex for the neural network

	//****CAutoMutex tlo(m_utxNeuralNet);


	////////////////////////////////////////////////
	//   权重预读取模块：
	//		读取保存有训练好的各层网络权重的文件
	///////////////////////////////////////////////
	//定义列表用来存储各层的初始权值
	vector<double>* vWeightOfLayer1=new vector<double>();  
	vector<double>* vWeightOfLayer3=new vector<double>();  
	vector<double>* vWeightOfLayer5=new vector<double>();  
	vector<double>* vWeightOfLayer7=new vector<double>();  
	vector<double>* vWeightOfLayer8=new vector<double>();  
	vector<double>* vWeightOfLayer9=new vector<double>(); 

	//  首先判断用户需要载入权重文件,若是则预先打开文件并得到各层的权重值序列
	if(bLoadWeightFile == true)  //bLoadWeightFile表征进行的为测试操作
	{
		cout<<endl<<"加载权重文件中>>> >>>"<<endl;

		char path[100];
		
		sprintf(path,"%s\\1.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer1,path);
		
		sprintf(path,"%s\\3.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer3,path);

		sprintf(path,"%s\\5.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer5,path);

		sprintf(path,"%s\\7.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer7,path);

		sprintf(path,"%s\\8.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer8,path);

		sprintf(path,"%s\\9.txt",m_sWeightSavePath.c_str());
		this->LoadWeightFile(vWeightOfLayer9,path);

		cout<<"加载完毕！！！"<<endl<<endl;
	}


	cout<<"正在初始化网络>>> >>>"<<endl;

	// initialize and build the neural net
	NeuralNetwork& NN = m_NN;  // for easier nomenclature
	NN.Initialize();
	
	NNLayer* pLayer;
	
	int ii, jj, kk;
	int icNeurons = 0;
	int icWeights = 0;
	double initWeight;
	CString label;

	// layer zero, the input layer.
	// Create neurons: exactly the same number of neurons as the input
	// vector of 39x39=1521 pixels, and no weights/connections
	
	pLayer = new NNLayer( _T("Layer00") );
	NN.m_Layers.push_back( pLayer );
	
	for (ii=0; ii<1521; ++ii)
	{
		label.Format(_T("Layer00_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	// layer one:
	// This layer is a convolutional layer that has 20 feature maps.  Each feature 
	// map is 36x36, and each unit in the feature maps is a 5x5 convolutional kernel
	// of the input layer.
	// So, there are 36x36x20 = 25920 neurons, (4x4+1)x20 = 340 weights
	
	pLayer = new NNLayer( _T("Layer01"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( ii=0; ii<25920; ++ii )
	{
		label.Format( _T("Layer01_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( ii=0; ii<340; ++ii )
	{
		label.Format( _T("Layer01_Weight%04d_Num%06d"), ii, icWeights );

		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer1->size() != 340)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer1)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// interconnections with previous layer: this is difficult
	// The previous layer is a top-down bitmap image that has been padded to size 29x29
	// Each neuron in this layer is connected to a 5x5 kernel in its feature map, which 
	// is also a top-down bitmap of size 13x13.  We move the kernel by TWO pixels, i.e., we
	// skip every other pixel in the input image
	
	int kernelTemplate[16] = {
								0,  1,  2,  3, 
								39, 40, 41, 42,                                                                                                                                                                      
								78, 79, 80, 81, 
								117, 118, 119, 120 };
		
	int iNumWeight=0;
		
	int fm=0;
	for (fm=0; fm<20; ++fm)
	{
		for ( ii=0; ii<36; ++ii )
		{
			for ( jj=0; jj<36; ++jj )
			{
				iNumWeight = fm * 17;  // 17 is the number of weights per feature map
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*36 + fm*1296 ] );
				
				n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
				
				for ( kk=0; kk<16; ++kk )
				{
					// note: max val of index == 840, corresponding to 841 neurons in prev layer
					n.AddConnection( 1*jj + 39*ii + kernelTemplate[kk], iNumWeight++ );
				}
			}
		}
	}
	
	
	// layer two:
	// This layer is a Pooling layer that has 20 feature maps.  Each feature 
	// map is 18x18, and each unit in the feature maps is a 2x2 Pooling kernel
	// of corresponding areas of all 20 of the previous layers, each of which is a 36x36 feature map
	// So, there are 18x18x20 = 6480 neurons, 36x36x20 = 25920 weights

	pLayer = new NNLayer( _T("Layer02"), pLayer );
	NN.m_Layers.push_back( pLayer );

	for ( ii=0; ii<6480; ++ii )
	{
		label.Format( _T("Layer02_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	//for ( ii=0; ii<25920; ++ii )
	for ( ii=0; ii<80; ++ii )
	{
		label.Format( _T("Layer02_Weight%04d_Num%06d"), ii, icWeights );
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, 0.25 ) );  //pooling层的神经元权值为0.25且为常数
	}

	
	int kernelTemplate2[4] = {
		0,  1,  
		36, 37 };


	for (fm=0; fm<20; ++fm)
	{
		for ( ii=0; ii<18; ++ii )			
		{
			for ( jj=0; jj<18; ++jj )
			{
				iNumWeight = fm * 4;  // 4 is the number of weights per feature map
					
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*18 + fm*324 ] );

				//n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight

				for ( kk=0; kk<4; ++kk )
				{
					// note: max val of index == 1013, corresponding to 1014 neurons in prev layer
					n.AddConnection( 1296*fm + 2*jj + 72*ii + kernelTemplate2[kk], iNumWeight++ );
				}
			}
		}
	}


	// layer three:
	// This layer is a convolutional layer that has 40 feature maps.  Each feature 
	// map is 16x16, and each unit in the feature maps is a 3x3 convolutional kernel
	// of corresponding areas of all 20 of the previous layers, each of which is a 18x18 feature map
	// So, there are 16x16x40 = 10240 neurons, (3x3+1)x20x40 = 8000 weights

	pLayer = new NNLayer( _T("Layer03"), pLayer );
	NN.m_Layers.push_back( pLayer );

	for ( ii=0; ii<10240; ++ii )
	{
		label.Format( _T("Layer03_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}

	for (ii=0; ii<8000; ++ii)
	{
		label.Format( _T("Layer03_Weight%04d_Num%06d"), ii, icWeights );
		
		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer3->size() != 8000)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer3)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}

		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}

	int kernelTemplate3[9] = 
	{
		0,  1,  2, 
		18, 19, 20,
		36, 37, 38    
	};


	for (fm=0; fm<40; ++fm)
	{
		for ( ii=0; ii<16; ++ii )
		{
			for ( jj=0; jj<16; ++jj )
			{
				iNumWeight = fm * 10;  // 5 is the number of weights per feature map
						
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*16 + fm*256 ] );

				// 每一个神经元只需要加上一个偏置用于对前一层输出累加结果的调控，以加快收敛速度
				n.AddConnection(ULONG_MAX, iNumWeight++);  // bias weight

				int mm;
				for (kk=0; kk<9; ++kk)
				{
					for(mm=0;mm<20;mm++)
					{
						n.AddConnection(324*mm + jj + 18*ii + kernelTemplate3[kk], iNumWeight++ );
					}
				}

			}
		}
	}

	// layer four:
	// Pooling layer:
	// This layer is a  Pooling layer that has 50 feature maps.  Each feature 
	// map is 8x8, and each unit in the feature maps is a 2x2  Pooling kernel
	// of corresponding areas of the previous layers, each of which is a 16x16 feature map
	// So, there are 8x8x40 = 2560 neurons, 2x2x2560 =10240 weights

	pLayer = new NNLayer( _T("Layer04"), pLayer );
	NN.m_Layers.push_back( pLayer );

	for ( ii=0; ii<2560; ++ii )
	{
		label.Format( _T("Layer04_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}

	// pooling层的所有连接权值是否都为1 ？？？？如果是的话那是否pooling层的权值数组长度便为1，且值为1？？？
	for ( ii=0; ii<160; ++ii )
	{
		label.Format( _T("Layer04_Weight%04d_Num%06d"), ii, icWeights );
		//initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, 0.25 ) );
	}

		
	int kernelTemplate4[4] = {
		0,  1, 
		16, 17  };


	for ( fm=0; fm<40; ++fm)
	{
		for ( ii=0; ii<8; ++ii )
		{
			for ( jj=0; jj<8; ++jj )
			{
				iNumWeight = fm * 4;  // 10 is the number of weights per feature map
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*8 + fm*64 ] );

				//n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight

				for ( kk=0; kk<4; ++kk )
				{	
					n.AddConnection( 256*fm + 2*jj + 32*ii + kernelTemplate4[kk], iNumWeight++ );
				}
			}
		}
	}
			
	
	// layer five:
	// This layer is a convolutional layer that has 60 feature maps.  Each feature 
	// map is 6x6, and each unit in the feature maps is a 2x2 convolutional kernel
	// of corresponding areas of all 40 of the previous layers, each of which is a 6x6 feature map
	// So, there are 6x6x60 = 2180 neurons, (3x3+1)x40x60 = 24000 weights

	pLayer = new NNLayer( _T("Layer05"), pLayer );
	NN.m_Layers.push_back( pLayer );

	//**** for ( ii=0; ii<1980; ++ii )			
	for ( ii=0; ii<2160; ++ii )
	{
		label.Format( _T("Layer05_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}

	for ( ii=0; ii<24000; ++ii )
	{
		label.Format( _T("Layer05_Weight%04d_Num%06d"), ii, icWeights );
		
		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer5->size() != 24000)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer5)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}

		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}


	int kernelTemplate5[9] = 
	{
		0,  1,  3,  
		8,  9, 10,
		16, 17, 18 
	};

	for ( fm=0; fm<60; ++fm)
	{
		for ( ii=0; ii<6; ++ii )
		{
			for ( jj=0; jj<6; ++jj )
			{
				iNumWeight = fm * 10;  // 10 is the number of weights per feature map
							
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*6 + fm*36 ] );  //获得目标神经元在保存本层所有神经元的数组中的索引
				n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
							
				int mm;
				for ( kk=0; kk<9; ++kk )
				{
					for( mm=0;mm<40;mm++)
					{
						n.AddConnection(64*mm + jj + 8*ii + kernelTemplate5[kk], iNumWeight++ );
					}
				}
				//每一个神经元只需要加上一个偏置用于对前一层输出累加结果的调控，以加快收敛速度
			}
		}
	}



	// layer six:
	// Pooling layer:
	// This layer is a  Pooling layer that has 60 feature maps.  Each feature 
	// map is 3x3, and each unit in the feature maps is a 2x2  Pooling kernel
	// of corresponding areas of the previous layers, each of which is a 16x16 feature map
	// So, there are 3x3x60 = 540 neurons, 2x2x540 =2160 weights

	pLayer = new NNLayer( _T("Layer06"), pLayer );
	NN.m_Layers.push_back( pLayer );

	for ( ii=0; ii<540; ++ii )
	{
		label.Format( _T("Layer06_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}

	for ( ii=0; ii<240; ++ii )
	{
		label.Format( _T("Layer06_Weight%04d_Num%06d"), ii, icWeights );
		//initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, 0.25 ) );
	}

	// Interconnections with previous layer: this is difficult
	// Each feature map in the previous layer is a top-down bitmap image whose size
	// is 13x13, and there are 6 such feature maps.  Each neuron in one 5x5 feature map of this 
	// layer is connected to a 5x5 kernel positioned correspondingly in all 6 parent
	// feature maps, and there are individual weights for the six different 5x5 kernels.  As
	// before, we move the kernel by TWO pixels, i.e., we
	// skip every other pixel in the input image.  The result is 50 different 5x5 top-down bitmap
	// feature maps

	int kernelTemplate6[4] = 
	{
		0, 1, 
		6, 7  
	};


	for ( fm=0; fm<60; ++fm)
	{
		for ( ii=0; ii<3; ++ii )
		{
			for ( jj=0; jj<3; ++jj )
			{
				iNumWeight = fm * 4;  // 10 is the number of weights per feature map
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*3 + fm*9 ] );

				//n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight

				for ( kk=0; kk<4; ++kk )
				{
					//n.AddConnection(  2*jj + 6*ii + kernelTemplate4[kk], iNumWeight++ );
					n.AddConnection(36*fm + 2*jj + 12*ii + kernelTemplate6[kk], iNumWeight++ );

				}
			}
		}
	}


	// layer seven:
	// This layer is a convolutional layer that has 80 feature maps.  Each feature 
	// map is 2x2, and each unit in the feature maps is a 2x2 convolutional kernel
	// of corresponding areas of all 60 of the previous layers, each of which is a 3x3 feature map
	// So, there are 2x2x80 = 320 neurons, (2x2+1)x60x80 = 24000 weights

	pLayer = new NNLayer( _T("Layer07"), pLayer );
	NN.m_Layers.push_back( pLayer );

	for ( ii=0; ii<320; ++ii )
	{
		label.Format( _T("Layer07_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}

	for ( ii=0; ii<24000; ++ii )
	{
		label.Format( _T("Layer07_Weight%04d_Num%06d"), ii, icWeights );
		
		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer7->size() != 24000)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer7)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}


		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
		//pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, 1 ) );
	}


	int kernelTemplate7[4] = {
		0,  1, 
		3, 4   };


	for ( fm=0; fm<80; ++fm)
	{
		for ( ii=0; ii<2; ++ii )
		{
			for ( jj=0; jj<2; ++jj )
			{
				iNumWeight = fm * 5;  //5 is the number of weights per feature map
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*2 + fm*4 ] );

				n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight

				int mm;
				for ( kk=0; kk<4; ++kk )
				{
					for( mm=0;mm<60;mm++)
					{
						n.AddConnection(9*mm + jj + 3*ii + kernelTemplate7[kk], iNumWeight++ );
					}
				}

			}
		}
	}

	// layer eight:
	// This layer is a fully-connected layer with 100 units.  Since it is fully-connected,
	// each of the 120 neurons in the layer is connected to all 320 neurons in
	// the previous layer.
	// So, there are 120 neurons and 120*(320+1)=38520 weights
	
	pLayer = new NNLayer( _T("Layer08"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( ii=0; ii<120; ++ii )
	{
		label.Format( _T("Layer08_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( ii=0; ii<38520; ++ii )
	{
		label.Format( _T("Layer08_Weight%04d_Num%06d"), ii, icWeights );
		
		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer8->size() != 38520)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer8)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}

		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// Interconnections with previous layer: fully-connected
	
	iNumWeight = 0;  // weights are not shared in this layer
	
	for ( fm=0; fm<120; ++fm )
	{
		NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
		n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
		
		for ( ii=0; ii<320; ++ii )
		{
			n.AddConnection( ii, iNumWeight++ );
		}
	}
	
	// layer nine, the final (output) layer:
	// This layer is a fully-connected layer with 10 units.  Since it is fully-connected,
	// each of the 10 neurons in the layer is connected to all 120 neurons in
	// the previous layer.
	// So, there are 10 neurons and 10*(120+1)=1210 weights
	
	pLayer = new NNLayer( _T("Layer09"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( ii=0; ii<10; ++ii )
	{
		label.Format( _T("Layer09_Neuron%04d_Num%06d"), ii, icNeurons ); 
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( ii=0; ii<1210; ++ii )
	{
		label.Format( _T("Layer09_Weight%04d_Num%06d"), ii, icWeights );
		
		if(bLoadWeightFile == true)  //是否设置训练好的权值
		{
			if(vWeightOfLayer9->size() != 1210)  //若读取到权值文件中的权重数不符则采用随机化权重
			{
				cout<<"权值数目不相符，改为随机化初始权重"<<endl;
				initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
			}
			else
			{
				initWeight = (*vWeightOfLayer9)[ii];
			}
		}
		else
		{
			initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		}

		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// Interconnections with previous layer: fully-connected
	
	iNumWeight = 0;  // weights are not shared in this layer
	
	for ( fm=0; fm<10; ++fm )
	{
		NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
		n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
		
		for ( ii=0; ii<120; ++ii )
		{
			n.AddConnection( ii, iNumWeight++ );
		}
	}
	
	//*******
	//SetModifiedFlag( TRUE );
	cout<<"初始化完成！！！"<<endl<<endl;

	
	return TRUE;
}


// 读取训练/测试的图片，准备网络的输入向量
bool CCreateNetwork::ForwardPropagation(void)
{
	 // runs the current character image through the neural net
	
	// 当用户选择预测默认目录下的人脸图片的特征点时，逐个读入图片输入网络，保存预测结果
	CString strNum;

	// now get image data
	unsigned char grayArray[ g_cImageSize * g_cImageSize] ={0};
															
	double arrFeaturePoint[10]={0};
	
	int index=0;   // 获得当前图片在目录搜索结果中的索引，同时获得特征点的标定位置，五个坐标点位置保存在结构体数组中

	double outputVector[10] = {0.0};
	double targetOutputVector[10] = {0.0};
	
	//选择测试集图片进行检验还是训练集图片进行检验，可进行多次测试
	bool isContinueTest=true;
	while(isContinueTest)
	{
		cout<<"请选择需要进行特征点检测的人脸图片... ..."<<endl;
		//////////////////////////////////////
		//***	  测试过程，打开文件选择框选择进行检测的图片
		//////////////////////////////////////
		AfxSetResourceHandle(GetModuleHandle(NULL));
		CFileDialog fdTest(true,NULL,NULL,OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT | OFN_EXPLORER);
		fdTest.m_ofn.lpstrFilter="Test Image (*.jpg)|*.jpg;*.JPG|All Files (*.*)|*.*||";
		fdTest.m_ofn.lpstrTitle="Test Images";
		fdTest.m_ofn.lpstrInitialDir=this->m_TestFilePath;

		if(fdTest.DoModal() != IDOK)
		{
			//this->m_fileTestingLabels.Close();
			//this->m_fileTestingImages.Close();

			return false;
		}
		

		CFileException fe;
		CString csFilePath=fdTest.GetPathName();
		CString	csFileName=fdTest.GetFileName();
		CString csInfoPath;
		CString csPoint;
		
		cout<<"完成选择： "<<csFileName<<endl;

		cout<<endl<<"正在进行人脸特征点检测>>> >>>"<<endl;

		if(m_fileTestingImages.Open((LPCTSTR)csFilePath,CFile::modeRead|CFile::shareDenyNone,&fe) != 0)
		{
			m_fileTestingImages.Read(grayArray,g_cImageSize*g_cImageSize); 

			////////////////////////////////////////////////////////////////////
			//根据选定的人脸图片到标签文件夹读取对应的标签信息，初始化特征点数组
			////////////////////////////////////////////////////////////////////
			
			//提取字符串型的特征点序列
			csFileName=csFileName.SpanExcluding(".")+".txt";
			csInfoPath.Format("E:\\人脸识别\\face point detection\\trainingImage\\ttInfo\\%s",csFileName);
			
			if(m_fileTestingLabels.Open((LPCTSTR)csInfoPath,CFile::modeRead) != 0)
			{
				m_fileTestingLabels.ReadString(csPoint);
			}

			//逐个解析特征点并存储
			int index;
			int i=0;
			CString csTemp;
			string sTemp;
			while((index=csPoint.Find(" "))!=-1)
			{
				csTemp=csPoint.SpanExcluding(" ").GetString();
				sTemp=csTemp.GetString();
				targetOutputVector[i]=atof(sTemp.c_str())/39;
				csPoint.Delete(0,index+1);

				if(i<9)
				{
					i++;
				}
				else
				{
					i=0;
					break;
				}
			}
			sTemp=csPoint.GetString();
			targetOutputVector[9]=atof(sTemp.c_str())/39;

		}

		m_fileTestingImages.Close();
		m_fileTestingLabels.Close();

		//开始为测试过程计时
		DWORD tick = ::GetTickCount();

		double inputVector[1521];  	 //存储为灰度值归一化的像素数组

		int ii,jj;

		//对输入向量进行归一化
		for ( ii=0; ii<g_cImageSize; ++ii )
		{
			for ( jj=0; jj<g_cImageSize; ++jj )
			{
				//inputVector[jj + 39*(ii) ] = (double)((int)(unsigned char)grayArray[ jj + g_cImageSize*ii ])/128.0 - 1.0;  // one is white, -one is black
				inputVector[jj + 39*(ii) ] = (double)((int)(unsigned char)grayArray[ jj + g_cImageSize*ii ])/255.0;  // one is white, -one is black
			}
		}

		// get state of "Distort input 
		BOOL bDistort = false;  // 是否需要对图像进行形变 ，默认不需要

		/////////////////////////////////////////////////////////
		//
		//
		//  将输入向量输入到神经网络中进行计算，返回并输出计算结果到控制台
		//
		/////////////////////////////////////////////////////////
		CalculateNeuralNet( inputVector, 1521, outputVector, 10, &m_NeuronOutputs, bDistort );

		//获得计算花费的时间，单位为毫秒
		DWORD diff = ::GetTickCount() - tick;

		CString	 strLine,strResult;
		double dTemp, sampleMse = 0.0;
	
		cout<<endl;
		cout<<"*************************************************"<<endl;
		strResult.Format( _T("Face point detected results:\n") );
	
		for ( ii=0; ii<10; ii++ )
		{
			strLine.Format( _T(" %2i = %+6.3f \n"), ii, outputVector[ii] );
			strResult += strLine;
		
			dTemp = targetOutputVector[ ii ] - outputVector[ ii ];
			sampleMse += dTemp * dTemp;
		}

		sampleMse = 0.5 * sampleMse;
		strLine.Format( _T("\n检测平均误差:\n Ep = %g\n\n耗时:\n %i mSecs"), sampleMse, diff);
		strResult += strLine;

		//命令行输出当前图像的检测结果与检测误差
		cout<<strResult<<endl;
		cout<<"*************************************************"<<endl;

		//询问是否需要实时展示检测效果
		cout<<endl<<"是否需要特征点检测效果？[Y(yes) or N(no)]:  ";
		char cSelected;
		cin>>cSelected;
		if(cSelected == 'Y' || cSelected == 'y')
		{
			CString csImageFilePath=csFilePath.SpanExcluding(".")+".jpg";
			this->DisplayPointDetected(csImageFilePath,outputVector);
		}

		cout<<endl<<"是否需要继续测试？[Y(yes) or N(no)]:  ";
		cin>>cSelected;
		if(cSelected != 'Y' && cSelected != 'y')
		{
			isContinueTest=false;
		}
	}
	return true;
}

void CCreateNetwork::CalculateNeuralNet(double *inputVector, int count, 
								        double* outputVector /* =NULL */, int oCount /* =0 */,
										std::vector< std::vector< double > >* pNeuronOutputs /* =NULL */,
										BOOL bDistort /* =FALSE */ )
{
	// wrapper function for neural net's Calculate() function, needed because the NN is a protected member
	// waits on the neural net mutex (using the CAutoMutex object, which automatically releases the
	// mutex when it goes out of scope) so as to restrict access to one thread at a time
	
	//***** CAutoMutex tlo( m_utxNeuralNet );
	
	if ( bDistort != FALSE )
	{	
		//************！！！！！引入乱序输入机制 ，在本系统中可能用不到
		//GenerateDistortionMap();
		//ApplyDistortionMap( inputVector );
	}
	
	////////////////////////////////////////////////
	//
	//   调用NeuralNetwork类的神经网络前向计算函数
	//	 m_NN.Calculate ：forward propagation
	////////////////////////////////////////////////
	m_NN.Calculate( inputVector, count, outputVector, oCount, pNeuronOutputs );
}


/////////////////////////////////////////////////////////////////////////////////
//
//    BP操作步骤一： 初始化BP过程的相关参数，以及训练实时信息的控制的输出
//                   调用 CCreateNetwork::StartBackpropagation进行具体操作
//
/////////////////////////////////////////////////////////////////////////////////

void CCreateNetwork::BackPropagation(void)
{
	CPreferences* pfs=CPreferences::GetPreferences();

//	this->m_cNumThreads=pfs->m_cNumBackpropThreads;	   //初始化系统可以开启的BP线程数上限
	this->m_InitialEta=pfs->m_dInitialEtaLearningRate;  //初始化训练的学习速率
	this->m_MinimumEta=pfs->m_dMinimumEtaLearningRate;  //初始化学习速率的下限值
	this->m_EtaDecay=pfs->m_dLearningRateDecay;		   //初始化学习速率的下降速度
	this->m_AfterEvery=pfs->m_nAfterEveryNBackprops;	   //设定需要每个epoch进行的BP次数，大小达到时则调整学习速率
	this->m_StartingPattern=0;						   //设定首次进行训练的图片在数据集中的索引
	this->m_EstimatedCurrentMSE=0.10;
	
	//this->m_bDistortPatterns=TRUE;					   //是否需要对输入图像序列进行乱序处理
	this->m_bDistortPatterns=FALSE;					   //是否需要对输入图像序列进行乱序处理

	//输出当前网络参数的学习速率以及当前正输入网络参与训练的图片标识
	cout<<endl<<"******************************************************"<<endl;
	CString strInitEta;
	strInitEta.Format("Initial Learning Rate eta (currently, eta = %11.8f)\n", GetCurrentEta());
	cout<<strInitEta;
	cout<<endl<<"******************************************************"<<endl;

	///////////////////////////////////////////////////
	//
	//	在BP开始前，获取参与网络训练的文件目录下的训练文件信息
	//
	//////////////////////////////////////////////////
	CFileFind finder;
	CString filename;
	CString filepath;

	m_InfoFileNames.clear();
	m_InfoFilePaths.clear();
	m_ImageFilePaths.clear();
	
	BOOL isWorking=finder.FindFile(m_TrainInfoPath);
	while(isWorking)
	{
		isWorking=finder.FindNextFileA();
		filename=finder.GetFileName();
		filepath=finder.GetFilePath();

		m_InfoFileNames.push_back(filename);
		m_InfoFilePaths.push_back(filepath);

		//将对应图片的路径也保存下来
		this->m_TrainImagePath=m_TrainFilePath+filename.SpanExcluding(".")+".jpg";
		m_ImageFilePaths.push_back(m_TrainImagePath);
	}

	//////////////////////////////////////////////////
	////
	////   核心处理：m_pDoc->StartBackpropagation
	//////////////////////////////////////////////////
	BOOL bRet= this->StartBackpropagation( m_StartingPattern,
			m_InitialEta, m_MinimumEta, m_EtaDecay, m_AfterEvery, 
			m_bDistortPatterns, m_EstimatedCurrentMSE );
	if(bRet !=FALSE)
	{
		//m_iEpochsCompleted = 0;
		//m_iBackpropsPosted = 0;
		//m_dMSE = 0.0;

		////m_cMisrecognitions = 0;

		//m_dwEpochStartTime = ::GetTickCount();

		////控制台输出：BP周期完成的数目
		//CString str;
		//str.Format( _T("%d Epochs completed\n"), m_iEpochsCompleted );
		//cout<<str;
		
		//cout<<"Backpropagation started... \n";
	}
}


////////////////////////////////////////////////////////
//
//   BP操作步骤二： 
//	   输入参数：由BP初始化模块传送过来的BP相关参数值
//       1)准备当前BP的输入向量,以及定义输出向量，为调用BP操作做准备；
//		 2)调用实际的BP计算函数：pThis->BackpropagateNeuralNet
//		 3)调用本函数执行一次后返回，计算MSE，判断是否需要下一次的BP
//
////////////////////////////////////////////////////////
BOOL CCreateNetwork::StartBackpropagation(UINT iStartPattern /* =0 */, double initialEta /* =0.005 */, 
									 double minimumEta /* =0.000001 */, double etaDecay /* =0.990 */, 
									 UINT nAfterEvery  /* =1000 */, BOOL bDistortPatterns /* =TRUE */, double estimatedCurrentMSE /* =1.0 */)
{
	//测试当前是否已经有BP线程在运行，有则返回FALSE（只允许一个BP操作存在，但BP操作过程中支持多线程）
	/*if ( m_bBackpropThreadsAreRunning == TRUE )   
		return FALSE;
	*/

//	m_bBackpropThreadAbortFlag = FALSE;
//	m_bBackpropThreadsAreRunning = TRUE;
//	m_iNumBackpropThreadsRunning = 0;   //BP线程计数
//	m_iBackpropThreadIdentifier = 0;
	m_cBackprops = iStartPattern;   //指示用于训练的字符图像索引号
	
	//m_bNeedHessian = TRUE;          //指示是否加上hessian矩阵，即是否采用二次收敛机制

	m_iNextTrainingPattern = iStartPattern;

	if ( m_iNextTrainingPattern < 0 ) 
		m_iNextTrainingPattern = 0;
	if ( m_iNextTrainingPattern >= CPreferences::GetPreferences()->m_nItemsTrainingImages )   //m_nItemsTrainingImages=10000
		m_iNextTrainingPattern = CPreferences::GetPreferences()->m_nItemsTrainingImages - 1;
	
//	if ( iNumThreads < 1 ) 
//		iNumThreads = 1;
//	if ( iNumThreads > 10 )  // 10 is arbitrary upper limit
//		iNumThreads = 10;
	
	m_NN.m_etaLearningRate = initialEta;
	m_NN.m_etaLearningRatePrevious = initialEta;
	m_dMinimumEta = minimumEta;
	m_dEtaDecay = etaDecay;
	m_nAfterEveryNBackprops = nAfterEvery;
	m_bDistortTrainingPatterns = bDistortPatterns;

	//// MSE价值衡量：用于判断是否需要计算下一次的backpropagation
	m_dEstimatedCurrentMSE = estimatedCurrentMSE;  // estimated number that will define whether a forward calculation's error is significant enough to warrant backpropagation

	///////////////////////////////////
	///
	///  开始BP操作
	///////////////////////////////////
	CCreateNetwork* pThis = this;
	
	ASSERT( pThis != NULL );
	
	// do the work
	
	double inputVector[1521] = {0.0};  // note: 29x29, not 28x28
	double targetOutputVector[10] = {0.0};
	double actualOutputVector[10] = {0.0};
	//double dMSE;
	//UINT scaledMSE;

	unsigned char grayLevels[g_cImageSize * g_cImageSize]; //用于保存获取的pattern图像灰度值，大小为39*39
	
	int label = 0;
	int ii, jj,index;
	UINT iSequentialNum;
	
	std::vector< std::vector< double > > memorizedNeuronOutputs;  
	
	bool isNeedQuitBP=FALSE;
	while ( isNeedQuitBP == FALSE )  //检测当前线程是否被终止
	{
		//依次取下一个用于训练的pattern，将pattern图像的灰度值序列保存到数组grayLevels，共1521个像素点，并将特征点坐标位置保存到targetOutputVector[10]
		memset(grayLevels,0,39*39);

		int iRet = pThis->GetNextTrainingPattern( grayLevels, targetOutputVector, FALSE, FALSE, &iSequentialNum );
		
		double grayValue;
		for ( ii=0; ii<g_cImageSize; ++ii )
		{
			for ( jj=0; jj<g_cImageSize; ++jj )
			{
				grayValue=(double)((int)(unsigned char)grayLevels[ jj + g_cImageSize*ii ]);
				//inputVector[jj + 39*(ii) ] = grayValue/128.0 - 1.0;  // one is white, -one is black
				inputVector[jj + 39*(ii) ] = grayValue/255.0;  
			}
		}

		///！！！还需对理想输出向量targetOutputVector[10]进行初始化 ！！！
		CString csInfoPath=m_InfoFilePaths[iSequentialNum];
		CString csPoint;
		CString csTemp;
		string sTemp;
		int i=0;

		CStdioFile m_filePoint;
		m_filePoint.Open(csInfoPath,CFile::modeRead);
		m_filePoint.ReadString(csPoint);

		while((index=csPoint.Find(" "))!=-1)
		{
			csTemp=csPoint.SpanExcluding(" ").GetString();
			sTemp=csTemp.GetString();
			targetOutputVector[i]=atof(sTemp.c_str())/39;  //归一化处理
			csPoint.Delete(0,index+1);

			if(i<9)
			{
				i++;
			}
			else
			{
				i=0;
				break;
			}
		}
		sTemp=csPoint.GetString();
		targetOutputVector[9]=atof(sTemp.c_str())/39;  //归一化处理
		m_filePoint.Close();
		
		/////////////////////////////////////////////////////////////////////
		//	BP需要的输入向量以及相关参数已准备好，开始BP实际操作
		/////////////////////////////////////////////////////////////////////
		bool rt=pThis->BackpropagateNeuralNet( inputVector, 1521, targetOutputVector, actualOutputVector, 10, 
			&memorizedNeuronOutputs, pThis->m_bDistortTrainingPatterns );
		
		//显示输出特征点检测的理想结果与实际结果
		CString strPoint;

		cout<<"The actual output of facial point before refine is:\n";
		for(ii=0;ii<10;)
		{
			strPoint.Format("(%f , %f)\n", actualOutputVector[ii], actualOutputVector[ii+1]);
			cout<<strPoint;
			ii=ii+2;
		}
		cout<<endl<<endl;

		cout<<"The target output of facial point is:\n";
		for(ii=0;ii<10;)
		{
			strPoint.Format("(%f , %f)\n",targetOutputVector[ii],targetOutputVector[ii+1]);
			cout<<strPoint;
			ii=ii+2;
		}
		cout<<endl;
		
		memset(actualOutputVector,0,10);
		memset(targetOutputVector,0,10);

		if(rt == false)  //误差小于期望值时训练完成，退出BP线程
		{
			cout<<"Skip Backpropagation!!!";
		}
		cout<<endl<<endl;
	}

	cout<<"BP exit!!"<<endl;
	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    BP操作步骤三： BP训练过程的两部操作过程
//                   1）调用CalculateNeuralNet前向计算推导函数；
//				     2）获得输出向量结果若需调节则进行反方向传播，调用m_NN.Backpropagate( actualOutputVector, targetOutputVector，...);
//						,以调节参数权重值；
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CCreateNetwork::BackpropagateNeuralNet(double *inputVector, int iCount, double* targetOutputVector, 
									   double* actualOutputVector, int oCount, 
									   std::vector< std::vector< double > >* pMemorizedNeuronOutputs, 
									   BOOL bDistort )
{
	// function to backpropagate through the neural net. 
	ASSERT( (inputVector != NULL) && (targetOutputVector != NULL) && (actualOutputVector != NULL) );

	///////////////////////////////////////////////////////////////////////
	//
	// CODE REVIEW NEEDED:
	//
	// It does not seem worthwhile to backpropagate an error that's very small.  "Small" needs to be defined
	// and for now, "small" is set to a fixed size of pattern error ErrP <= 0.10 * MSE, then there will
	// not be a backpropagation of the error.  The current MSE is updated from the neural net dialog CDlgNeuralNet
	///////////////////////////////////////////////////////////////////////

	BOOL bWorthwhileToBackpropagate;  /////// part of code review

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//	！！！！！！！！！！局部作用域，作用域结束时，需要对CAutoMutex对象进行析构，从而释放对m_utxNeuralNet，即解除对神经网络的占用
	//	在FP计算过程中，必须独占神经网络
	//	local scope for capture of the neural net, only during the forward calculation step,
	//	i.e., we release neural net for other threads after the forward calculation, and after we
	//	have stored the outputs of each neuron, which are needed for the backpropagation step
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	{	
		//CAutoMutex tlo( m_utxNeuralNet ); //资源互斥上锁：获得神经网络控制权
		
		// determine if it's time to adjust the learning rate  是否需要调整学习速率(每经过10000次BP，目前取2000)
		if ( (( m_cBackprops % m_nAfterEveryNBackprops ) == 0) && (m_cBackprops != 0) )
		{
			double eta = m_NN.m_etaLearningRate;
			eta *= m_dEtaDecay;
			if ( eta < m_dMinimumEta )   //eta学习速率的下限
				eta = m_dMinimumEta;	 							  ///////////////////////////////////////////
			m_NN.m_etaLearningRatePrevious = m_NN.m_etaLearningRate;  //  记录之前的学习速率，作用？？？？
			m_NN.m_etaLearningRate = eta;                             ///////////////////////////////////////////
		}
		
		
		// determine if it's time to adjust the Hessian (currently once per epoch) 是否需要调整Hessian矩阵
		// 每次完整的训练周期为10000次BP（因为有10000张training images）
		//**** 本系统暂时不计算输入向量的hessian矩阵
		//if ( (m_bNeedHessian != FALSE) || (( m_cBackprops % CPreferences::GetPreferences().m_nItemsTrainingImages ) == 0) )
		//{
		//	// adjust the Hessian.  This is a lengthy operation, since it must process approx 500 labels
		//	CalculateHessian();  // 计算hessian矩阵
		//	
		//	m_bNeedHessian = FALSE;
		//}
		
		// determine if it's time to randomize the sequence of training patterns (currently once per epoch)
		// 每一次epoch的输入均是不同的图像输入序列
		//**** 本系统暂时不对输入训练序列进行乱序处理
		/*if ( ( m_cBackprops %  CPreferences::GetPreferences().m_nItemsTrainingImages ) == 0 )
		{
			RandomizeTrainingPatternSequence();   
		}*/
		
		// increment counter for tracking number of backprops
		m_cBackprops++;
		
		// 前向计算推导
		CalculateNeuralNet( inputVector, iCount, actualOutputVector, oCount, pMemorizedNeuronOutputs, bDistort );

		// calculate error in the output of the neural net
		// note that this code duplicates that found in many other places, and it's probably sensible to 
		// define a (global/static ??) function for it
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//  
		//	计算神经网络的计算平均误差MSE   
		//	本段代码可重用度很高，因此要特别注意全局/静态变量的定义
		//【targetOutputVector[10]保存真实的输出向量,对应五个预测特征点的位置值】
		//
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		double dMSE = 0.0;
		for ( int ii=0; ii<10; ++ii )
		{
			dMSE += ( actualOutputVector[ii]-targetOutputVector[ii] ) * ( actualOutputVector[ii]-targetOutputVector[ii] );
		}
		dMSE /= 2.0;

		//////////////////////////////////////////////////////////
		//
		//  设计技巧：当神经网络平均误差适当合理地小时则跳过当前BP
		//////////////////////////////////////////////////////////
		if ( dMSE <= ( 0.05 * m_dEstimatedCurrentMSE ) )
		{
			bWorthwhileToBackpropagate = FALSE;
		}
		else
		{
			bWorthwhileToBackpropagate = TRUE;
		}

		if( bWorthwhileToBackpropagate == FALSE)
		{
			//SaveLastWeight();
			return false;
		}


		//当训练过程未收敛且是首次FP完成时... ...
		if ( (bWorthwhileToBackpropagate != FALSE) && (pMemorizedNeuronOutputs == NULL) )
		{
			// the caller has not provided a place to store neuron outputs, so we need to
			// backpropagate now, while the neural net is still captured.  Otherwise, another thread
			// might come along and call CalculateNeuralNet(), which would entirely change the neuron
			// outputs and thereby inject errors into backpropagation 
			
			m_NN.Backpropagate( actualOutputVector, targetOutputVector, oCount, NULL );
			
			//SetModifiedFlag( TRUE );
			
			// we're done, so return
			return true;
		}
		
	}//局部作用域结束，即前向计算过程结束，释放对神经网络的占用
	
	// if we have reached here, then the mutex for the neural net has been released for other 
	// threads.  The caller must have provided a place to store neuron outputs, which we can 
	// use to backpropagate, even if other threads call CalculateNeuralNet() and change the outputs
	// of the neurons

	//当训练未收敛且不是首次FP完成... ...
	if ( (bWorthwhileToBackpropagate != FALSE) )
	{
		//////////////////////////////
		// back propagation
		// 回退计算
		//////////////////////////////
		m_NN.Backpropagate( actualOutputVector, targetOutputVector, oCount, pMemorizedNeuronOutputs );
		
		// set modified flag to prevent closure of doc without a warning
		//SetModifiedFlag( TRUE );
	}
	 
	//参数调整后输出新的检测结果
	/*double actualOutputVector2[10] = {0.0};
	CalculateNeuralNet( inputVector, iCount, actualOutputVector2, oCount, pMemorizedNeuronOutputs, bDistort );

	CString strPoint;
	cout<<endl<<endl;
	cout<<"The actual output of facial point after refine is:\n";
	for(int ii=0;ii<10;)
	{
		strPoint.Format("(%f , %f)\n",actualOutputVector2[ii],actualOutputVector2[ii+1]);
		cout<<strPoint;
		ii=ii+2;
	}
	memset(actualOutputVector2,0,10);
	cout<<endl<<endl;*/

	return true;
}


UINT CCreateNetwork::GetNextTrainingPattern(unsigned char* pArray, double* pPointLocation, BOOL bFlipGrayscale ,
		BOOL bFromRandomizedPatternSequence, UINT* iSequenceNum)
{
	 UINT iPatternNum;

	 iPatternNum = m_iNextTrainingPattern;

	 ASSERT(iPatternNum < CPreferences::GetPreferences()->m_nItemsTrainingImages);

	 GetTrainingPatternArrayValues(iPatternNum,pArray,FALSE);

	 if(iSequenceNum!=NULL)
	 {
		*iSequenceNum=m_iNextTrainingPattern;
	 }

	 m_iNextTrainingPattern++;

	 if ( m_iNextTrainingPattern >= CPreferences::GetPreferences()->m_nItemsTrainingImages )
	{
		m_iNextTrainingPattern = 0;
	}

	 return  iPatternNum;

}

void CCreateNetwork::GetTrainingPatternArrayValues(int iNumImage, unsigned char* pArray,BOOL bFlipGrayscale)
{
	int cCount = g_cImageSize*g_cImageSize; 
	CFileException fe;
	string sPath=m_ImageFilePaths[iNumImage].SpanExcluding(".").GetString();
	cout<<endl<<endl<<"***** "<<iNumImage+1<<" Processing: "<<sPath.c_str()<<endl;
	if(m_fileTrainingImages.Open(sPath.c_str(),CFile::modeRead,&fe) !=0)
	{
		if ( pArray != NULL )
		{
			m_fileTrainingImages.Read(pArray,cCount);

			if ( bFlipGrayscale != FALSE )
			{
				for ( int ii=0; ii<cCount; ++ii )
				{
					pArray[ ii ] = 255 - pArray[ ii ];
				}
			}

			m_fileTrainingImages.Close();
		}
	}
}


double CCreateNetwork::GetCurrentEta()
{
	return m_NN.m_etaLearningRate;
}


double CCreateNetwork::GetPreviousEta()
{
	// provided because threads might change the current eta before we are able to read it
	
	return m_NN.m_etaLearningRatePrevious;
}


UINT CCreateNetwork::GetCurrentTrainingPatternNumber( BOOL bFromRandomizedPatternSequence /* =FALSE */ )
{
	// returns the current number of the training pattern, either from the straight sequence, or from
	// the randomized sequence
	
	UINT iRet;
	
	if ( bFromRandomizedPatternSequence == FALSE )
	{
		iRet = m_iNextTrainingPattern;
	}
	else
	{
		iRet = m_iRandomizedTrainingPatternSequence[ m_iNextTrainingPattern ];
	}
	
	return iRet;
}

//对输入的训练数据集进行乱序处理
void CCreateNetwork::RandomizeTrainingPatternSequence()
{
	// randomizes the order of m_iRandomizedTrainingPatternSequence, which is a UINT array
	// holding the numbers 0..59999 in random order
	
	UINT ii, jj, iiMax, iiTemp;
	
	iiMax = CPreferences::GetPreferences()->m_nItemsTrainingImages;
	
	ASSERT( iiMax == 10000 );  // requirement of sloppy and unimaginative code
	
	// initialize array in sequential order
	
	for ( ii=0; ii<iiMax; ++ii )
	{
		m_iRandomizedTrainingPatternSequence[ ii ] = ii;  
	}
	
	// now at each position, swap with a random position
	// 随机乱序交换输入序列索引
	for ( ii=0; ii<iiMax; ++ii )
	{
		jj = (UINT)( UNIFORM_ZERO_THRU_ONE * iiMax ); //UNIFORM_ZERO_THRU_ONE = 0 or 1
		
		ASSERT( jj < iiMax );
		
		iiTemp = m_iRandomizedTrainingPatternSequence[ ii ];
		m_iRandomizedTrainingPatternSequence[ ii ] = m_iRandomizedTrainingPatternSequence[ jj ];
		m_iRandomizedTrainingPatternSequence[ jj ] = iiTemp;
	}
	
}

// 加载权值文件,得到各层训练好的权重值
void CCreateNetwork::LoadWeightFile(vector<double>* vWeightOfLayer, char* path)
{
	CStdioFile sfp;			//权值文件句柄
	CString csWeightValue;  //存储从文件读出的权值字符串
	string strFilePath;     //完整的权值文件路径

	strFilePath=path;
	if(sfp.Open(strFilePath.c_str(),CFile::modeRead)!=0)
	{
		while(sfp.ReadString(csWeightValue))
		{
			vWeightOfLayer->push_back(atof(csWeightValue));
		}
	}
	else
	{
		cout<<"权值文件:"<<path<<"打开失败！"<<endl;
	}
	sfp.Close();
}


// 调用Opencv，实时在人脸图片上标记出检测到的特征点
void CCreateNetwork::DisplayPointDetected(CString csImageFilePath, double* dPointDetected)
{
	//将特征点坐标反归一化,并转换为整形像素点坐标
	int iPointValue[10];
	double dResidual;
	for(int ii=0;ii<10;ii++)
	{
		dPointDetected[ii]=dPointDetected[ii]*39;
		dResidual=dPointDetected[ii]-(int)dPointDetected[ii];

		if(dResidual >=0 && dResidual <= 0.5)  //对坐标值取整处理，便于标记
		{
			iPointValue[ii]=(int)dPointDetected[ii];
		}
		else
		{
			iPointValue[ii]=(int)dPointDetected[ii]+1;
		}
	}

	IplImage* image=cvLoadImage(csImageFilePath.GetBuffer(0),0);

//	cvNamedWindow("人脸特征点检测结果展示");

	CvPoint ptFace;
	
	for(int ii=0;ii<10;)
	{
		ptFace.x=iPointValue[ii];
		ptFace.y=iPointValue[ii+1];
		ii=ii+2;

		cvCircle(image,ptFace,1,CV_RGB(255,0,0),-1,8,0);
	}
	cvNamedWindow("人脸特征点检测结果展示",CV_WINDOW_AUTOSIZE);
	cvShowImage("人脸特征点检测结果展示",image);
	cvWaitKey(0);
	cvDestroyWindow("人脸特征点检测结果展示");//销毁窗口
	cvReleaseImage(&image);

}


// BP完成退出训练前保存当前网络的权重值
void CCreateNetwork::SaveLastWeight(void)
{
	string strPath="";
	string strFileName="";
	char ch[30];
	char wt[15];
	double dValue;  //保存权重值

	int nWeightCount=0; //保存当前层的权重个数
	CFileException fe;

	VectorLayers::iterator lit = m_NN.m_Layers.end() - 1;
	int flagBP=9; //标记当前已处理到哪一层，从最后一层开始
	for ( lit; lit>m_NN.m_Layers.begin(),flagBP>0; lit--,flagBP--)
	{
		switch(flagBP)
		{
			case 9: 
			case 8:
			case 7:
			case 5:
			case 3:
			case 1:
				sprintf(ch,"\\%d-complete.txt",flagBP);
				strFileName=ch;
				strPath=m_sWeightSavePath;
				strPath+=strFileName;

				nWeightCount=((*lit)->m_Weights).size();

				if(m_fileWeight.Open(strPath.c_str(),CFile::modeCreate|CFile::modeWrite,&fe) !=0)
				{
					for(int i=0;i<nWeightCount;i++)
					{
						dValue=(((*lit)->m_Weights)[i])->value;
						sprintf(wt,"%f\r\n",dValue);   //一个权值单独保存为一行，便于读取
						m_fileWeight.WriteString((LPCTSTR)wt);
					}
				}
				m_fileWeight.Close();
				break;

		case 6:  //pooling层不需要保存
		case 4:  
		case 2: 
			break;
		}
	}
}
