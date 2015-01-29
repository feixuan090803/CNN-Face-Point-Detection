// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "NeuralNetwork.h"
#include <malloc.h>  // for the _alloca function
//#include <afx.h>

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif


///////////////////////////////////////////////////////////////////////
//
//  NeuralNetwork class definition

NeuralNetwork::NeuralNetwork()
{
	Initialize();
}

void NeuralNetwork::Initialize()
{
	// delete all layers
	
	VectorLayers::iterator it;
	
	for( it=m_Layers.begin(); it<m_Layers.end(); it++ )
	{
		delete *it;
	}
	
	m_Layers.clear();
	
	m_etaLearningRate = .001;  // arbitrary, so that brand-new NNs can be serialized with a non-ridiculous number
	m_cBackprops = 0;
	
	m_sWeightSavePath= "E:\\人脸识别\\face point detection\\weight\\";
}

NeuralNetwork::~NeuralNetwork()
{
	// call Initialize(); makes sense if you think
	Initialize();
}

///////////////////////////////////////////////////////////////////
//	NeuralNetwork::Calculate:	前向计算
//		输入初始向量，经过整个神经网络的计算，得到计算的输出结果
//		期间逐层调用相邻层的计算函数NNLayer::Calculate(...)完成计算
//		最后将神经网络每一层的output输出向量统一存储用于BP过程
///////////////////////////////////////////////////////////////////
void NeuralNetwork::Calculate(double* inputVector, UINT iCount, 
							  double* outputVector /* =NULL */, UINT oCount /* =0 */,
							  std::vector< std::vector< double > >* pNeuronOutputs /* =NULL */ )
{
	VectorLayers::iterator lit = m_Layers.begin();
	VectorNeurons::iterator nit;
	
	int flagLayer;
	// first layer is input layer: directly set outputs of all of its neurons
	// to the input vector
	// 初始化输入层
	if ( lit<m_Layers.end() )  
	{
		nit = (*lit)->m_Neurons.begin();
		int count = 0;
		flagLayer=0;

		ASSERT( iCount == (*lit)->m_Neurons.size() );  // there should be exactly one neuron per input
		
		while( ( nit < (*lit)->m_Neurons.end() ) && ( count < iCount ) )
		{
			(*nit)->output = inputVector[ count ];
			nit++;
			count++;
		}
	}
	
	//从前往后依次对神经网络各层进行迭代计算
	for( lit++,flagLayer++; lit<m_Layers.end(); lit++, flagLayer++ )
	{
		if(flagLayer%2 !=0 || flagLayer >=8)   //若当前层为卷积层或全连接层
		{
			(*lit)->Calculate();
		}						  
		else								   //若当前层为下采样层
		{
			(*lit)->AveragedPooling();
		}
	}
	
	// load up output vector with results
	//获得神经网络计算的输出向量结果
	if ( outputVector != NULL )
	{
		lit = m_Layers.end();
		lit--;
		
		nit = (*lit)->m_Neurons.begin();
		
		for ( int ii=0; ii<oCount; ++ii )
		{
			outputVector[ ii ] = (*nit)->output;
			nit++;
		}
	}
	
	// load up neuron output values with results
	//若输出向量数组为空，则为首次FP，需要分配存储空间（vector::push_back）
	if ( pNeuronOutputs != NULL )
	{
		// check for first time use (re-use is expected)
		
		if ( pNeuronOutputs->empty() != FALSE )
		{
			// it's empty, so allocate memory for its use
			
			pNeuronOutputs->clear();  // for safekeeping
			
			//将神经网络的每一层的输出向量保存至二维数组*pNeuronOutputs(即 std::vector< std::vector< double > > memorizedNeuronOutputs)
			int ii = 0;
			for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
			{
				std::vector< double > layerOut;
				
				for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
				{
					layerOut.push_back( (*lit)->m_Neurons[ ii ]->output );
				}
				
				pNeuronOutputs->push_back( layerOut);
			}
		}
		else 
		{
			// it's not empty, so assume it's been used in a past iteration and memory for
			// it has already been allocated internally.  Simply store the values
			int ii, jj = 0;
			for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
			{
				for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
				{
					//保存output向量的数组空间已分配，直接赋值覆盖即可
					(*pNeuronOutputs)[ jj ][ ii ] = (*lit)->m_Neurons[ ii ]->output ;
				}
				++jj;
			}
		}
	}
}


void NeuralNetwork::Backpropagate(double *actualOutput, double *desiredOutput, UINT count,
								  std::vector< std::vector< double > >* pMemorizedNeuronOutputs )
{
	// backpropagates through the neural net
	
	ASSERT( ( actualOutput != NULL ) && ( desiredOutput != NULL ) && ( count < 256 ) );
	
	ASSERT( m_Layers.size() >= 2 );  // there must be at least two layers in the net
	
	if ( ( actualOutput == NULL ) || ( desiredOutput == NULL ) || ( count >= 256 ) )
		return;
	
	// check if it's time for a weight sanity check
	
	m_cBackprops++;
	
	if ( (m_cBackprops % 2000) == 0 )
	{
		// every 10000 backprops
		PeriodicWeightSanityCheck();
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////////////
	// proceed from the last layer to the first, iteratively
	// We calculate the last layer separately, and first, since it provides the needed derviative
	// (i.e., dErr_wrt_dXnm1) for the previous layers
	
	// nomenclature:   命名术语
	//
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer
	// Wn is the vector of weights of the n-th layer
	// Yn is the activation value of the n-th layer, i.e., the weighted sum of inputs BEFORE the squashing function is applied
	// F is the squashing function: Xn = F(Yn)
	// F' is the derivative of the squashing function
	//   Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2, i.e., the derivative can be calculated from the output, without knowledge of the input
	/////////////////////////////////////////////////////////////////////////////////////////////////
	
	//先从倒数第二层
	VectorLayers::iterator lit = m_Layers.end() - 1;
	
	std::vector< double > dErr_wrt_dXlast( (*lit)->m_Neurons.size() );
	std::vector< std::vector< double > > differentials;
	
	int iSize = m_Layers.size();
	
	differentials.resize( iSize );
	
	int ii;
	
	// start the process by calculating dErr_wrt_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
	// the difference between the target and the actual
	// 从最后一层开始，获得实际输出与理论输出的差值作为误差偏导数，组成一个误差偏导向量
	for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
	{
		dErr_wrt_dXlast[ ii ] = actualOutput[ ii ] - desiredOutput[ ii ];
	}
	
	// store Xlast and reserve memory for the remaining vectors stored in differentials
	//保存最后一层的误差偏导值向量，并初始化其他层的偏导向量
	differentials[ iSize-1 ] = dErr_wrt_dXlast;  // last one
	for ( ii=0; ii<iSize-1; ++ii )
	{
		differentials[ ii ].resize( m_Layers[ii]->m_Neurons.size(), 0.0 );
	}
	
	// now iterate through all layers including the last but excluding the first, and ask each of
	// them to backpropagate error and adjust their weights, and to return the differential
	// dErr_wrt_dXnm1 for use as the input value of dErr_wrt_dXn for the next iterated layer
	// 开始反向传递误差，并调整权重值

	//检测FP过程完成后，各层output值是否已存储
	BOOL bMemorized = ( pMemorizedNeuronOutputs != NULL );
	
	//从倒数第二层往前逐层计算，调用NNLayer::Backpropagate() ,开始真正的BP操作
	lit = m_Layers.end() - 1;  // re-initialized to last layer for clarity, although it should already be this value

	ii = iSize - 1;
	int i=0,flagBP=9;

	for ( lit; lit>m_Layers.begin(),flagBP>0; lit--,flagBP--)
	{
		if ( bMemorized != FALSE )
		{
			if(flagBP%2 != 0 || flagBP >=8)	  //全连接层或“pooling-卷积层”的BP参数调整过程
			{
				(*lit)->Backpropagate( differentials[ ii ], differentials[ ii - 1 ], 
					&(*pMemorizedNeuronOutputs)[ ii ], &(*pMemorizedNeuronOutputs)[ ii - 1 ], m_etaLearningRate );
			}
			else	 //“卷积层-pooling层”间的BP参数调整过程
			{
				(*lit)->Backpropagate2( differentials[ ii ], differentials[ ii - 1 ], 
					&(*pMemorizedNeuronOutputs)[ ii ], &(*pMemorizedNeuronOutputs)[ ii - 1 ], m_etaLearningRate );
			}
		}
		else
		{
			if(flagBP%2 != 0 || flagBP >=8)
			{
				(*lit)->Backpropagate( differentials[ ii ], differentials[ ii - 1 ], 
					NULL, NULL, m_etaLearningRate );
			}
			else
			{
				(*lit)->Backpropagate2( differentials[ ii ], differentials[ ii - 1 ], 
					NULL, NULL, m_etaLearningRate );
			}
		}
		--ii;
	}

	///////////////////////////////////////////////////////////////
	//加入网络权值保存模块（当前设定为每经过100次BP操作保存一次）
	///////////////////////////////////////////////////////////////
	if(this->m_cBackprops % 100 == 0)
	{
		lit = m_Layers.end() - 1;  // locate the penultimat layer 
		//ii = iSize - 1;
		string strPath="";
		string strFileName="";
		char ch[10];
		char wt[15];
		double dValue;  //保存权重值

		int nWeightCount=0; //保存当前层的权重个数
		CFileException fe;

		flagBP=9; //标记当前已处理到哪一层，从最后一层开始

		for ( lit; lit>m_Layers.begin(),flagBP>0; lit--,flagBP--)
		{
			switch(flagBP)
			{
				case 9: 
				case 8:
				case 7:
				case 5:
				case 3:
				case 1:
					sprintf(ch,"%d-%d.txt",flagBP,this->m_cBackprops/100);
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

	differentials.clear();
}

// 权值取值的合理性检测								  
void NeuralNetwork::PeriodicWeightSanityCheck()
{
	// fucntion that simply goes through all weights, and tests them against an arbitrary
	// "reasonable" upper limit.  If the upper limit is exceeded, a warning is displayed
	
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++)
	{
		(*lit)->PeriodicWeightSanityCheck();
	}
}


void NeuralNetwork::EraseHessianInformation()
{
	// controls each layer to erase (set to value of zero) all its diagonal Hessian info
	
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
	{
		(*lit)->EraseHessianInformation();
	}
	
}


void NeuralNetwork::DivideHessianInformationBy( double divisor )
{
	// controls each layer to divide its current diagonal Hessian info by a common divisor. 
	// A check is also made to ensure that each Hessian is strictly zero-positive
	
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
	{
		(*lit)->DivideHessianInformationBy( divisor );
	}
	
}


void NeuralNetwork::BackpropagateSecondDervatives( double* actualOutputVector, 
												  double* targetOutputVector, UINT count )
{
	// calculates the second dervatives (for diagonal Hessian) and backpropagates
	// them through neural net
	
	ASSERT( ( actualOutputVector != NULL ) && ( targetOutputVector != NULL ) && ( count < 256 ) );
	
	ASSERT( m_Layers.size() >= 2 );  // there must be at least two layers in the net
	
	if ( ( actualOutputVector == NULL ) || ( targetOutputVector == NULL ) || ( count >= 256 ) )
		return;
	
	// we use nearly the same nomenclature as above (e.g., "dErr_wrt_dXnm1") even though everything here
	// is actually second derivatives and not first derivatives, since otherwise the ASCII would 
	// become too confusing.  To emphasize that these are second derivatives, we insert a "2"
	// such as "d2Err_wrt_dXnm1".  We don't insert the second "2" that's conventional for designating
	// second derivatives
	
	
	VectorLayers::iterator lit;
	
	lit = m_Layers.end() - 1;  // set to last layer
	
	std::vector< double > d2Err_wrt_dXlast( (*lit)->m_Neurons.size() );
	std::vector< std::vector< double > > differentials;
	
	int iSize = m_Layers.size();
	
	differentials.resize( iSize );
	
	int ii;
	
	// start the process by calculating the second derivative dErr_wrt_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is 
	// exactly one
	
	lit = m_Layers.end() - 1;  // point to last layer
	
	for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
	{
		d2Err_wrt_dXlast[ ii ] = 1.0;
	}
	
	
	// store Xlast and reserve memory for the remaining vectors stored in differentials
	
	differentials[ iSize-1 ] = d2Err_wrt_dXlast;  // last one
	
	for ( ii=0; ii<iSize-1; ++ii )
	{
		differentials[ ii ].resize( m_Layers[ii]->m_Neurons.size(), 0.0 );
	}
	
	
	// now iterate through all layers including the last but excluding the first, starting from
	// the last, and ask each of
	// them to backpropagate the second derviative and accumulate the diagonal Hessian, and also to
	// return the second dervative
	// d2Err_wrt_dXnm1 for use as the input value of dErr_wrt_dXn for the next iterated layer (which
	// is the previous layer spatially)
	
	lit = m_Layers.end() - 1;  // re-initialized to last layer for clarity, although it should already be this value
	
	ii = iSize - 1;
	for ( lit; lit>m_Layers.begin(); lit--)
	{
		(*lit)->BackpropagateSecondDerivatives( differentials[ ii ], differentials[ ii - 1 ] );	   
		--ii;
	}
	
	differentials.clear();
}


void NeuralNetwork::Serialize(CArchive &ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
		
		ar << m_etaLearningRate;
		
		ar << m_Layers.size();
		
		VectorLayers::iterator lit;
		
		for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
		{
			(*lit)->Serialize( ar );
		}
		
	}
	else
	{
		// TODO: add loading code here
		
		double eta; 
		ar >> eta;
		m_etaLearningRate = eta;  // two-step storage is needed since m_etaLearningRate is "volatile"
		
		int nLayers;
		NNLayer* pLayer = NULL;
		
		ar >> nLayers;
		
		for ( int ii=0; ii<nLayers; ++ii )
		{
			pLayer = new NNLayer( _T(""), pLayer );
			
			m_Layers.push_back( pLayer );
			
			pLayer->Serialize( ar );
		}
		
	}
	
}


///////////////////////////////////////////////////////////////////////
//
//  NNLayer class definition


NNLayer::NNLayer() :
label( _T("") ), m_pPrevLayer( NULL )
{
	Initialize();
}

NNLayer::NNLayer( LPCTSTR str, NNLayer* pPrev /* =NULL */ ) :
label( str ), m_pPrevLayer( pPrev )
{
	Initialize();
}


void NNLayer::Initialize()
{
	VectorWeights::iterator wit;
	VectorNeurons::iterator nit;
	
	for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		delete *nit;
	}
	
	for( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		delete *wit;
	}
	
	m_Weights.clear();
	m_Neurons.clear();
	
	m_bFloatingPointWarning = false;
	
}

NNLayer::~NNLayer()
{
	// call Initialize(); makes sense if you think
	
	Initialize();
}

///////////////////////////////////////////////
//本层FP操作，相邻层计算输入并得到输出
///////////////////////////////////////////////
//1、计算卷积
void NNLayer::Calculate()
{
	ASSERT( m_pPrevLayer != NULL );
	
	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;
	
	double dSum;
	
	for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // to ease the terminology
		
		cit = n.m_Connections.begin();
		
		ASSERT( (*cit).WeightIndex < m_Weights.size() );
		
		dSum = m_Weights[ (*cit).WeightIndex ]->value;  // weight of the first connection is the bias; neuron is ignored
		
		for ( cit++ ; cit<n.m_Connections.end(); cit++ )
		{
			ASSERT( (*cit).WeightIndex < m_Weights.size() );
			ASSERT( (*cit).NeuronIndex < m_pPrevLayer->m_Neurons.size() );
			
			dSum += ( m_Weights[ (*cit).WeightIndex ]->value ) * 
				( m_pPrevLayer->m_Neurons[ (*cit).NeuronIndex ]->output );
		}
		//!!!!!!!!!!!!!!!!!!!!!!!!!!
		//使用sigmod函数作为激活函数
		//n.output = abs(SIGMOID( dSum ));

		//使用ReLU线性修正函数作为激活函数
		if(dSum>=0)
		{
			n.output=dSum;
		}
		else
		{
			n.output=0;
		}
	}
	
}

//2、计算average pooling
void NNLayer::AveragedPooling()
{
	ASSERT( m_pPrevLayer != NULL );
	
	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;
	
	double dSum;
	
	for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // to ease the terminology
		
		cit = n.m_Connections.begin();
		
		ASSERT( (*cit).WeightIndex < m_Weights.size() );
		
		//dSum = m_Weights[ (*cit).WeightIndex ]->value;  // weight of the first connection is the bias; neuron is ignored
		dSum=0;

//		for ( cit++ ; cit<n.m_Connections.end(); cit++ )
		for ( ; cit<n.m_Connections.end(); cit++ )
		{
			//ASSERT( (*cit).WeightIndex < m_Weights.size() );
			ASSERT( (*cit).NeuronIndex < m_pPrevLayer->m_Neurons.size() );
			
			dSum += ( m_Weights[ (*cit).WeightIndex ]->value ) * 
				( m_pPrevLayer->m_Neurons[ (*cit).NeuronIndex ]->output );
		}
		
		n.output=dSum;
		//n.output = SIGMOID( dSum );
		
	}


}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
//全连接层、“pooling层-卷积层”BP操作，误差计算，调节本层权重
/////////////////////////////////////////////////////////////////
void NNLayer::Backpropagate( std::vector< double >& dErr_wrt_dXn /* in */, 
							std::vector< double >& dErr_wrt_dXnm1 /* out */, 
							std::vector< double >* thisLayerOutput,  // memorized values of this layer's output
							std::vector< double >* prevLayerOutput,  // memorized values of previous layer's output
							double etaLearningRate )
{
	// nomenclature (repeated from NeuralNetwork class):
	// 
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer
	// Wn is the vector of weights of the n-th layer
	// Yn is the activation value of the n-th layer, i.e., the weighted sum of inputs BEFORE the squashing function is applied
	// F is the squashing function: Xn = F(Yn)
	// F' is the derivative of the squashing function
	// Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2, i.e., the derivative can be calculated from the output, without knowledge of the input
	
	ASSERT( dErr_wrt_dXn.size() == m_Neurons.size() );
	ASSERT( m_pPrevLayer != NULL );
	ASSERT( dErr_wrt_dXnm1.size() == m_pPrevLayer->m_Neurons.size() );
	


	int ii, jj;
	UINT kk;
	int nIndex;
	double output;
	
	int iDeriveOfReLU;
	std::vector< double > dErr_wrt_dYn( m_Neurons.size() );
	//
	//	std::vector< double > dErr_wrt_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	//////////////////////////////////////////////////
	//
	///// DESIGN TRADEOFF: REVIEW !!
	// We would prefer (for ease of coding) to use STL vector for the array "dErr_wrt_dWn", which is the 
	// differential of the current pattern's error wrt weights in the layer.  However, for layers with
	// many weights, such as fully-connected layers, there are also many weights.  The STL vector
	// class's allocator is remarkably stupid when allocating large memory chunks, and causes a remarkable 
	// number of page faults, with a consequent slowing of the application's overall execution time.
	
	// To fix this, I tried using a plain-old C array, by new'ing the needed space from the heap, and 
	// delete[]'ing it at the end of the function.  However, this caused the same number of page-fault
	// errors, and did not improve performance.
	
	// So I tried a plain-old C array allocated on the stack (i.e., not the heap).  Of course I could not
	// write a statement like 
	//    double dErr_wrt_dWn[ m_Weights.size() ];
	// since the compiler insists upon a compile-time known constant value for the size of the array.  
	// To avoid this requirement, I used the _alloca function, to allocate memory on the stack.
	// The downside of this is excessive stack usage, and there might be stack overflow probelms.  That's why
	// this comment is labeled "REVIEW"
	
	double* dErr_wrt_dWn = (double*)( _alloca( sizeof(double) *  m_Weights.size() ) );
	
	for ( ii=0; ii<m_Weights.size(); ++ii )
	{
		dErr_wrt_dWn[ ii ] =0.0;
	}
	
	
	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;
	
	
	BOOL bMemorized = ( thisLayerOutput != NULL ) && ( prevLayerOutput != NULL );
	
	
	// calculate dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
	
	for ( ii=0; ii<m_Neurons.size(); ++ii )
	{
		ASSERT( ii<dErr_wrt_dYn.size() );
		ASSERT( ii<dErr_wrt_dXn.size() );
		
		if ( bMemorized != FALSE )
		{
			output = (*thisLayerOutput)[ ii ];
		}
		else
		{
			output = m_Neurons[ ii ]->output;
		}
		
		if(output>=0)
		{
			iDeriveOfReLU=1;
		}
		else
		{
			iDeriveOfReLU=0;
		}

		//dErr_wrt_dYn[ ii ] = DSIGMOID( output ) * dErr_wrt_dXn[ ii ];
		dErr_wrt_dYn[ ii ] = iDeriveOfReLU * dErr_wrt_dXn[ ii ];
	
	}
	
	// calculate dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
	// For each neuron in this layer, go through the list of connections from the prior layer, and
	// update the differential for the corresponding weight

	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk = (*cit).NeuronIndex;
			if ( kk == ULONG_MAX )
			{
				output = 1.0;  // this is the bias weight
			}
			else
			{
				ASSERT( kk<m_pPrevLayer->m_Neurons.size() );
				
				if ( bMemorized != FALSE )
				{
					output = (*prevLayerOutput)[ kk ];
				}
				else
				{
					output = m_pPrevLayer->m_Neurons[ kk ]->output;
				}
			}
			////////////	ASSERT( (*cit).WeightIndex < dErr_wrt_dWn.size() );  // since after changing dErr_wrt_dWn to a C-style array, the size() function this won't work
			ASSERT( ii<dErr_wrt_dYn.size() );
			dErr_wrt_dWn[ (*cit).WeightIndex ] += dErr_wrt_dYn[ ii ] * output;
		}
		
		ii++;
	}
	
	
	// calculate dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn, which is needed as the input value of
	// dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
	// For each neuron in this layer
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk=(*cit).NeuronIndex;
			if ( kk != ULONG_MAX )
			{
				// we exclude ULONG_MAX, which signifies the phantom bias neuron with
				// constant output of "1", since we cannot train the bias neuron
				
				nIndex = kk;
				
				ASSERT( nIndex<dErr_wrt_dXnm1.size() );
				ASSERT( ii<dErr_wrt_dYn.size() );
				ASSERT( (*cit).WeightIndex<m_Weights.size() );
				
				dErr_wrt_dXnm1[ nIndex ] += dErr_wrt_dYn[ ii ] * m_Weights[ (*cit).WeightIndex ]->value;
			}
			
		}
		
		ii++;  // ii tracks the neuron iterator
		
	}
	
	//struct DOUBLE_UNION  //联合结构
	//{
	//	union 
	//	{
	//		double dd;
	//		unsigned __int64 ullong;
	//	};
	//};
	
	double oldValue;
	//newValue;
	
	// 更新权值，比较与更新权值操作必须为原子操作！！！！
	// finally, update the weights of this layer neuron using dErr_wrt_dW and the learning rate eta
	// Use an atomic compare-and-exchange operation, which means that another thread might be in 
	// the process of backpropagation and the weights might have shifted slightly
	
	//二次收敛有关参数，使用的是随机梯度下降法
	
	//*********double dMicron = ::GetPreferences().m_dMicronLimitParameter;
	double dMicron =0.10;
	double epsilon, divisor;
	
	for ( jj=0; jj<m_Weights.size(); ++jj )
	{
		divisor = m_Weights[ jj ]->diagHessian + dMicron ; 
		
		// the following code has been rendered unnecessary, since the value of the Hessian has been
		// verified when it was created, so as to ensure that it is strictly
		// zero-positve.  Thus, it is impossible for the diagHessian to be less than zero,
		// and it is impossible for the divisor to be less than dMicron
		/*
		if ( divisor < dMicron )  
		{
		// it should not be possible to reach here, since everything in the second derviative equations 
		// is strictly zero-positive, and thus "divisor" should definitely be as large as MICRON.
		
		  ASSERT( divisor >= dMicron );
		  divisor = 1.0 ;  // this will limit the size of the update to the same as the size of gloabal eta
		  }
		*/
		
		//epsilon = etaLearningRate / divisor;  //epsilon二次收敛系数
		oldValue= m_Weights[ jj ]->value;
		//newValue = oldValue.dd - epsilon * dErr_wrt_dWn[ jj ];
		m_Weights[ jj ]->value = oldValue - etaLearningRate * dErr_wrt_dWn[ jj ];
		

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		//  
		//  InterlockedCompareExchange：把目标操作数（第1参数所指向的内存中的数）与一个值（第3参数）比较，
		//  如果相等，则用另一个值（第2参数）与目标操作数（第1参数所指向的内存中的数）交换；
		//  整个操作过程是锁定内存的，其它处理器不会同时访问内存，从而实现多处理器环境下的线程互斥。
		//  在此运用是为了保证权值的比较和更改为原子操作，在此过程不会被其他线程更改！！！！！！！
		////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		//*****************
		//while ( oldValue.ullong != _InterlockedCompareExchange64( (unsigned __int64*)(&m_Weights[ jj ]->value), 
		//	newValue.ullong, oldValue.ullong ) ) 
		//{
		//	// another thread must have modified the weight.  Obtain its new value, adjust it, and try again
		//	
		//	oldValue.dd = m_Weights[ jj ]->value;
		//	newValue.dd = oldValue.dd - epsilon * dErr_wrt_dWn[ jj ];
		//}
	}
	
}

///////////////////////////////////////////////////////////////////
//“卷积层-pooling层”BP操作，只需进行误差传递，不需对权重进行调整，卷积层的代码可以部分复用
///////////////////////////////////////////////////////////////////
void NNLayer::Backpropagate2( std::vector< double >& dErr_wrt_dXn /* in */, 
							std::vector< double >& dErr_wrt_dXnm1 /* out */, 
							std::vector< double >* thisLayerOutput,  // memorized values of this layer's output
							std::vector< double >* prevLayerOutput,  // memorized values of previous layer's output
							double etaLearningRate )
{
	// nomenclature (repeated from NeuralNetwork class):
	// 
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer
	// Wn is the vector of weights of the n-th layer
	// Yn is the activation value of the n-th layer, i.e., the weighted sum of inputs BEFORE the squashing function is applied
	// F is the squashing function: Xn = F(Yn)
	// F' is the derivative of the squashing function
	// Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2, i.e., the derivative can be calculated from the output, without knowledge of the input
	
	ASSERT( dErr_wrt_dXn.size() == m_Neurons.size() );
	ASSERT( m_pPrevLayer != NULL );
	ASSERT( dErr_wrt_dXnm1.size() == m_pPrevLayer->m_Neurons.size() );
	
	int ii, jj;
	UINT kk;
	int nIndex;
	double output;
	
	int iDeriveOfReLU;

	std::vector< double > dErr_wrt_dYn( m_Neurons.size() );
	//
	//	std::vector< double > dErr_wrt_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	//////////////////////////////////////////////////
	//
	///// DESIGN TRADEOFF: REVIEW !!		使用alloca在栈上分配空间用于存储权值
	// We would prefer (for ease of coding) to use STL vector for the array "dErr_wrt_dWn", which is the 
	// differential of the current pattern's error wrt weights in the layer.  However, for layers with
	// many weights, such as fully-connected layers, there are also many weights.  The STL vector
	// class's allocator is remarkably stupid when allocating large memory chunks, and causes a remarkable 
	// number of page faults, with a consequent slowing of the application's overall execution time.
	
	// To fix this, I tried using a plain-old C array, by new'ing the needed space from the heap, and 
	// delete[]'ing it at the end of the function.  However, this caused the same number of page-fault
	// errors, and did not improve performance.
	
	// So I tried a plain-old C array allocated on the stack (i.e., not the heap).  Of course I could not
	// write a statement like 
	//    double dErr_wrt_dWn[ m_Weights.size() ];
	// since the compiler insists upon a compile-time known constant value for the size of the array.  
	// To avoid this requirement, I used the _alloca function, to allocate memory on the stack.
	// The downside of this is excessive stack usage, and there might be stack overflow probelms.  That's why
	// this comment is labeled "REVIEW"
	
	double* dErr_wrt_dWn = (double*)( _alloca( sizeof(double) *  m_Weights.size() ) );
	
	for ( ii=0; ii<m_Weights.size(); ++ii )
	{
		dErr_wrt_dWn[ ii ] =0.0;
	}
	
	
	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;
	
	
	BOOL bMemorized = ( thisLayerOutput != NULL ) && ( prevLayerOutput != NULL );
	
	
	// calculate dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
	
	for ( ii=0; ii<m_Neurons.size(); ++ii )
	{
		ASSERT( ii<dErr_wrt_dYn.size() );
		ASSERT( ii<dErr_wrt_dXn.size() );
		
		if ( bMemorized != FALSE )
		{
			output = (*thisLayerOutput)[ ii ];
		}
		else
		{
			output = m_Neurons[ ii ]->output;
		}
		
		if(output>=0)
		{
			iDeriveOfReLU=1;
		}
		else
		{
			iDeriveOfReLU=0;
		}

		//dErr_wrt_dYn[ ii ] = DSIGMOID( output ) * dErr_wrt_dXn[ ii ];
		dErr_wrt_dYn[ ii ] = iDeriveOfReLU * dErr_wrt_dXn[ ii ];
	}
	
	// calculate dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn, which is needed as the input value of
	// dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
	// For each neuron in this layer
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk=(*cit).NeuronIndex;
			if ( kk != ULONG_MAX )
			{
				// we exclude ULONG_MAX, which signifies the phantom bias neuron with
				// constant output of "1", since we cannot train the bias neuron
				
				nIndex = kk;
				
				ASSERT( nIndex<dErr_wrt_dXnm1.size() );
				ASSERT( ii<dErr_wrt_dYn.size() );
				ASSERT( (*cit).WeightIndex<m_Weights.size() );
				
				dErr_wrt_dXnm1[ nIndex ] += dErr_wrt_dYn[ ii ] * m_Weights[ (*cit).WeightIndex ]->value;
			}
		}
		ii++;  // ii tracks the neuron iterator
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

void NNLayer::PeriodicWeightSanityCheck()
{
	// called periodically by the neural net, to request a check on the "reasonableness" of the 
	// weights.  The warning message is given only once per layer
	
	VectorWeights::iterator wit;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		NNWeight& ww = *(*wit);
		double val = fabs( ww.value );  //fabs() 取浮点数的绝对值(abs of float)
		
		if ( (val>100.0) && (m_bFloatingPointWarning == false) )
		{
			// 100.0 is an arbitrary value, that no reasonable weight should ever exceed
			
			CString strMess;
			strMess.Format( _T( "Caution: Weights are becoming unboundedly large \n" )
				_T( "Layer: %s \nWeight: %s \nWeight value = %g \nWeight Hessian = %g\n\n" )
				_T( "Suggest abandoning this backpropagation and investigating" ),
				label.c_str(), ww.label.c_str(), ww.value, ww.diagHessian );
			
			::MessageBox( NULL, strMess, _T( "Problem With Weights" ), MB_ICONEXCLAMATION | MB_OK );
			
			m_bFloatingPointWarning = true;
		}
	}
}


void NNLayer::EraseHessianInformation()
{
	// goes through all the weights associated with this layer, and sets each of their
	// diagHessian value to zero
	
	VectorWeights::iterator wit;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		(*wit)->diagHessian = 0.0;
	}
	
}

void NNLayer::DivideHessianInformationBy(double divisor)
{
	// goes through all the weights associated with this layer, and divides each of their
	// diagHessian value by the indicated divisor
	
	VectorWeights::iterator wit;
	double dTemp;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		dTemp = (*wit)->diagHessian;
		
		if ( dTemp < 0.0 )
		{
			// it should not be possible to reach here, since all calculations for the second
			// derviative are strictly zero-positive.  However, there are some early indications 
			// that this check is necessary anyway
			
			ASSERT ( dTemp >= 0.0 );  // will break in debug mode
			dTemp = 0.0;
		}
		
		(*wit)->diagHessian = dTemp / divisor ;
	}
}


/////////////////////////////////////////////////////
//
//  二次收敛
/////////////////////////////////////////////////////
void NNLayer::BackpropagateSecondDerivatives( std::vector< double >& d2Err_wrt_dXn /* in */, 
											 std::vector< double >& d2Err_wrt_dXnm1 /* out */)
{
	// nomenclature (repeated from NeuralNetwork class)
	// NOTE: even though we are addressing SECOND derivatives ( and not first derivatives),
	// we use nearly the same notation as if there were first derivatives, since otherwise the
	// ASCII look would be confusing.  We add one "2" but not two "2's", such as "d2Err_wrt_dXn",
	// to give a gentle emphasis that we are using second derivatives
	//
	// Err is output error of the entire neural net
	// Xn is the output vector on the n-th layer
	// Xnm1 is the output vector of the previous layer
	// Wn is the vector of weights of the n-th layer
	// Yn is the activation value of the n-th layer, i.e., the weighted sum of inputs BEFORE the squashing function is applied
	// F is the squashing function: Xn = F(Yn)
	// F' is the derivative of the squashing function
	//   Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2, i.e., the derivative can be calculated from the output, without knowledge of the input
	
	ASSERT( d2Err_wrt_dXn.size() == m_Neurons.size() );
	ASSERT( m_pPrevLayer != NULL );
	ASSERT( d2Err_wrt_dXnm1.size() == m_pPrevLayer->m_Neurons.size() );

	int ii, jj;
	UINT kk;
	int nIndex;
	double output;
	double dTemp;
		
	std::vector< double > d2Err_wrt_dYn( m_Neurons.size() );
	//
	// std::vector< double > d2Err_wrt_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	//////////////////////////////////////////////////
	//
	///// DESIGN TRADEOFF: REVIEW !!
	//
	// Note that the reasoning of this comment is identical to that in the NNLayer::Backpropagate() 
	// function, from which the instant BackpropagateSecondDerivatives() function is derived from
	// 
	// We would prefer (for ease of coding) to use STL vector for the array "d2Err_wrt_dWn", which is the 
	// second differential of the current pattern's error wrt weights in the layer.  However, for layers with
	// many weights, such as fully-connected layers, there are also many weights.  The STL vector
	// class's allocator is remarkably stupid when allocating large memory chunks, and causes a remarkable 
	// number of page faults, with a consequent slowing of the application's overall execution time.
	
	// To fix this, I tried using a plain-old C array, by new'ing the needed space from the heap, and 
	// delete[]'ing it at the end of the function.  However, this caused the same number of page-fault
	// errors, and did not improve performance.
	
	// So I tried a plain-old C array allocated on the stack (i.e., not the heap).  Of course I could not
	// write a statement like 
	//    double d2Err_wrt_dWn[ m_Weights.size() ];
	// since the compiler insists upon a compile-time known constant value for the size of the array.  
	// To avoid this requirement, I used the _alloca function, to allocate memory on the stack.
	// The downside of this is excessive stack usage, and there might be stack overflow probelms.  That's why
	// this comment is labeled "REVIEW"
	
	double* d2Err_wrt_dWn = (double*)( _alloca( sizeof(double) *  m_Weights.size() ) ); //alloca 分配栈内存，用完就自动释放
	
	for ( ii=0; ii<m_Weights.size(); ++ii )
	{
		d2Err_wrt_dWn[ ii ] =0.0;
	}

	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;

	
	// calculate d2Err_wrt_dYn = ( F'(Yn) )^2 * dErr_wrt_Xn (where dErr_wrt_Xn is actually a second derivative )
	
	for ( ii=0; ii<m_Neurons.size(); ++ii )
	{
		ASSERT( ii<d2Err_wrt_dYn.size() );
		ASSERT( ii<d2Err_wrt_dXn.size() );
		
		output = m_Neurons[ ii ]->output;
		
		if(output>=0)
		{
			dTemp=1;
		}
		else
		{
			dTemp=0;
		}


		//dTemp = DSIGMOID( output ) ;
		d2Err_wrt_dYn[ ii ] = d2Err_wrt_dXn[ ii ] * dTemp * dTemp;
	}
	
	// calculate d2Err_wrt_Wn = ( Xnm1 )^2 * d2Err_wrt_Yn (where dE2rr_wrt_Yn is actually a second derivative)
	// For each neuron in this layer, go through the list of connections from the prior layer, and
	// update the differential for the corresponding weight
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk = (*cit).NeuronIndex;
			if ( kk == ULONG_MAX )
			{
				output = 1.0;  // this is the bias connection; implied neuron output of "1"
			}
			else
			{
				ASSERT( kk<m_pPrevLayer->m_Neurons.size() );
				
				output = m_pPrevLayer->m_Neurons[ kk ]->output;
			}
			
			////////////	ASSERT( (*cit).WeightIndex < d2Err_wrt_dWn.size() );  // since after changing d2Err_wrt_dWn to a C-style array, the size() function this won't work
			ASSERT( ii<d2Err_wrt_dYn.size() );
			d2Err_wrt_dWn[ (*cit).WeightIndex ] += d2Err_wrt_dYn[ ii ] * output * output ;
		}
		
		ii++;
	}
	
	
	// calculate d2Err_wrt_Xnm1 = ( Wn )^2 * d2Err_wrt_dYn (where d2Err_wrt_dYn is a second derivative not a first).
	// d2Err_wrt_Xnm1 is needed as the input value of
	// d2Err_wrt_Xn for backpropagation of second derivatives for the next (i.e., previous spatially) layer
	// For each neuron in this layer
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk=(*cit).NeuronIndex;
			if ( kk != ULONG_MAX )
			{
				// we exclude ULONG_MAX, which signifies the phantom bias neuron with
				// constant output of "1", since we cannot train the bias neuron
				
				nIndex = kk;
				
				ASSERT( nIndex<d2Err_wrt_dXnm1.size() );
				ASSERT( ii<d2Err_wrt_dYn.size() );
				ASSERT( (*cit).WeightIndex<m_Weights.size() );
				
				dTemp = m_Weights[ (*cit).WeightIndex ]->value ; 
				
				d2Err_wrt_dXnm1[ nIndex ] += d2Err_wrt_dYn[ ii ] * dTemp * dTemp ;
			}
			
		}
		
		ii++;  // ii tracks the neuron iterator
		
	}
	
	struct DOUBLE_UNION
	{
		union 
		{
			double dd;
			unsigned __int64 ullong;
		};
	};
	
	DOUBLE_UNION oldValue, newValue;
	
	// finally, update the diagonal Hessians for the weights of this layer neuron using dErr_wrt_dW.
	// By design, this function (and its iteration over many (approx 500 patterns) is called while a 
	// single thread has locked the nueral network, so there is no possibility that another
	// thread might change the value of the Hessian.  Nevertheless, since it's easy to do, we
	// use an atomic compare-and-exchange operation, which means that another thread might be in 
	// the process of backpropagation of second derivatives and the Hessians might have shifted slightly
	
	for ( jj=0; jj<m_Weights.size(); ++jj )
	{
		oldValue.dd = m_Weights[ jj ]->diagHessian;
		newValue.dd = oldValue.dd + d2Err_wrt_dWn[ jj ];
		
		//*****************
		/*while ( oldValue.ullong != _InterlockedCompareExchange64( (unsigned __int64*)(&m_Weights[ jj ]->diagHessian), 
			newValue.ullong, oldValue.ullong ) ) */
		{
			// another thread must have modified the weight.  Obtain its new value, adjust it, and try again
			
			oldValue.dd = m_Weights[ jj ]->diagHessian;
			newValue.dd = oldValue.dd + d2Err_wrt_dWn[ jj ];
		}
		
	}
	
}


void NNLayer::Serialize(CArchive &ar)
{
	VectorNeurons::iterator nit;
	VectorWeights::iterator wit;
	VectorConnections::iterator cit;
	
	int ii, jj;
	
	if (ar.IsStoring())
	{
		// TODO: add storing code here
		
		ar.WriteString( label.c_str() );
		ar.WriteString( _T("\r\n") );  // ar.ReadString will look for \r\n when loading from the archive
		ar << m_Neurons.size();
		ar << m_Weights.size();
		
		
		
		for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
		{
			NNNeuron& n = *(*nit);
			ar.WriteString( n.label.c_str() );
			ar.WriteString( _T("\r\n") );
			ar << n.m_Connections.size();
			
			for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
			{
				ar << (*cit).NeuronIndex;
				ar << (*cit).WeightIndex;
			}
		}
		
		for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
		{
			ar.WriteString( (*wit)->label.c_str() );
			ar.WriteString( _T("\r\n") );
			ar << (*wit)->value;
		}
		
		
	}
	else
	{
		// TODO: add loading code here
		
		CString str;
		ar.ReadString( str );
		
		label = str;
		
		int iNumNeurons, iNumWeights, iNumConnections;
		double value;
		
		NNNeuron* pNeuron;
		NNWeight* pWeight;
		NNConnection conn;
		
		ar >> iNumNeurons;
		ar >> iNumWeights;
		
		for ( ii=0; ii<iNumNeurons; ++ii )
		{
			ar.ReadString( str );
			pNeuron = new NNNeuron( (LPCTSTR)str );
			m_Neurons.push_back( pNeuron );
			
			ar >> iNumConnections;
			
			for ( jj=0; jj<iNumConnections; ++jj )
			{
				ar >> conn.NeuronIndex;
				ar >> conn.WeightIndex;
				
				pNeuron->AddConnection( conn );
			}
		}
		
		for ( jj=0; jj<iNumWeights; ++jj )
		{
			ar.ReadString( str );
			ar >> value;
			
			pWeight = new NNWeight( (LPCTSTR)str, value );
			m_Weights.push_back( pWeight );
		}
		
	}
	
}



///////////////////////////////////////////////////////////////////////
//
//  NNWeight

NNWeight::NNWeight() : 
label( _T("") ),
value( 0.0 ), diagHessian( 0.0 )
{
	Initialize();
}

NNWeight::NNWeight( LPCTSTR str, double val /* =0.0 */ ) :
label( str ),
value( val ), diagHessian( 0.0 )
{
	Initialize();
}


void NNWeight::Initialize()
{
	
}

NNWeight::~NNWeight()
{
	
}








///////////////////////////////////////////////////////////////////////
//
//  NNNeuron


NNNeuron::NNNeuron() :
label( _T("") ), output( 0.0 )
{
	Initialize();
}

NNNeuron::NNNeuron( LPCTSTR str ) : 
label( str ), output( 0.0 )
{
	Initialize();
}


void NNNeuron::Initialize()
{
	m_Connections.clear();
}

NNNeuron::~NNNeuron()
{
	Initialize();
}


void NNNeuron::AddConnection( UINT iNeuron, UINT iWeight )
{
	m_Connections.push_back( NNConnection( iNeuron, iWeight ) );
}


void NNNeuron::AddConnection( NNConnection const & conn )
{
	m_Connections.push_back( conn );
}














