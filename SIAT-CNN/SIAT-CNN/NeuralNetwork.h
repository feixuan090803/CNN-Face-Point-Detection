// NeuralNetwork.h: interface for the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
#define AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <math.h>
#include <vector>
#include "afx.h"



///////////////////////////////////////////////////////////////////////
//
// global function to support 64-bit atomic compare-and-exchange
// Probably will work only on Intel and AMD products, and no others
// Needed since the windows API provides an InterlockedCompareExchange64 function only for 
// "Vista" and higher, and because I do not have access to the VC++ 2005 compiler
// intrinsic _InterlockedCompareExchange64.
// See my newsgroup post:
// "InterlockedCompareExchange64 under VC++ 6.0: in-line assembly using cmpxchg8b ??"
// at:
// http://groups.google.com/group/comp.programming.threads/browse_thread/thread/1c3b38cd249ff2ba/e90ff2c919f84612
//
// The following code was obtained from:
// http://www.audiomulch.com/~rossb/code/lockfree/ATOMIC.H

#pragma warning(push)
#pragma warning(disable : 4035) // disable no-return warning

////////////////////////////////////////////////////////////////////////////////////////////
//	volatile类型修饰符表明dest变量可能由于程序使用了多线程的原因，从而会在不知情情况下而改变
//	因此必须每次读取该变量在内存中的真实值
////////////////////////////////////////////////////////////////////////////////////////////
inline unsigned __int64 
_InterlockedCompareExchange64(volatile unsigned __int64 *dest,   
                           unsigned __int64 exchange,
                           unsigned __int64 comparand)  
{
    //value returned in eax::edx
    __asm {
        lea esi,comparand;
        lea edi,exchange;
        
        mov eax,[esi];
        mov edx,4[esi];
        mov ebx,[edi];
        mov ecx,4[edi];
        mov esi,dest;
        //lock CMPXCHG8B [esi] is equivalent to the following except
        //that it's atomic:
        //ZeroFlag = (edx:eax == *esi);
        //if (ZeroFlag) *esi = ecx:ebx;
        //else edx:eax = *esi;
        lock CMPXCHG8B [esi];			
    }
}
#pragma warning(pop)



using namespace std;

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))  // derivative of the sigmoid as a function of the sigmoid's output



// forward declarations

class NNLayer;
class NNWeight;
class NNNeuron;
class NNConnection;




// helpful typedef's

typedef std::vector< NNLayer* >  VectorLayers;
typedef std::vector< NNWeight* >  VectorWeights;
typedef std::vector< NNNeuron* >  VectorNeurons;
typedef std::vector< NNConnection > VectorConnections;
typedef std::basic_string<TCHAR>  tstring;


class NeuralNetwork  
{
public:
	volatile double m_etaLearningRatePrevious;
	volatile double m_etaLearningRate;

	volatile UINT m_cBackprops;  // counter used in connection with Weight sanity check
	void PeriodicWeightSanityCheck();

	void Calculate(double* inputVector, UINT count, 
		double* outputVector = NULL, UINT oCount = 0,
		std::vector< std::vector< double > >* pNeuronOutputs = NULL );

	void Backpropagate(double *actualOutput, double *desiredOutput, UINT count,
		std::vector< std::vector< double > >* pMemorizedNeuronOutputs );

	void EraseHessianInformation();
	void DivideHessianInformationBy( double divisor );
	void BackpropagateSecondDervatives( double* actualOutputVector, double* targetOutputVector, UINT count );

	void Serialize(CArchive &ar);

	NeuralNetwork();
	virtual ~NeuralNetwork();
	void Initialize();

	VectorLayers m_Layers;


	//void AveragedPooling(void);
private:
	// 网络权重文件保存路径
	string m_sWeightSavePath;
	// 权重文件句柄
	CStdioFile m_fileWeight;
};

class NNLayer
{
public:

	void PeriodicWeightSanityCheck();  // check if weights are "reasonable"
	
	//卷积层计算卷积
	void Calculate();

	//池化层进行maxpooling操作
	void AveragedPooling();

	void Backpropagate( std::vector< double >& dErr_wrt_dXn /* in */, 
		std::vector< double >& dErr_wrt_dXnm1 /* out */, 
		std::vector< double >* thisLayerOutput,  // memorized values of this layer's output
		std::vector< double >* prevLayerOutput,  // memorized values of previous layer's output
		double etaLearningRate );

	void Backpropagate2( std::vector< double >& dErr_wrt_dXn /* in */, 
		std::vector< double >& dErr_wrt_dXnm1 /* out */, 
		std::vector< double >* thisLayerOutput,  // memorized values of this layer's output
		std::vector< double >* prevLayerOutput,  // memorized values of previous layer's output
		double etaLearningRate );

	void EraseHessianInformation();
	void DivideHessianInformationBy( double divisor );
	void BackpropagateSecondDerivatives( std::vector< double >& dErr_wrt_dXn /* in */, 
		std::vector< double >& dErr_wrt_dXnm1 /* out */);

	void Serialize(CArchive& ar );

	NNLayer();
	NNLayer( LPCTSTR str, NNLayer* pPrev = NULL );
	virtual ~NNLayer();


	VectorWeights m_Weights;
	VectorNeurons m_Neurons;

	tstring label;
	NNLayer* m_pPrevLayer;

	bool m_bFloatingPointWarning;  // flag for one-time warning (per layer) about potential floating point overflow

protected:
	void Initialize();

};




class NNConnection
{
public: 
	NNConnection(UINT neuron = ULONG_MAX, UINT weight = ULONG_MAX):NeuronIndex( neuron ), WeightIndex( weight ) {};
	virtual ~NNConnection() {};
	UINT NeuronIndex, WeightIndex;
};




class NNWeight
{
public:
	NNWeight();
	NNWeight( LPCTSTR str, double val = 0.0 );
	virtual ~NNWeight();

	tstring label;
	double value;
	double diagHessian;


protected:
	void Initialize();

};


class NNNeuron
{
public:
	NNNeuron();
	NNNeuron( LPCTSTR str );
	virtual ~NNNeuron();

	void AddConnection( UINT iNeuron, UINT iWeight );
	void AddConnection( NNConnection const & conn );


	tstring label;
	double output;

	VectorConnections m_Connections;

///	VectorWeights m_Weights;
///	VectorNeurons m_Neurons;
	
protected:
	void Initialize();

};


#endif // !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
