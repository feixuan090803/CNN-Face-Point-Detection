// disable the template warning C4786 : identifier was truncated to '255' characters in the browser information

#pragma warning( push )
#pragma warning( disable : 4786 )


#include "NeuralNetwork.h"	


using namespace std;

#include <vector>
#include <afxmt.h>  // for critical section, multi-threaded etc

typedef std::vector< double >  VectorDoubles;

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CCreateNetwork
{
public:
	CCreateNetwork(void);

// Implementation
// 可能不需要进行乱序输入处理
public:

	void ApplyDistortionMap( double* inputVector );
	//void GenerateDistortionMap( double severityFactor = 1.0 );
	double* m_DispH;  // horiz distortion map array
	double* m_DispV;  // vert distortion map array

	//double m_GaussianKernel[ GAUSSIAN_FIELD_SIZE ] [ GAUSSIAN_FIELD_SIZE ];

	int m_cCols;  // size of the distortion maps
	int m_cRows;
	int m_cCount;

	CStdioFile m_fileTrainingLabels; //训练标签信息文本
	CFile m_fileTrainingImages; //训练集数据
	CStdioFile m_fileTestingLabels;  //测试标签信息文本
	CFile m_fileTestingImages;	//测试集数据

	CStdioFile m_fileWeight; 

	BOOL bTraining;
private:
	CString m_TrainFilePath;
	CString m_TestFilePath;

	CString m_TrainInfoPath;

	CString m_TrainImagePath;
	

public:
	//inline double& At( double* p, int row, int col )  // zero-based indices, starting at bottom-left
	//	{ int location = row * m_cCols + col;
	//	  ASSERT( location>=0 && location<m_cCount && row<m_cRows && row>=0 && col<m_cCols && col>=0 );
	//	  return p[ location ];
	//	}

	double GetCurrentEta();
	double GetPreviousEta();
	bool BackpropagateNeuralNet(double *inputVector, int iCount, double* targetOutputVector, 
		double* actualOutputVector, int oCount, 
		std::vector< std::vector< double > >* pMemorizedNeuronOutputs, 
		BOOL bDistort );	
	void CalculateNeuralNet(double* inputVector, int count, double* outputVector = NULL, 
		int oCount = 0, std::vector< std::vector< double > >* pNeuronOutputs = NULL, BOOL bDistort = FALSE );
	void BackPropagation(void);

	vector< VectorDoubles > m_NeuronOutputs;

private:
	//保存训练标签目录下的所有标签文件名以及文件路径
	static vector<CString> m_InfoFileNames;
	static vector<CString> m_InfoFilePaths;
	static vector<CString> m_ImageFilePaths;

public:

	// backpropagation and training-related members
	volatile UINT m_cBackprops;

	// volatile BOOL m_bNeedHessian;  不需要计算hessian矩阵
	
	// HWND m_hWndForBackpropPosting;
	UINT m_nAfterEveryNBackprops;
	double m_dEtaDecay;
	double m_dMinimumEta;
	volatile double m_dEstimatedCurrentMSE;  // this number will be changed by one thread and used by others
	
	//移植了对话框的参数设置变量
	double m_InitialEta;
	double m_MinimumEta;
	double m_EtaDecay;
	UINT m_AfterEvery;
	UINT m_StartingPattern;
	double m_EstimatedCurrentMSE;
	BOOL m_bDistortPatterns;

	double m_dMSE;
	//UINT m_cMisrecognitions;
	DWORD m_dwEpochStartTime;
	UINT m_iEpochsCompleted;
	UINT m_iBackpropsPosted;

	BOOL m_bDistortTrainingPatterns;  //不需要乱序输入

	BOOL StartBackpropagation(UINT iStartPattern = 0, 
			double initialEta = 0.005, double minimumEta = 0.000001, double etaDecay = 0.990,
			UINT nAfterEvery = 1000, BOOL bDistortPatterns = TRUE, double estimatedCurrentMSE = 1.0 );
	
	//用于标识训练数据集的输入索引
	volatile UINT m_iNextTrainingPattern;
	volatile UINT m_iRandomizedTrainingPatternSequence[ 10000 ];  //训练集数目为10000

	void RandomizeTrainingPatternSequence();
	UINT GetCurrentTrainingPatternNumber( BOOL bFromRandomizedPatternSequence = FALSE );
	void GetTrainingPatternArrayValues( int iNumImage = 0,unsigned char* pArray = NULL,BOOL bFlipGrayscale =FALSE );

	/*UINT GetNextTrainingPattern(unsigned char* pArray = NULL, int* pLabel = NULL, BOOL bFlipGrayscale = TRUE,
		BOOL bFromRandomizedPatternSequence = TRUE, UINT* iSequenceNum = NULL );*/
	UINT GetNextTrainingPattern(unsigned char* pArray = NULL, double* pPointLocation = NULL, BOOL bFlipGrayscale = FALSE,
		BOOL bFromRandomizedPatternSequence = FALSE, UINT* iSequenceNum = NULL );
	UINT GetRandomTrainingPattern(unsigned char* pArray=NULL, int* pLabel=NULL, BOOL bFlipGrayscale=FALSE);

	// testing-related members
	volatile UINT m_iNextTestingPattern;
	
#ifdef _DEBUG
	//virtual void AssertValid() const;
	//virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	NeuralNetwork m_NN;

public:
	~CCreateNetwork(void);

	// 初始化网络结构
	BOOL InitNetwork(bool bLoadWeightFile);

	// 读取训练/测试的图片，准备网络的输入向量
	bool ForwardPropagation(void);
	// 初始化BP过程的相关参数
	void PreBackPropagaton(void);
	
private:
	// 权重文件保存路径
	string m_sWeightSavePath;
public:
	// 依次加载权值文件
	void LoadWeightFile(vector<double>* vWeightOfLayer, char* path);
private:
	// 调用Opencv，实时在人脸标记出检测到的特征点
	void DisplayPointDetected(CString csImageFilePath, double* dPointDetected);
	// BP完成退出训练前保存当前网络的权重值
	void SaveLastWeight(void);
};

// re-enable warning C4786 re : identifier was truncated to '255' characters in the browser information

#pragma warning( pop )