#pragma once
#include <vector>

class CCalculateNetwork
{
public:
	CCalculateNetwork(void);


	void CalculateNetwork(void);

	std::vector< VectorDoubles > m_NeuronOutputs;

protected:
	NeuralNetwork m_NN;

public:

	~CCalculateNetwork(void);
};

