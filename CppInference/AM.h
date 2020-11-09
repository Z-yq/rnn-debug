#include "VoxCommon.hpp"
#include "ext/CppFlow/include/Model.h"


class AM
{
private:
	Model* AMModel;


public:
	
	bool Initialize(const char* ModelPath);
	
	
	TFTensor<int32_t> DoInference(const std::vector<float>& InWav);
	
	AM();
	~AM();
};

