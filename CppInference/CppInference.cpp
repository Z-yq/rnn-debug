#include <fstream>
#include <iostream>
#include "AM.h"
#include "ext/AudioFile.hpp"
#include "ext/Tokens.hpp"
#include  <time.h>

int main()
{
	/* Demo For RNNT Structure */

	//Init
	AM am ;
	Tokener AM_Token;

	const char* ampath =  "with_state" ;
	
	
	AM_Token.load_token("./tokens/am_tokens.txt");


	am.Initialize(ampath);


	std::cout << "Hello TensorflowASR!\n";

	//Read Wav File
	std::cout << "Read File Now...\n";
	std::string wav_path = "./test.wav";
	AudioFile<float> audioFile;
	audioFile.load(wav_path);
	int channel = 0;
	int numSamples = audioFile.getNumSamplesPerChannel();
	std::cout << "Bit Depth: " << audioFile.getBitDepth() << std::endl;
	std::cout << "Sample Rate: " << audioFile.getSampleRate() << std::endl;
	std::cout << "Num Channels: " << audioFile.getNumChannels() << std::endl;
	std::cout << "Length in Seconds: " << audioFile.getLengthInSeconds() << std::endl;
	int all_length = audioFile.getLengthInSeconds() * audioFile.getSampleRate();

	// Prepare AM inputs
	
    std::vector<float>wav_in;
	for (int i = 0; i < all_length; i++)
	{
		wav_in.push_back(audioFile.samples[channel][i]);
	}

	
	// Do AM  session run
	
	TFTensor<int32_t> am_out =am.DoInference(wav_in);

	//get am result to string
	//here 'blank_at_zero=False' in am_data.yml
	std::vector<std::string>am_result;
	
	for (int i = 0; i < am_out.Data.size(); i++)
	{
		int32_t key = am_out.Data[i];
		am_result.push_back(AM_Token.id_to_token[key]);
	
	}

	
	//show result
	
	std::cout << "the AM result:\n";
	for (int i = 0; i < am_result.size(); i++)
	{
		std::cout << am_result[i] << ' ';
	}
	
	std::cout << "\n";
	
	
	return 0;

}
