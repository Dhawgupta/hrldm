
#include <iostream>
#include <string.h>
#include <cstdlib>
#include <pthread.h>
#include <stdlib.h>


using namespace std;
class Parameter{
	public:
	int intent;
	long long int episodes;
	int gpu_core; // this contains the ifromation to run teh gpu core
	bool run_on_gpu;
	// this will tell wether to rin teh code on gpu or not
};


void *runIntentTrain(void *params){
	//long tid;
	string command = "";
	Parameter *param = (Parameter *)params;
	if (param->run_on_gpu){
		command = "CUDA_VISIBLE_DEVICES=" + to_string(param->gpu_core) + " ";
	}

	command = command + "python train_general_intent.py " + to_string(param->intent) + " " + to_string(param->episodes);
	
	//tid = (long)threadid;
	// now we need to convert the command to const char * to pass to systme
	const char *cstr = command.c_str();
	int run = system(cstr);
	pthread_exit(NULL);
}
	
int main(int argc, char ** argv){
	//int result = system("python train_general_intent.py 3 1"); // this command is able to rin a program
 	pthread_t threads[5];
 	int rc;
	int i;
	for(i =0;i<5;i++){
		cout<<"Main Thread : Starting thread for "<<i<<endl;
		Parameter params;
		params.episodes= 100000; // run for 1lakh epsiode
		params.intent = i;
		params.run_on_gpu = true;
		params.gpu_core = stoi(argv[1]);
		rc = pthread_create(&threads[i], NULL, runIntentTrain, (void *)&params);

		if (rc){
			cout<<"Error Not able to start thrrread"<<rc<<endl;
			exit(-1);
		}
	}
	pthread_exit(NULL);

	
}


