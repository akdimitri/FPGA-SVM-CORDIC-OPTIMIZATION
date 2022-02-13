#include <ap_fixed.h>
//#include <ap_int.h>
//#include <stdio.h>
//#include  <hls_math.h>
//#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/theta.h"
//#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/fx_theta.h"
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/classifier.h"
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/fx_alphas.h"
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/fx_svs.h"
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/fx_bias.h"
//#include "/home/e2/Desktop/mnist_classification/c_headers/ap_test_data.h"

void classifier(fx_svm x[IMG_SIZE], fx_sum *sum){

	static fx_sum bias = fx_bias[0];
	*sum = 0.0;		// the biggest sum value is less than 32768.multiple of 0.25
	fx_exp_value K;

SVM_PARALLEL1:
	for (int i=0; i<NSV; i++){
#pragma HLS LOOP_MERGE
#pragma HLS UNROLL factor=55

		fx_alpha alpha = fx_alphas[i];
		// rbf kernel
		fx_l2Squared l2Squared = 0.0;

		SVM_IMG1:
		for (int j=0; j<IMG_SIZE; j++)
		{
#pragma HLS UNROLL factor=8

			fx_svm _sv = fx_svs[i*IMG_SIZE + j];
		    fx_x _x = x[j];

		    l2Squared = l2Squared + ((fx_l2Squared) (_sv - _x) * (_sv - _x));
		    //printf("sv:%f x:%f L2:%f \n", _x.to_double(), _sv.to_double(), l2Squared);
		}

		K = my_exponential(-(l2Squared>>10));
		*sum += alpha * K;

		//printf("i:%d sum:%f alpha:%f K:%f\n", i, ((double)*sum), alpha.to_double(), K.to_double());
	}
	*sum = *sum + bias;
	//printf("sum:%f\n",((double)sum));
}

fx_exp_value my_exponential(fx_angle_rad a){
#pragma HLS INLINE region


	int k, k2;
	fx_exp_value tx, expo;
	fx_angle_rad z, r;
	ap_ufixed<16,16> FACTOR;
	ap_fixed<8,7> q;
	ap_ufixed<1,1> flag = 0;

	static ap_fixed<16,6> FX_THETA_HYPER[]={
			0.549306,
			0.255413,
			0.125657,
			0.062582,
			0.031260,
			0.015626,
	};



	if( a < -1){
		q = -a/((ap_fixed<16,1>)0.6931);
		r = +a+q*((ap_fixed<16,1>)0.6931);
		z = r;
		flag = 1;
		//printf("Q:%f R:%f \n", q.to_double(), r.to_double());
	}
	else{
		z = a;
	}

	FACTOR = 1;
	fx_exp_value x = 1.2051363;
	fx_exp_value y = 1.2051363;


	for (k=1,k2=4; k<CORDIC_HYPER_MAXITER;) {
		tx = x;
		if (z>=0) {
			x = x + (y>>FACTOR);
			y = y + (tx>>FACTOR);
			z = z - FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		else {

			x = x - (y>>FACTOR);
			y = y - (tx>>FACTOR);
			z = z + FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		if(k==k2)
			k2=k*3+1;
		else{
			k++;
			FACTOR = FACTOR<<1;
		}
		printf("IN EXPO X,Y %f,%f\n", x.to_double(),y.to_double());
		//printf("IN EXPO F %f\n", FACTOR.to_double());
	}

	//printf("IN EXPO %f\n", x.to_double());
	expo = x;
	if(flag == 1){


		expo = expo >> q;
		//printf("Q:%f \n", q.to_double());
	}

	//printf("EXPO: %f\n", expo.to_double());
	return expo;
}


/*

void classifier(fx_svm x1[IMG_SIZE], fx_sum *sum1, fx_svm x2[IMG_SIZE], fx_sum *sum2, fx_svm x3[IMG_SIZE], fx_sum *sum3){
#pragma HLS INTERFACE s_axilite port=sum3
#pragma HLS INTERFACE s_axilite port=x3
#pragma HLS INTERFACE s_axilite port=sum2
#pragma HLS INTERFACE s_axilite port=x2
#pragma HLS INTERFACE s_axilite port=sum1
#pragma HLS INTERFACE s_axilite port=x1

	fx_sum sum1_ = my_classifier1( x1);
	fx_sum sum2_ = my_classifier2( x2);
	fx_sum sum3_ = my_classifier3( x3);

	*sum1 = sum1_;
	*sum2 = sum2_;
	*sum3 = sum3_;
}

fx_sum my_classifier1(fx_svm x[IMG_SIZE]){
//#pragma HLS ARRAY_PARTITION variable=fx_alphas cyclic factor=55 dim=1
//#pragma HLS ARRAY_PARTITION variable=x cyclic factor=66 dim=1
	static fx_sum bias = fx_bias[0];
	fx_sum sum = 0.0;		// the biggest sum value is less than 32768.multiple of 0.25
	fx_exp_value K;

SVM_PARALLEL1:
	for (int i=0; i<NSV; i++){
#pragma HLS LOOP_MERGE
#pragma HLS UNROLL factor=55

		fx_alpha alpha = fx_alphas[i];
		// rbf kernel
		fx_l2Squared l2Squared = 0.0;

		SVM_IMG1:
		for (int j=0; j<IMG_SIZE; j++)
		{
#pragma HLS UNROLL factor=56

			fx_svm _sv = fx_svs[i*IMG_SIZE + j];
		    fx_x _x = x[j];

		    l2Squared = l2Squared + ((fx_l2Squared) (_sv - _x) * (_sv - _x));
		    //printf("sv:%f x:%f L2:%f \n", _x.to_double(), _sv.to_double(), l2Squared);
		}

		K = my_exponential1(-(l2Squared>>10));
		sum += alpha * K;

		//printf("i:%d sum:%f alpha:%f K:%f\n", i, ((double)*sum), alpha.to_double(), K.to_double());
	}
	sum = sum + bias;
	//printf("sum:%f\n",((double)sum));
	return sum;
}


fx_sum my_classifier2(fx_svm x[IMG_SIZE]){
//#pragma HLS ARRAY_PARTITION variable=fx_alphas cyclic factor=55 dim=1
//#pragma HLS ARRAY_PARTITION variable=x cyclic factor=66 dim=1
	static fx_sum bias = fx_bias[0];
	fx_sum sum = 0.0;		// the biggest sum value is less than 32768.multiple of 0.25
	fx_exp_value K;

SVM_PARALLEL2:
	for (int i=0; i<NSV; i++){
#pragma HLS LOOP_MERGE
#pragma HLS UNROLL factor=55

		fx_alpha alpha = fx_alphas[i];
		// rbf kernel
		fx_l2Squared l2Squared = 0.0;

		SVM_IMG2:
		for (int j=0; j<IMG_SIZE; j++)
		{
#pragma HLS UNROLL factor=56

			fx_svm _sv = fx_svs[i*IMG_SIZE + j];
		    fx_x _x = x[j];

		    l2Squared = l2Squared + ((fx_l2Squared) (_sv - _x) * (_sv - _x));
		    //printf("sv:%f x:%f L2:%f \n", _x.to_double(), _sv.to_double(), l2Squared);
		}

		K = my_exponential2(-(l2Squared>>10));
		sum += alpha * K;

		//printf("i:%d sum:%f alpha:%f K:%f\n", i, ((double)*sum), alpha.to_double(), K.to_double());
	}
	sum = sum + bias;
	//printf("sum:%f\n",((double)sum));
	return sum;
}

fx_sum my_classifier3(fx_svm x[IMG_SIZE]){
//#pragma HLS ARRAY_PARTITION variable=fx_alphas cyclic factor=55 dim=1
//#pragma HLS ARRAY_PARTITION variable=x cyclic factor=66 dim=1
	static fx_sum bias = fx_bias[0];
	fx_sum sum = 0.0;		// the biggest sum value is less than 32768.multiple of 0.25
	fx_exp_value K;

SVM_PARALLEL3:
	for (int i=0; i<NSV; i++){
#pragma HLS LOOP_MERGE
#pragma HLS UNROLL factor=55

		fx_alpha alpha = fx_alphas[i];
		// rbf kernel
		fx_l2Squared l2Squared = 0.0;

		SVM_IMG3:
		for (int j=0; j<IMG_SIZE; j++)
		{
#pragma HLS UNROLL factor=56

			fx_svm _sv = fx_svs[i*IMG_SIZE + j];
		    fx_x _x = x[j];

		    l2Squared = l2Squared + ((fx_l2Squared) (_sv - _x) * (_sv - _x));
		    //printf("sv:%f x:%f L2:%f \n", _x.to_double(), _sv.to_double(), l2Squared);
		}

		K = my_exponential3(-(l2Squared>>10));
		sum += alpha * K;

		//printf("i:%d sum:%f alpha:%f K:%f\n", i, ((double)*sum), alpha.to_double(), K.to_double());
	}
	sum = sum + bias;
	//printf("sum:%f\n",((double)sum));
	return sum;
}



fx_exp_value my_exponential1(fx_angle_rad a){
#pragma HLS INLINE region


	int k, k2;
	fx_exp_value tx, expo;
	fx_angle_rad z, r;
	ap_ufixed<16,16> FACTOR;
	ap_fixed<8,7> q;
	ap_ufixed<1,1> flag = 0;

	static ap_fixed<16,6> FX_THETA_HYPER[]={
	0.549306,
	0.255413,
	0.125657,
	0.062582,
	0.031260
	};



	if( a < -1){
		q = -a/((ap_fixed<16,1>)0.6931);
		r = +a+q*((ap_fixed<16,1>)0.6931);
		z = r;
		flag = 1;
		//printf("Q:%f R:%f \n", q.to_double(), r.to_double());
	}
	else{
		z = a;
	}

	FACTOR = 1;
	fx_exp_value x = 1.2051363;
	fx_exp_value y = 1.2051363;


	for (k=1,k2=4; k<CORDIC_HYPER_MAXITER;) {
		tx = x;
		if (z>=0) {
			x = x + (y>>FACTOR);
			y = y + (tx>>FACTOR);
			z = z - FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		else {

			x = x - (y>>FACTOR);
			y = y - (tx>>FACTOR);
			z = z + FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		if(k==k2)
			k2=k*3+1;
		else{
			k++;
			FACTOR = FACTOR<<1;
		}
		//printf("IN EXPO X,Y %f,%f\n", x.to_double(),y.to_double());
		//printf("IN EXPO F %f\n", FACTOR.to_double());
	}

	//printf("IN EXPO %f\n", x.to_double());
	expo = x;
	if(flag == 1){


		expo = expo >> q;
		//printf("Q:%f \n", q.to_double());
	}

	return expo;
}


fx_exp_value my_exponential2(fx_angle_rad a){
#pragma HLS INLINE region


	int k, k2;
	fx_exp_value tx, expo;
	fx_angle_rad z, r;
	ap_ufixed<16,16> FACTOR;
	ap_fixed<8,7> q;
	ap_ufixed<1,1> flag = 0;

	static ap_fixed<16,6> FX_THETA_HYPER[]={
	0.549306,
	0.255413,
	0.125657,
	0.062582,
	0.031260
	};



	if( a < -1){
		q = -a/((ap_fixed<16,1>)0.6931);
		r = +a+q*((ap_fixed<16,1>)0.6931);
		z = r;
		flag = 1;
		//printf("Q:%f R:%f \n", q.to_double(), r.to_double());
	}
	else{
		z = a;
	}

	FACTOR = 1;
	fx_exp_value x = 1.2051363;
	fx_exp_value y = 1.2051363;


	for (k=1,k2=4; k<CORDIC_HYPER_MAXITER;) {
		tx = x;
		if (z>=0) {
			x = x + (y>>FACTOR);
			y = y + (tx>>FACTOR);
			z = z - FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		else {

			x = x - (y>>FACTOR);
			y = y - (tx>>FACTOR);
			z = z + FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		if(k==k2)
			k2=k*3+1;
		else{
			k++;
			FACTOR = FACTOR<<1;
		}
		//printf("IN EXPO X,Y %f,%f\n", x.to_double(),y.to_double());
		//printf("IN EXPO F %f\n", FACTOR.to_double());
	}

	//printf("IN EXPO %f\n", x.to_double());
	expo = x;
	if(flag == 1){


		expo = expo >> q;
		//printf("Q:%f \n", q.to_double());
	}

	return expo;
}


fx_exp_value my_exponential3(fx_angle_rad a){
#pragma HLS INLINE region


	int k, k2;
	fx_exp_value tx, expo;
	fx_angle_rad z, r;
	ap_ufixed<16,16> FACTOR;
	ap_fixed<8,7> q;
	ap_ufixed<1,1> flag = 0;

	static ap_fixed<16,6> FX_THETA_HYPER[]={
	0.549306,
	0.255413,
	0.125657,
	0.062582,
	0.031260
	};



	if( a < -1){
		q = -a/((ap_fixed<16,1>)0.6931);
		r = +a+q*((ap_fixed<16,1>)0.6931);
		z = r;
		flag = 1;
		//printf("Q:%f R:%f \n", q.to_double(), r.to_double());
	}
	else{
		z = a;
	}

	FACTOR = 1;
	fx_exp_value x = 1.2051363;
	fx_exp_value y = 1.2051363;


	for (k=1,k2=4; k<CORDIC_HYPER_MAXITER;) {
		tx = x;
		if (z>=0) {
			x = x + (y>>FACTOR);
			y = y + (tx>>FACTOR);
			z = z - FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		else {

			x = x - (y>>FACTOR);
			y = y - (tx>>FACTOR);
			z = z + FX_THETA_HYPER[k-1];
			//printf("IN EXPO X,Y,Z %f,%f,%f\n", x.to_double(),y.to_double(), z.to_double());
			//printf("IN EXPO F %f\n", FACTOR.to_double());
		}
		if(k==k2)
			k2=k*3+1;
		else{
			k++;
			FACTOR = FACTOR<<1;
		}
		//printf("IN EXPO X,Y %f,%f\n", x.to_double(),y.to_double());
		//printf("IN EXPO F %f\n", FACTOR.to_double());
	}

	//printf("IN EXPO %f\n", x.to_double());
	expo = x;
	if(flag == 1){


		expo = expo >> q;
		//printf("Q:%f \n", q.to_double());
	}

	return expo;
}

*/
