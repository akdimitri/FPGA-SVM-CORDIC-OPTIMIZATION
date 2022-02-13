#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#define IMG_SIZE 784 // Each image is 28x28
#define NSV 165
#define CORDIC_HYPER_MAXITER 15

typedef 	ap_fixed<16,4> 	fx_sum;
typedef 	ap_fixed<8,7> 	fx_svm;
typedef 	ap_fixed<8,7> 	fx_x;
typedef		ap_fixed<24,16> fx_l2Squared;
typedef		ap_fixed<8,5> 	fx_alpha;
typedef 	ap_fixed<16,6> 	fx_angle_rad;
typedef 	ap_fixed<16,2> 	fx_exp_value;

fx_exp_value my_exponential(fx_angle_rad a);
void classifier(fx_svm x[IMG_SIZE], fx_sum *sum);

/*
fx_exp_value my_exponential1(fx_angle_rad a);
fx_exp_value my_exponential2(fx_angle_rad a);
fx_exp_value my_exponential3(fx_angle_rad a);
fx_sum my_classifier1(fx_svm x[IMG_SIZE]);
fx_sum my_classifier2(fx_svm x[IMG_SIZE]);
fx_sum my_classifier3(fx_svm x[IMG_SIZE]);

void classifier(fx_svm x1[IMG_SIZE], fx_sum *sum1, fx_svm x2[IMG_SIZE], fx_sum *sum2, fx_svm x3[IMG_SIZE], fx_sum *sum3);
*/
#endif
