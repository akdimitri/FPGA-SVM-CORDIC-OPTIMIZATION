#include <ap_fixed.h>
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/classifier.h"
#include <stdio.h>


int main(){


	double a = -1.5;
	double expo;
	my_exponential_2( a, &expo);
	printf("EXP(%f)=%f\n", a, expo);

	fx_angle_rad a2 = -1.5;
	fx_exp_value expo2;
	expo2 = my_exponential(a2);
	printf("EXP(%f)=%f\n", a2.to_double(), expo2.to_double());



	return 0;
}
