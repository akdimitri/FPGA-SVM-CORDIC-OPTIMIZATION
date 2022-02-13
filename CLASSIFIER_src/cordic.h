#ifndef CORDIC_H_
#define CORDIC_H_
#include <ap_fixed.h>

#define NUM_ITERATIONS 5
#define CORDIC_HYPER_MAXITER 4
typedef float	COS_SIN_TYPE;
typedef float	THETA_TYPE;
typedef ap_fixed<5,2> 	ANGLE_RAD;
typedef ap_fixed<1,16> 	EXP_VALUE;

void my_exponential(ANGLE_RAD a, EXP_VALUE *exp);

#endif

