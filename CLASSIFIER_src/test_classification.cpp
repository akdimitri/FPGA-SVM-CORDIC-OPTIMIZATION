#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <iostream>
#include <ap_fixed.h>

#include "svs.h"
#include "bias.h"
#include "alphas.h"
#include "test_data.h"
#include "ground_truth.h"
#include "/home/e2/Workspace/Coursework/CLASSIFIER/CLASSIFIER/src/classifier.h"




#define IMG_SIZE 784 // Each image is 28x28
#define NUM_IMGS 10

#define NSV 165
double classify(double x[IMG_SIZE]);
void bin(ap_fixed<16,4> n);


int main()
{
    // for debugging
    std::ofstream scoresF;
    scoresF.open("scores.txt");
    std::ofstream my_scoresF;
    my_scoresF.open("my_scores.txt");

    double x[IMG_SIZE];
    //ap_fixed<8,7> my_x1[IMG_SIZE], my_x2[IMG_SIZE], my_x3[IMG_SIZE];
    ap_fixed<8,7> my_x[IMG_SIZE];
    double scores[NUM_IMGS];
    fx_sum my_scores[NUM_IMGS];
    for (int i=0; i<NUM_IMGS; i++)
    {
        // form input vector x
        for (int j=0; j<IMG_SIZE; j++){
            x[j] = test_data[i*IMG_SIZE+j];
            my_x[j] = ((ap_fixed<8,7>)test_data[i*IMG_SIZE+j]);
        }

        // call the function
        scores[i] = classify(x);
        //printf("My CLassifier:%d\n", i);
        classifier( my_x, &my_scores[i]);
        printf("my_scores[%d]=%f\n", i, my_scores[i].to_double());
        //bin(my_scores[i]);
        printf("\n");
        
        // store scores to file for debugging
        scoresF << scores[i] << std::endl;
        my_scoresF << my_scores[i] << std::endl;
    }

    /*
    for(int i=0; i<NUM_IMGS; i=i+3){
    	for (int j=0; j<IMG_SIZE; j++){
    		my_x1[j] = ((ap_fixed<8,7>)test_data[i*IMG_SIZE+j]);
    		my_x2[j] = ((ap_fixed<8,7>)test_data[(i+1)*IMG_SIZE+j]);
    		my_x3[j] = ((ap_fixed<8,7>)test_data[(i+2)*IMG_SIZE+j]);
    	}
    	classifier( my_x1, &my_scores[i], my_x2, &my_scores[i+1], my_x3, &my_scores[i+2]);

    	my_scoresF << my_scores[i] << std::endl;
    	my_scoresF << my_scores[i+1] << std::endl;
    	my_scoresF << my_scores[i+2] << std::endl;
    }
	*/

    scoresF.close();
    my_scoresF.close();

    // get predictions --> this takes the sign() of the output
    int predictions[NUM_IMGS];
    int my_predictions[NUM_IMGS];
    for (int i=0; i<NUM_IMGS; i++)
    {
        // classifying between 0 and 1
        if (scores[i] < 0)
            predictions[i] = 0;
        else
        	predictions[i] = 1;

        if (my_scores[i] < 0)
            my_predictions[i] = 0;
        else
            my_predictions[i] = 1;
    }

    // summary statistics
    double accuracy = 0.0;
    double my_accuracy = 0.0;
    int correct = 0;
    int my_correct = 0;
    for (int i=0; i<NUM_IMGS; i++){
        if (predictions[i] == ground_truth[i])
            correct++;
        if (my_predictions[i] == ground_truth[i])
            my_correct++;
    }
    accuracy = correct/double(NUM_IMGS);
    my_accuracy = my_correct/double(NUM_IMGS);
    printf("Classification Accuracy: %f\n", accuracy);
    printf("My Classification Accuracy: %f\n", my_accuracy);

    // summary statistics - confusion matrix
    double CM[2][2];
    double my_CM[2][2];
    for (int i=0; i<2; i++)
        for (int j=0; j<2; j++){
            CM[i][j]=0.0;
            my_CM[i][j]=0.0;
        }

    for (int i=0; i<NUM_IMGS; i++){
            CM[ground_truth[i]][predictions[i]]++;
            my_CM[ground_truth[i]][my_predictions[i]]++;
    }
    printf("Confusion Matrix (%d test points):\n", NUM_IMGS);
    printf("%f, %f\n", CM[0][0]/NUM_IMGS, CM[0][1]/NUM_IMGS);
    printf("%f, %f\n", CM[1][0]/NUM_IMGS, CM[1][1]/NUM_IMGS);

    printf("My Confusion Matrix (%d test points):\n", NUM_IMGS);
    printf("%f, %f\n", my_CM[0][0]/NUM_IMGS, my_CM[0][1]/NUM_IMGS);
    printf("%f, %f\n", my_CM[1][0]/NUM_IMGS, my_CM[1][1]/NUM_IMGS);
}

double classify(double x[IMG_SIZE])
{
    double sum = 0.0;
    for (int i=0; i<NSV; i++)
    {
        double alpha = alphas[i];
        
        // rbf kernel 
        double l2Squared = 0.0;
        for (int j=0; j<IMG_SIZE; j++)
        {
            double _sv = svs[i*IMG_SIZE + j];
            double _x = x[j];
            l2Squared += (_sv - _x) * (_sv - _x);
        }
        double K = exp(-0.001 * l2Squared); 

        sum += alpha * K;
    }
    sum = sum + bias[0];
    return sum;
}

void bin(ap_fixed<16,4> n)
{
    unsigned i;
    for (i = 1 << 15; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}
