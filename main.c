#include <stdio.h>
#include <stdlib.h>
//#include <list>
//#include <cstdlib>
#include <math.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define lr 0.1f
#define trainingTimes 10000
#define numTrainingSets 4

// Simple network that can learn XOR
// Feartures : sigmoid activation function, stochastic gradient descent, and mean square error fuction

// Potential improvements :
// Different activation functions
// Batch training
// Different error funnctions
// Arbitrary number of hidden layers
// Read training end test data from a file
// Add visualization of training
// Add recurrence? (maybe that should be a separate project)

//function prototypes
double sigmoid(double x) { return 1 / (1 + exp(-x)); }//to get a number between [0,1]
double dSigmoid(double x) { return x * (1 - x); }//derivative of sigmoid function
// dsigmoid < 0 when x > 1, dsigmoid > 0 when x < 1
double init_weight() { return ((double)rand())/((double)RAND_MAX); } //random floating number between 0 and 1
void shuffle(int *, size_t );
void result(double**,double*,double**,double*);
//end function prototypes

double* level_1_array(int a)
{
    return (double* ) malloc (a * sizeof(double*));

}
double** malloc_lv2_array(int a)
{
     return (double** ) malloc (a * sizeof(void* ));
}
void level_2_array (double** ptr,int row,int colume)
{
    for(int cnt=0; cnt < row; ++cnt)
    {
        *( ptr+cnt ) = (double* )malloc(colume * sizeof(double* ));

    }

}

int main() {
    int *trainingSetOrder = (int*) malloc( numTrainingSets * sizeof(int) );

    //double hiddenLayer[numHiddenNodes];
    double* hiddenLayer = level_1_array(numHiddenNodes);

    //double outputLayer[numOutputs];
    double* outputLayer =  level_1_array(numOutputs);

    //double hiddenLayerBias[numHiddenNodes];
    double* hiddenLayerBias = level_1_array(numHiddenNodes);

    //double outputLayerBias[numOutputs];
    double* outputLayerBias =  level_1_array(numOutputs);

    //double hiddenWeights[numInputs][numHiddenNodes];
    double** hiddenWeights = malloc_lv2_array(numInputs);
    level_2_array (hiddenWeights, numInputs, numHiddenNodes);

    //double outputWeights[numHiddenNodes][numOutputs];
    double** outputWeights = malloc_lv2_array(numOutputs);
    level_2_array (outputWeights, numHiddenNodes, numOutputs);

    //double training_inputs[numTrainingSets][numInputs];
    double** training_inputs = malloc_lv2_array(numTrainingSets);
    level_2_array (training_inputs, numTrainingSets, numInputs);

    //double training_outputs[numTrainingSets][numOutputs];
    double** training_outputs = malloc_lv2_array(numTrainingSets);
    level_2_array (training_outputs, numTrainingSets, numOutputs);


     //initialize training inputs
    *( *( training_inputs + 0) + 0) = 0.0f;
    *( *( training_inputs + 0) + 1) = 0.0f;
    *( *( training_inputs + 1) + 0) = 1.0f;
    *( *( training_inputs + 1) + 1) = 0.0f;
    *( *( training_inputs + 2) + 0) = 0.0f;
    *( *( training_inputs + 2) + 1) = 1.0f;
    *( *( training_inputs + 3) + 0) = 1.0f;
    *( *( training_inputs + 3) + 1) = 1.0f;

    //training sets for 4 bits input
/*
    *( *( training_inputs + 0) + 0) = 0.0f;
    *( *( training_inputs + 0) + 1) = 0.0f;
    *( *( training_inputs + 0) + 2) = 0.0f;
    *( *( training_inputs + 0) + 3) = 0.0f;

    *( *( training_inputs + 1) + 0) = 0.0f;
    *( *( training_inputs + 1) + 1) = 0.0f;
    *( *( training_inputs + 1) + 2) = 0.0f;
    *( *( training_inputs + 1) + 3) = 1.0f;

    *( *( training_inputs + 2) + 0) = 0.0f;
    *( *( training_inputs + 2) + 1) = 0.0f;
    *( *( training_inputs + 2) + 2) = 1.0f;
    *( *( training_inputs + 2) + 3) = 0.0f;

    *( *( training_inputs + 3) + 0) = 0.0f;
    *( *( training_inputs + 3) + 1) = 0.0f;
    *( *( training_inputs + 3) + 2) = 1.0f;
    *( *( training_inputs + 3) + 3) = 1.0f;

    *( *( training_inputs + 4) + 0) = 0.0f;
    *( *( training_inputs + 4) + 1) = 1.0f;
    *( *( training_inputs + 4) + 2) = 0.0f;
    *( *( training_inputs + 4) + 3) = 0.0f;

    *( *( training_inputs + 5) + 0) = 0.0f;
    *( *( training_inputs + 5) + 1) = 1.0f;
    *( *( training_inputs + 5) + 2) = 0.0f;
    *( *( training_inputs + 5) + 3) = 1.0f;

    *( *( training_inputs + 6) + 0) = 0.0f;
    *( *( training_inputs + 6) + 1) = 1.0f;
    *( *( training_inputs + 6) + 2) = 1.0f;
    *( *( training_inputs + 6) + 3) = 0.0f;

    *( *( training_inputs + 7) + 0) = 0.0f;
    *( *( training_inputs + 7) + 1) = 1.0f;
    *( *( training_inputs + 7) + 2) = 1.0f;
    *( *( training_inputs + 7) + 3) = 1.0f;

    *( *( training_inputs + 8) + 0) = 1.0f;
    *( *( training_inputs + 8) + 1) = 0.0f;
    *( *( training_inputs + 8) + 2) = 0.0f;
    *( *( training_inputs + 8) + 3) = 0.0f;

    *( *( training_inputs + 9) + 0) = 1.0f;
    *( *( training_inputs + 9) + 1) = 0.0f;
    *( *( training_inputs + 9) + 2) = 0.0f;
    *( *( training_inputs + 9) + 3) = 1.0f;

    *( *( training_inputs + 10) + 0) = 1.0f;
    *( *( training_inputs + 10) + 1) = 0.0f;
    *( *( training_inputs + 10) + 2) = 1.0f;
    *( *( training_inputs + 10) + 3) = 0.0f;

    *( *( training_inputs + 11) + 0) = 1.0f;
    *( *( training_inputs + 11) + 1) = 0.0f;
    *( *( training_inputs + 11) + 2) = 1.0f;
    *( *( training_inputs + 11) + 3) = 1.0f;

    *( *( training_inputs + 12) + 0) = 1.0f;
    *( *( training_inputs + 12) + 1) = 1.0f;
    *( *( training_inputs + 12) + 2) = 0.0f;
    *( *( training_inputs + 12) + 3) = 0.0f;

    *( *( training_inputs + 13) + 0) = 1.0f;
    *( *( training_inputs + 13) + 1) = 1.0f;
    *( *( training_inputs + 13) + 2) = 0.0f;
    *( *( training_inputs + 13) + 3) = 1.0f;

    *( *( training_inputs + 14) + 0) = 1.0f;
    *( *( training_inputs + 14) + 1) = 1.0f;
    *( *( training_inputs + 14) + 2) = 1.0f;
    *( *( training_inputs + 14) + 3) = 0.0f;

    *( *( training_inputs + 15) + 0) = 1.0f;
    *( *( training_inputs + 15) + 1) = 1.0f;
    *( *( training_inputs + 15) + 2) = 1.0f;
    *( *( training_inputs + 15) + 3) = 1.0f;*/



    //initialize training output
    *( *( training_outputs + 0) + 0) = 0.0f;
    *( *( training_outputs + 1) + 0) = 1.0f;
    *( *( training_outputs + 2) + 0) = 1.0f;
    *( *( training_outputs + 3) + 0) = 0.0f;

    //training 4 bits
   /* *( *( training_outputs + 0) + 0) = 0.0f;
    *( *( training_outputs + 1) + 0) = 1.0f;
    *( *( training_outputs + 2) + 0) = 1.0f;
    *( *( training_outputs + 3) + 0) = 0.0f;

    *( *( training_outputs + 4) + 0) = 1.0f;
    *( *( training_outputs + 5) + 0) = 0.0f;
    *( *( training_outputs + 6) + 0) = 0.0f;
    *( *( training_outputs + 7) + 0) = 1.0f;

    *( *( training_outputs + 8) + 0) = 1.0f;
    *( *( training_outputs + 9) + 0) = 0.0f;
    *( *( training_outputs + 10) + 0) = 0.0f;
    *( *( training_outputs + 11) + 0) = 1.0f;

    *( *( training_outputs + 12) + 0) = 0.0f;
    *( *( training_outputs + 13) + 0) = 1.0f;
    *( *( training_outputs + 14) + 0) = 1.0f;
    *( *( training_outputs + 15) + 0) = 0.0f;*/


    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            *(*(hiddenWeights + i) + j) = init_weight();
        }
    }
    for (int i=0; i<numHiddenNodes; i++) {
        *(hiddenLayerBias+i) = init_weight();
        for (int j=0; j<numOutputs; j++) {
            *(*(outputWeights + i) + j) = init_weight();
        }
    }
    for (int i=0; i<numOutputs; i++) {
        *( outputLayerBias + i) = init_weight();
    }



    for (int cnt=0; cnt < numTrainingSets; ++cnt)
    {
        *(trainingSetOrder + cnt) = cnt;
    }

    for (int n=0; n < trainingTimes; n++) {

        shuffle(trainingSetOrder,numTrainingSets);

        for (int x=0; x<numTrainingSets; x++) {

            int i = *(trainingSetOrder + x);

            // Forward pass

            for (int j=0; j<numHiddenNodes; j++) {
                double activation=*(hiddenLayerBias+j);
                 for (int k=0; k<numInputs; k++) {
                    activation += (*(*(training_inputs + i) + k)) * (*(*(hiddenWeights + k) + j));
                }
                *(hiddenLayer+j) = sigmoid(activation);
            }

            for (int j=0; j<numOutputs; j++) {
                double activation=*(outputLayerBias+j);
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=(*(hiddenLayer+k)) * (*(*(outputWeights+k)+j));
                }
                *(outputLayer+j) = sigmoid(activation);
            }

    //printf("Input:%.0f %.0f %.0f %.0f       Output:%lf      Expected Output: %lf ",*(*(training_inputs+i)+0),*(*(training_inputs+i)+1),
      //   *(*(training_inputs+i)+2),*(*(training_inputs+i)+3),
        //*(outputLayer+0),*(*(training_outputs+i)+0));

            // Backprop

            //double deltaOutput[numOutputs];

            double *deltaOutput = level_1_array(numOutputs);

            for (int j=0; j<numOutputs; j++) {
                double errorOutput = *(*(training_outputs+i)+j) - *(outputLayer+j);
                *(deltaOutput+j) = errorOutput * dSigmoid(*(outputLayer+j));
            }

         //   printf("    Loss Function MSE: %.6f \n", fabs(*(deltaOutput+0)));
            //std::cout <<"    Loss Function MSE: "  << abs(deltaOutput[0]) <<'\n' ;

            //double deltaHidden[numHiddenNodes];
            double *deltaHidden = level_1_array(numHiddenNodes);

            for (int j=0; j<numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k<numOutputs; k++) {
                    errorHidden+=*(deltaOutput+k)* (*(*(outputWeights+j)+k));
                }
                *(deltaHidden+j) = errorHidden*dSigmoid(*(hiddenLayer+j));
            }

            for (int j=0; j<numOutputs; j++) {
                *(outputLayerBias+j) += *(deltaOutput+j)*lr;
                for (int k=0; k<numHiddenNodes; k++) {
                    *(*(outputWeights+k)+j)+=*(hiddenLayer+k)* (*(deltaOutput+j))*lr;
                }
            }

            for (int j=0; j<numHiddenNodes; j++) {
                *(hiddenLayerBias+j) += *(deltaHidden+j)*lr;
                for(int k=0; k<numInputs; k++) {
                    *(*(hiddenWeights+k)+j)+=*(*(training_inputs+i)+k) * (*(deltaHidden+j))*lr;
                }
            }
        }
    }

    // Print weights
    printf("Final Hidden Weights\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("[ ");
        for(int k=0; k<numInputs; k++) {
            printf("%lf ",*(*(hiddenWeights+k)+j));
        }
        printf("] ");
    }
    printf("]\n");

    printf("Final Hidden Biases\n[ ");
    for (int j=0; j<numHiddenNodes; j++) {
        printf("%lf ", *(hiddenLayerBias+j));

    }
    printf("]\n");
    printf("Final Output Weights");
    for (int j=0; j<numOutputs; j++) {
        printf("[ ");
        for (int k=0; k<numHiddenNodes; k++) {
            printf("%lf ",*(*(outputWeights+k)+j));
        }
        printf("]\n");
    }
    printf("Final Output Biases\n[ ");
    for (int j=0; j<numOutputs; j++) {
        printf("%lf ",*(outputLayerBias+j));
    }
    printf("]\n");


    result ( hiddenWeights, hiddenLayerBias, outputWeights, outputLayerBias);

    return 0;
}



//random change
void shuffle(int *array, size_t n) //input the order and number of training sets, then shuffle them
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = *(array+j); //t is a temp variable
            *(array+j) = *(array+i);
            *(array+i) = t; // switch the order tempedly
        }
    }
}


//get the xor nn training result
void result(  double** hiddenWeights,
                         double* hiddenLayerBias,
                         double** outputWeights,
                         double* outputLayerBias)
{
    int first_bit, second_bit, third_bit, forth_bit;

    while(1)
    {
        printf("Please enter the two bits in order, enter 99 at the first bit to exit  ");
        printf("\n Input the first bit: ");
        scanf("%d", &first_bit);
        if(first_bit==99)
            break;

        printf(" Input the second bit: ");
        scanf("%d", &second_bit);

        //printf(" Input the third bit: ");
        //scanf("%d", &third_bit);

        //printf(" Input the forth bit: ");
        //scanf("%d", &forth_bit);


        double neuron1 , neuron_temp1;
        double neuron2 , neuron_temp2;

        neuron_temp1 = first_bit * (*(*(hiddenWeights+0)+0)) +  second_bit * (*(*(hiddenWeights+1)+0)) + *(hiddenLayerBias+0) ;
        neuron_temp2 = first_bit * (*(*(hiddenWeights+0)+1)) +  second_bit * (*(*(hiddenWeights+1)+1)) + *(hiddenLayerBias+1) ;


        double outputLayer_temp;
        outputLayer_temp = (sigmoid(neuron_temp1) * (*(*(outputWeights+0)+0))) + (sigmoid(neuron_temp2) * (*(*(outputWeights+1)+0))) + *(outputLayerBias+0) ;


 /*       double neuron1 , neuron_temp1;
        double neuron2 , neuron_temp2;
        double neuron3 , neuron_temp3;
        double neuron4 , neuron_temp4;

        neuron_temp1 = first_bit * (*(*(hiddenWeights+0)+0)) +  second_bit * (*(*(hiddenWeights+1)+0))  +
        third_bit * (*(*(hiddenWeights+2)+0)) +  forth_bit * (*(*(hiddenWeights+3)+0)) + *(hiddenLayerBias+0);  //calculate hidden layer

        neuron_temp2 = first_bit * (*(*(hiddenWeights+0)+1)) +  second_bit * (*(*(hiddenWeights+1)+1))  +
        third_bit * (*(*(hiddenWeights+2)+1)) +  forth_bit * (*(*(hiddenWeights+3)+1)) + *(hiddenLayerBias+1);

        neuron_temp3 = first_bit * (*(*(hiddenWeights+0)+2)) +  second_bit * (*(*(hiddenWeights+1)+2))  +
        third_bit * (*(*(hiddenWeights+2)+2)) +  forth_bit * (*(*(hiddenWeights+3)+2)) + *(hiddenLayerBias+2);

        neuron_temp4 = first_bit * (*(*(hiddenWeights+0)+3)) +  second_bit * (*(*(hiddenWeights+1)+3))  +
        third_bit * (*(*(hiddenWeights+2)+3)) +  forth_bit * (*(*(hiddenWeights+3)+3)) + *(hiddenLayerBias+3);

        double outputLayer_temp;
        outputLayer_temp = sigmoid(neuron_temp1) * (*(*(outputWeights+0)+0)) + sigmoid(neuron_temp2) * (*(*(outputWeights+1)+0))+
        sigmoid(neuron_temp3) * (*(*(outputWeights+2)+0))+ sigmoid(neuron_temp4) * (*(*(outputWeights+3)+0)) + *(outputLayerBias+0) ;
*/

        printf(" output : %.0f \n\n\n", sigmoid(outputLayer_temp));
    }


}


