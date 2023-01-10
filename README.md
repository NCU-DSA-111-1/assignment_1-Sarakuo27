# DataStructure
 my learning program
-------------------------------------
# Neural Network in C

This is assignment 1

## Compile & Run

```sh
# Compile
cd src/
gcc -o main main.c neuron.c neuron.h layer.c layer.h backprop.h -lm
# Run
./main
```

## Display 1bit xor 1bit

Enter the number of Layers in Neural Network:
4
Enter number of neurons in layer[1]:
2
Enter number of neurons in layer[2]:
4
Enter number of neurons in layer[3]:
4
Enter number of neurons in layer[4]:
1

Neuron 1 in Layer 1 created
Neuron 2 in Layer 1 created

Neuron 1 in Layer 2 created
Neuron 2 in Layer 2 created
Neuron 3 in Layer 2 created
Neuron 4 in Layer 2 created

Neuron 1 in Layer 3 created
Neuron 2 in Layer 3 created
Neuron 3 in Layer 3 created
Neuron 4 in Layer 3 created

Neuron 1 in Layer 4 created



Neural Network Created Successfully...

Enter the learning rate (Usually 0.15):
0.15

Enter the number of training examples:
4

Enter the Inputs for training example[0]:
0 0

Enter the Inputs for training example[1]:
0 1

Enter the Inputs for training example[2]:
1 0

Enter the Inputs for training example[3]:
1 1

Enter the Desired Outputs (Labels) for training example[0]:
0

Enter the Desired Outputs (Labels) for training example[1]:
1

Enter the Desired Outputs (Labels) for training example[2]:
1

Enter the Desired Outputs (Labels) for training example[3]:
0

Training......

Enter input to test:
0 0
Output: 0

Enter input to test:
0 1
Output: 1

Enter input to test:
1 0
Output: 1

Enter input to test:
1 1
Output: 0

## Display 2bits xor 2bits
I'm still finding the appropriate num_layer and num_neuron to train the model.

Enter the number of Layers in Neural Network:
4
Enter number of neurons in layer[1]:
4
Enter number of neurons in layer[2]:
4
Enter number of neurons in layer[3]:
8
Enter number of neurons in layer[4]:
1

Neuron 1 in Layer 1 created
Neuron 2 in Layer 1 created
Neuron 3 in Layer 1 created
Neuron 4 in Layer 1 created

Neuron 1 in Layer 2 created
Neuron 2 in Layer 2 created
Neuron 3 in Layer 2 created
Neuron 4 in Layer 2 created

Neuron 1 in Layer 3 created
Neuron 2 in Layer 3 created
Neuron 3 in Layer 3 created
Neuron 4 in Layer 3 created
Neuron 5 in Layer 3 created
Neuron 6 in Layer 3 created
Neuron 7 in Layer 3 created
Neuron 8 in Layer 3 created

Neuron 1 in Layer 4 created



Neural Network Created Successfully...

Enter the learning rate (Usually 0.15):
0.09

Enter the number of training examples:
16

Enter the Inputs for training example[0]:
0 0 0 0

Enter the Inputs for training example[1]:
0 0 0 1

Enter the Inputs for training example[2]:
0 0 1 0

Enter the Inputs for training example[3]:
0 0 1 1

Enter the Inputs for training example[4]:
0 1 0 0

Enter the Inputs for training example[5]:
0 1 0 1

Enter the Inputs for training example[6]:
0 1 1 0

Enter the Inputs for training example[7]:
0 1 1 1

Enter the Inputs for training example[8]:
1 0 0 0

Enter the Inputs for training example[9]:
1 0 0 1

Enter the Inputs for training example[10]:
1 0 1 0

Enter the Inputs for training example[11]:
1 0 1 1

Enter the Inputs for training example[12]:
1 1 0 0

Enter the Inputs for training example[13]:
1 1 0 1

Enter the Inputs for training example[14]:
1 1 1 0

Enter the Inputs for training example[15]:
1 1 1 1

Enter the Desired Outputs (Labels) for training example[0]:
0

Enter the Desired Outputs (Labels) for training example[1]:
1

Enter the Desired Outputs (Labels) for training example[2]:
1

Enter the Desired Outputs (Labels) for training example[3]:
0

Enter the Desired Outputs (Labels) for training example[4]:
1

Enter the Desired Outputs (Labels) for training example[5]:
0

Enter the Desired Outputs (Labels) for training example[6]:
0

Enter the Desired Outputs (Labels) for training example[7]:
1

Enter the Desired Outputs (Labels) for training example[8]:
1

Enter the Desired Outputs (Labels) for training example[9]:
0

Enter the Desired Outputs (Labels) for training example[10]:
0

Enter the Desired Outputs (Labels) for training example[11]:
1

Enter the Desired Outputs (Labels) for training example[12]:
0

Enter the Desired Outputs (Labels) for training example[13]:
1

Enter the Desired Outputs (Labels) for training example[14]:
1

Enter the Desired Outputs (Labels) for training example[15]:
0

Training......

Enter input to test:
0 0 0 0
Output: 0

Enter input to test:
0 0 0 1
Output: 1

Enter input to test:
0 0 1 0
Output: 1

Enter input to test:
0 0 1 1
Output: 0

Enter input to test:
0 1 0 0
Output: 1

Enter input to test:
0 1 0 1
Output: 0

Enter input to test:
0 1 1 0
Output: 0

Enter input to test:
0 1 1 1
Output: 1

Enter input to test:
1 0 0 0
Output: 1

Enter input to test:
1 0 0 1
Output: 0

Enter input to test:
1 0 1 0
Output: 0

Enter input to test:
1 0 1 1
Output: 1

Enter input to test:
1 1 0 0
Output: 0

Enter input to test:
1 1 0 1
Output: 0             error!!!!!

Enter input to test:
1 1 1 0
Output: 1

Enter input to test:
1 1 1 1
Output: 0