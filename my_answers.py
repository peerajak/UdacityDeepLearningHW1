import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        #print('Found ',n_records,' number of samples')
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        #print('weight_h_o.shape',self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            #print('X.shape',X.shape)
            delta_weights_i_h__i = np.zeros(self.weights_input_to_hidden.shape)
            delta_weights_h_o__i = np.zeros(self.weights_hidden_to_output.shape)
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h__i, delta_weights_h_o__i = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h__i, delta_weights_h_o__i)
            #print('delta_weights_h_o__i',delta_weights_h_o__i)
            
            delta_weights_i_h += delta_weights_i_h__i/n_records
            delta_weights_h_o += delta_weights_h_o__i/n_records
            #print('delta_weights_h_o',delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        X = np.expand_dims(X,axis=0)
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer
        #print('hidden_inputs',hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)# signals from hidden layer
        #print('hidden_outputs',hidden_outputs)
        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = hidden_outputs # signals into final output layer
        final_outputs = np.dot(final_inputs,self.weights_hidden_to_output) # signals from final output layer
        #print (final_outputs)
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        X = np.expand_dims(X,axis=0)
        error = 0.5*(y-final_outputs)**2 # Output layer error is the difference between desired target and actual output.
        #print('error',error)
        # TODO: Calculate the hidden layer's contribution to the error
        d_error_by_d_output =  -(y-final_outputs).squeeze()
        #print('d_error_by_d_output',d_error_by_d_output)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = d_error_by_d_output*hidden_outputs
        #print('output_error_term',output_error_term)
        dh_o =(d_error_by_d_output*self.weights_hidden_to_output).transpose()
        #print('dh_o',dh_o)
        hidden_error_term = np.multiply(dh_o,np.multiply(hidden_outputs,(1-hidden_outputs)))
        #print('hidden_error_term',hidden_error_term)
        
        # Weight step (input to hidden)
        delta_out_weights_i_h = np.dot(X.transpose(),hidden_error_term)
        # Weight step (hidden to output)
        delta_out_weights_h_o = output_error_term.transpose()
        return delta_out_weights_i_h, delta_out_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += -self.lr*delta_weights_h_o# update hidden-to-output weights with gradient descent step
        #print('answer weights_hidden_to_output',self.weights_hidden_to_output)
        self.weights_input_to_hidden += -self.lr*delta_weights_i_h # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''

        X = features#np.expand_dims(features,axis=0)
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) 
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = hidden_outputs 
        final_outputs = np.dot(final_inputs,self.weights_hidden_to_output)
        #print('run.final_outputs',final_outputs)
        return final_outputs
#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.5
hidden_nodes = 30
output_nodes = 1
