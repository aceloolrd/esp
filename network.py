import numpy as np

class RecurrentNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 weights_input_hidden=None, 
                 weights_hidden_hidden=None,
                 weights_hidden_output=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if weights_input_hidden is None:
            self.weights_input_hidden = np.random.randn(hidden_size, input_size) * 0.1
        else:
            self.weights_input_hidden = weights_input_hidden
        
        if weights_hidden_hidden is None:
            self.weights_hidden_hidden = np.random.randn(hidden_size, hidden_size) * 0.1
        else:
            self.weights_hidden_hidden = weights_hidden_hidden
        
        if weights_hidden_output is None:
            self.weights_hidden_output = np.random.randn(output_size, hidden_size) * 0.1
        else:
            self.weights_hidden_output = weights_hidden_output

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = np.zeros(self.hidden_size)
        h_new = np.tanh(
            np.dot(self.weights_input_hidden, x) + 
            np.dot(self.weights_hidden_hidden, hidden_state)
        )
        output = np.tanh(np.dot(self.weights_hidden_output, h_new))
        return output, h_new

    def get_weights(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_input_hidden": self.weights_input_hidden,
            "weights_hidden_hidden": self.weights_hidden_hidden,
            "weights_hidden_output": self.weights_hidden_output
        }

    @staticmethod
    def from_weights(data):
        return RecurrentNetwork(
            data['input_size'],
            data['hidden_size'],
            data['output_size'],
            data['weights_input_hidden'],
            data['weights_hidden_hidden'],
            data['weights_hidden_output']
        )
