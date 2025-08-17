from liver_model import *

#LIVER MODEL EQ

weights = []
biases = []

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        w, b = layer.get_weights()
        weights.append(w)
        biases.append(b)

# Generate Neural Network Equation
def generate_equation(weights, biases):
    equation_str = "Neural Network Equation:\n"

    prev_layer_size = X_train.shape[1]  # Number of input features
    layer_num = 1

    for w, b in zip(weights, biases):
        equation_str += f"\nLayer {layer_num}:\n"
        
        if layer_num < len(weights):
            equation_str += "Output = ReLU(\n"
        else:
            equation_str += "Final Output = \n"

        for neuron_idx in range(w.shape[1]):  # Iterate over neurons in the layer
            terms = [f"{w[input_idx, neuron_idx]:.6f} * x{input_idx}" for input_idx in range(prev_layer_size)]
            equation = " + ".join(terms) + f" + {b[neuron_idx]:.6f}"
            
            
            equation_str += f"    [{equation}]\n" if neuron_idx == 0 else f"  + [{equation}]\n"

        equation_str += ")" if layer_num < len(weights) else ""  # Close ReLU if it's not the last layer
        equation_str += "\n"
        
        prev_layer_size = w.shape[1]
        layer_num += 1

    return equation_str

# Print the final equation with actual values
equation_str = generate_equation(weights, biases)
print(equation_str)

