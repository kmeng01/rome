from math import ceil
from numpy import zeros, ones, reshape
import matplotlib.pyplot as plt


def save_layer_edges(layer_num, layer_edges, save_path):
    # write out the list of strings
    filename = f'layer_{layer_num}_edges.txt'
    with open(save_path + '/' + filename, 'a+') as outf:
        outf.write('\n'.join(layer_edges))
        outf.write('\n')


def process_neuron_result(context_string, context_id, neuron_result, save_path='', layer_average=False, save_edges=False):
    token_total = neuron_result['scores'].shape[0]
    layer_total = neuron_result['scores'].shape[1]
    layer_width = 80
    layer_shape = [int(ceil(neuron_result['scores'][0]
                       [0].flatten().shape[0] / layer_width)), layer_width]
    total_matrix_shape = [layer_shape[0] *
                          layer_total, layer_shape[1] * token_total]
    total_matrix = zeros(total_matrix_shape)

    for t in range(token_total):
        for l in range(layer_total):
            layer_edges = []
            if layer_average:
                layer_mean = float(neuron_result['scores'][t][l].mean(
                )) + float(neuron_result['scores'][t][l].max())
                layer_matrix = ones(layer_shape) * layer_mean
            else:
                layer_matrix = zeros([layer_shape[0] * layer_shape[1]])
                flat_scores = neuron_result['scores'][t][l].flatten()
                true_size = flat_scores.shape[0]
                layer_matrix[:true_size] = flat_scores
                layer_matrix = reshape(layer_matrix, layer_shape)
                if save_edges:
                    for i, w in enumerate(flat_scores):
                        if w != 0:
                            # formatting to work with networkx lib, weighted edge list
                            layer_edges.append(
                                f'layer_{l};{neuron_result["kind"]};n_{i} token_{t};context_{context_id} {w} # {context_string}')
                    save_layer_edges(l, layer_edges, save_path)
            row = l * layer_shape[0]
            col = t * layer_shape[1]
            row_stop = row + layer_matrix.shape[0]
            col_stop = col + layer_matrix.shape[1]
            total_matrix[row:row_stop, col:col_stop] = layer_matrix
            layer_labels = list(range(0, layer_total + 1))
    return total_matrix, layer_shape, layer_labels


def make_matrix_plot(total_matrix, layer_shape, layer_labels, input_tokens):
    fig, ax = plt.subplots(figsize=(15, 20))
    mat = ax.matshow(total_matrix, cmap='Reds', vmin=total_matrix.min())
    ax.grid(True, alpha=0.15)
    # ax.invert_yaxis()
    ax.set_yticks([i for i in range(0, total_matrix.shape[0], layer_shape[0])])
    ax.set_xticks(
        [0.5 + i for i in range(0, total_matrix.shape[1], layer_shape[1])])
    ax.set_xticklabels(input_tokens)  # differences.shape[1] - 6, 5)))
    ax.set_yticklabels(layer_labels)
    ax.set_ylabel('model layer')
    ax.set_xlabel('context tokens')
    ax.tick_params(axis='y', labelsize='8')
    plt.show()
