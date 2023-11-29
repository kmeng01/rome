
import json
import torch

def preprocess_max_layer(input_file_name='return_mlp.txt', output_file_name='max_layer_requests.json'):
    processed_data = [] 
    with open(input_file_name, 'r') as input_file:
        for line in input_file:
            # Directly parse each line as a JSON object
            data = json.loads(line.strip())

            scores = torch.Tensor(data["scores"])
            end_idx = data["subject_range"][1]
            layers = [torch.argmax(scores[end_idx]).item()]
            data["max_score_layer"] = layers

            # data.pop('correct_prediction', None) 
            data.pop('scores', None)

            processed_data.append(data)

    with open(output_file_name, 'w') as output_file:
        json.dump(processed_data, output_file, indent=4)

if __name__ == "__main__":
    preprocess_max_layer()