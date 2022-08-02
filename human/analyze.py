from csv import DictReader
import json
from collections import defaultdict

code_to_answers = {}

with open('anon_responses.csv') as f:
    for record in DictReader(f):
        record = {k.strip(): v for k, v in record.items()}
        pc = record['PARTICIPANT CODE']
        for i in range(10):
            for c, consistency, consistent in [
                    ['c', 'CONSISTENCY', 'CONSISTENT'],
                    ['f', 'FLUENCY', 'FLUENT']]:
                most_least = []
                for m, most in [['m', 'MOST'], ['l', 'LEAST']]:
                    key = f'PAGE {i+1}, {consistency} [{most} {consistent}?]'
                    choice = record[key].lower()
                    most_least.append(choice)
                    code_to_answers[pc, i, c, choice] = m
                missing = (next(iter(set('abc') - set(most_least))))

                # most_least[1:1] = [missing]

with open('ground_truth_0.json') as f:
    data = json.load(f)

mk = {
        'ROME': 'rome',
        'GPT-2 XL': 'gpt',
        'FT_L ': 'ft_l'
}
case_to_ratings = defaultdict(list)
case_to_data = defaultdict(dict)

with open('dump_template.html') as f:
    dump_template = f.read()

for record in data:
    pc = record['participant']
    for i in range(10):
        fname = record[f'page_{i+1}_fname']
        cfact = record[f'page_{i+1}_counterfactual']
        for a in 'abc':
            passage = record[f'page_{i+1}_passage_{a}']
            label = mk[record[f'page_{i+1}_passage_{a}_label']]
            rating = code_to_answers.get((pc, i, 'c', a), None)
            if rating is not None:
                case_to_ratings[fname, 'c', label, rating].append(pc)
            case_to_data[fname]['fname'] = fname
            case_to_data[fname]['counterfactual'] = cfact
            case_to_data[fname][f'passage_{label}'] = passage

            print(label)


            case_to_data[fname][f'votes_{label}'] = votes




