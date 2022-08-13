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
                code_to_answers[pc, i, c, missing] = 'n'

with open('ground_truth_0.json') as f:
    data = json.load(f)

mk = {
    'ROME': 'rome',
    'GPT-2 XL': 'gpt',
    'FT_L': 'ft_l'
}
case_to_ratings = defaultdict(list)
case_to_data = defaultdict(dict)
label_to_ratings = defaultdict(list)
two_way_comparison = defaultdict(int)

def ranking(rating):
    return dict(m=1,n=2,l=3)[rating]

for record in data:
    pc = record['participant']
    for i in range(10):
        fname = record[f'page_{i+1}_fname']
        cfact = record[f'page_{i+1}_counterfactual']
        for a in 'abc':
            passage = record[f'page_{i+1}_passage_{a}']
            label = mk[record[f'page_{i+1}_passage_{a}_label']]
            case_to_data[fname]['fname'] = fname
            case_to_data[fname]['counterfactual'] = cfact
            case_to_data[fname][f'passage_{label}'] = passage
            votes = []
            for c in 'cf':
                rating = code_to_answers.get((pc, i, c, a), None)
                case_to_data[fname][f'rating_{label}_{c}'] = rating
                if rating is not None:
                    case_to_ratings[fname, label, c, rating].append(pc)
                    label_to_ratings[label, c, rating].append(pc)
                for m in 'ml':
                    voters = case_to_ratings[fname, label, c, m]
                    if len(voters):
                        votes.append(f'{c}{m}:' + ','.join(voters))
            case_to_data[fname][f'votes_{label}'] = '<br>\n'.join(votes)
        for c in 'cf':
            for ea, eb in [['rome', 'ft_l'], ['rome', 'gpt'], ['ft_l', 'gpt']]:
                if (ranking(case_to_data[fname][f'rating_{ea}_{c}'])
                     < ranking(case_to_data[fname][f'rating_{eb}_{c}'])):
                    two_way_comparison[f'{ea}_vs_{eb}_{c}'] += 1

summary = {}
for label in mk.values():
    for c in 'cf':
        for m in 'mnl':
            summary[f'votes_{label}_{c}{m}'] = len(label_to_ratings[label, c, m])

with open('summary_template.html') as f:
    summary_template = f.read()
with open('dump_template.html') as f:
    dump_template = f.read()

output = [
    summary_template.format(**summary, **two_way_comparison)
] + [
    dump_template.format(**case_to_data[fname])
    for fname in sorted(case_to_data.keys())
] + [
    f'<hr>\n<pre>{json.dumps(summary, indent=1)}\n\n{json.dumps(two_way_comparison, indent=1)}</pre>'
]

with open('www/responses.html', 'w') as f:
    f.write('\n'.join(output))
