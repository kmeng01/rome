import json

with open('sample_collection.json') as f:
    data = json.load(f)

with open('template.html') as f:
    template = f.read()

# 15 people, each looking at 20 cases out of 100.
# Ideally: out of 100 cases, there is no
# case where the same triple is looking at it

import random
oversample = 3
start = 50 # First counterfactual case
cc = 50   # number of counterfactual cases
tc = cc * oversample
raters = 15
pages = tc // raters
choices = ['GPT-2 XL', 'FT_L', 'ROME']
random.seed(start)

participants = ['%x' % random.randrange(256, 256*16) for _ in range(raters)]
assignments = [(person, q) for person in range(raters) for q in range(pages)]
labels = [tuple(random.sample(choices, k=len(choices))) for q in range(tc)]
random.shuffle(assignments)


def random_swap(a, i):
    j = random.randrange(len(a))
    t = a[i]
    a[i] = a[j]
    a[j] = t

assert(len(assignments) == tc)

# Make a latin square that avoids duplicate evaluations.
needpass = True
while needpass:
    needpass = False
    group_to_case = {}
    for i in range(tc):
        # Mix things if a person is assigned to the same thing twice.
        if assignments[i][0] in [
                assignments[(i + s) % tc][0] for s in range(cc, tc, cc)]:
            random_swap(assignments, i)
            needpass = True
        # Mix things if the same group is looking at the same case twice
        group = tuple(sorted(assignments[(i + s) % tc][0] for s in range(0, tc, cc)))
        if group_to_case.get(group, i % cc) != i % cc:
            random_swap(assignments, i)
            needpass = True
        else:
            group_to_case[group] = i % cc
        # Mix things if the same ordering is used for the same question twice
        if labels[i] in [
                labels[(i + s) % tc] for s in range(cc, tc, cc)]:
            random_swap(labels, i)
            needpass = True

# Now collate all the rater's assignments.
work = {
  assigned: dict(test=test, truelabels=truelabels, d=d)
  for test, assigned, truelabels, d in zip(
      range(start*oversample, start*oversample+tc),
      assignments,
      labels,
      (data[start:start+cc] * oversample))
}
ground_truth = []

# Now produce the template files
for r in range(raters):
    participant = participants[r]
    job = { 'participant': participant }
    for p in range(pages):
        d = work[r, p]['d']
        tl = work[r, p]['truelabels']
        counterfactual = (d['request']['prompt'].format(d['request']['subject']) + ' ' +
           d['request']['target_new']['str'])
        texts = [d[t][0] for t in tl]
        job[f'page_{p+1}_fname'] = d['fname']
        job[f'page_{p+1}_counterfactual'] = counterfactual
        for letter, text, t in zip('abc', texts, tl):
            job[f'page_{p+1}_passage_{letter}'] = text
            job[f'page_{p+1}_passage_{letter}_label'] = t
    filename = f'www/participant_{participant}.html'
    with open(filename, 'w') as f:
        f.write(template.format(**job))
    ground_truth.append(job)
    print(f'participant_{participant}.html')

# Write out ground-truth
with open(f'ground_truth_{start}.json', 'w') as f:
    json.dump(ground_truth, f, indent=1)

