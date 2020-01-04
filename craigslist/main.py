import numpy as np
import json
import re
from collections import defaultdict


def read_and_clean(filename):
    with open(filename, encoding='utf-8') as f:
        f.readline()  # skip first
        data_train = [parse_example(line) for line in f]
    return data_train


def parse_example(line):
    # excludes special symbols and numbers. keep all words in array
    dic = json.loads(line)
    dic['orig_heading'] = dic['heading']

    heading = dic['heading']
    heading = re.sub('[^A-Za-z]+', ' ', heading).lower()
    heading = heading.split()
    if not len(heading):
        heading = ['-1']
    dic['heading'] = heading
    return dic


def enrich_features(data, min_count=7):
    # exclude prepositions and very rare words
    exclude = {'off', 'the', 'from', 'for', 'amp', 'with'}
    words_count = defaultdict(int)
    for row in data:
        for word in row['heading']:
            if len(word) > 2 and word not in exclude:
                words_count[word] += 1
    for row in data:
        row['top_words'] = [word for word in row['heading'] if words_count[word] > min_count]


def get_ratio(data, col_name):
    # for col_name in data gets how often 'value' occurs
    ratios = {}
    for row in data:
        if col_name in row:
            if row[col_name] in ratios:
                ratios[row[col_name]] += 1
            else:
                ratios[row[col_name]] = 1
    for key in ratios.keys():
        ratios[key] /= len(data)
    return ratios


def get_conditionals(data):
    # per each category get probabilities of city, section and words

    data_grouped = defaultdict(list)
    for arr in data:
        data_grouped[arr['category']].append(arr)

    conditionals = {}
    for category, group in data_grouped.items():
        category_cond = {}
        category_cond['_city_'] = get_ratio(group, 'city')
        category_cond['_section_'] = get_ratio(group, 'section')
        category_cond['_words_'] = defaultdict(int)
        for row in group:
            top_words = set(row['top_words'])
            for word in top_words:
                category_cond['_words_'][word] += 1/len(group)

        conditionals[category] = category_cond
    return conditionals


def find_category(input, priors, conditionals):
    # per each category find numerator from Bayes formula as sum of log. pick one with maximum
    categories = list(priors.keys())

    scores = []
    for category in categories:
        score = np.log(priors[category])
        score += np.log(conditionals[category]['_city_'].get(input['city'], 0.000001))
        score += np.log(conditionals[category]['_section_'].get(input['section'], 0.000001))

        for word in input['top_words']:
            score += np.log(conditionals[category]['_words_'].get(word, 0.000001))

        scores.append(score)

    return categories[np.argmax(scores)]


data_train = read_and_clean('training.json')
enrich_features(data_train, min_count=7)
conditionals = get_conditionals(data_train)
priors = get_ratio(data_train, 'category')


test_in = read_and_clean('sample-test.in.json')
enrich_features(test_in, min_count=1)

with open('sample-test.out.json', encoding='utf-8') as f:
    test_out = np.array([line.rstrip() for line in f])

predictions = np.array([find_category(row, priors, conditionals) for row in test_in])
accuracy = sum(predictions == test_out) / len(test_out)
print(accuracy)  # 0.79



# if submit to HackerRank
# size_inp = int(input())
# test_in = [parse_example(input()) for line in range(size_inp)]
# enrich_features(test_in, min_count=1)
# for row in test_in:
#     print(find_category(row, priors, conditionals))
