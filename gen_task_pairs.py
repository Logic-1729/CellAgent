import json, os, itertools
import random
from sklearn.model_selection import train_test_split
import pandas as pd

UNRELATED_SIM = 0.6
MIN_SIM = 0.7
MAX_SIM = 1.0

def get_task_templates(raw_template):
    results = []
    left_pos = 0
    while True:
        left = raw_template.find('(', left_pos)
        if left == -1:
            break    
        right = raw_template.find(')', left + 1)
        if right == -1:
            break
            
        content = raw_template[left + 1: right]
        contents = content.replace("NULL", "").split('|')
        
        results.append({
            'left': left,
            'right': right,
            'contents': contents
        })
        
        left_pos = right + 1
    combinations = itertools.product(*[result['contents'] for result in results])
    task_templates = []
    for combination in combinations:
        segments = []
        last_left = 0
        for i, content in enumerate(combination):
            left = results[i]['left']
            right = results[i]['right']
            segments.append(raw_template[last_left:left])
            segments.append(content)
            last_left = right + 1
        segments.append(raw_template[last_left:])
        task_templates.append(''.join(segments))
    return task_templates
    
def get_trajectory(trajectory_template, fmt):
    trajectory = []
    for act in trajectory_template:
        for k in fmt.keys():
            if f"{{{k}}}" in act:
                act = act.replace(f"{{{k}}}", fmt[k])
            if f"<{k}>" in act:
                act = act.replace(f"<{k}>", f"<{fmt[k]}>")
        trajectory.append(act)
    return trajectory

def balance_pairs(pairs):
    ret = []
    sim_samples = {}
    for task1, task2, sim in pairs:
        if sim not in sim_samples:
            sim_samples[sim] = 0
        sim_samples[sim] += 1
    avg_samples = sum(sim_samples.values()) / len(sim_samples)
    for task1, task2, sim in pairs:
        if sim_samples[sim] < avg_samples:
            repeat_num = int(avg_samples / sim_samples[sim])
            ret.extend([(task1, task2, sim)] * repeat_num)
        else:
            ret.append((task1, task2, sim))
    return ret

def add_perturbation(pairs):
    max_perturbation = 0.001
    return [(task1, task2, round(sim + random.uniform(-max_perturbation, max_perturbation), 4)) for task1, task2, sim in pairs]

app_pairs = {}

def single_app(task_trajectory_pairs):
    num_tasks = len(task_trajectory_pairs)
    possible_lshare = {}
    for i, j in itertools.combinations(range(num_tasks), 2):
        task1, trajectory1 = task_trajectory_pairs[i]
        task2, trajectory2 = task_trajectory_pairs[j]
        len1 = len(trajectory1)
        len2 = len(trajectory2)
        k = 0
        l_share = 0
        while k < len1 and k < len2:
            if trajectory1[k] == trajectory2[k]:
                l_share += 1
            else:
                break
            k += 1
        if l_share not in possible_lshare:
            possible_lshare[l_share] = []
        possible_lshare[l_share].append((task1, task2))
    num_tiers = len(possible_lshare.keys())
    lshares = sorted(possible_lshare.keys())
    gap = (MAX_SIM - MIN_SIM) / num_tiers
    pairs = []
    for i in range(num_tiers):
        l_share = lshares[i]
        sim = MIN_SIM + i * gap
        for t1, t2 in possible_lshare[l_share]:
            pairs.append((t1, t2, sim))
    pairs = balance_pairs(pairs)
    return round(gap, 6), pairs

def get_app_task_trajectories(domain_dir):
    with open(os.path.join(domain_dir, "templates.json")) as f:
        templates = json.load(f)
    print(f"Domain: {domain_dir}")
    task_trajectories = {}
    for template in templates:
        raw_task_template = template["task"]
        trajectory_template = template["trajectory"]
        task_templates = get_task_templates(raw_task_template)
        candidates = template["candidates"]
        dependency = template.get("dependency", "no")
        keys = list(candidates.keys())
        
        if dependency == "one-to-one":
            combinations = zip(*[candidates[k] for k in keys])
        elif dependency == "no":
            combinations = itertools.product(*[candidates[k] for k in keys])
        else:
            print(f"Unknonw dependency type: {dependency}")
            continue
            
        for combination in combinations:
            fmt = {}
            for i, k in enumerate(keys):
                fmt[k] = combination[i]
            trajectory = get_trajectory(trajectory_template, fmt)
            for task_template in task_templates:
                task = task_template.format(**fmt)
                task_trajectories[task] = trajectory
    # print(task_trajectories)
    app_task_trajectories = {}
    for task, trajectory in task_trajectories.items():
        app = trajectory[0].split(' ')[1]
        app = app.replace("<", "").replace(">", "")
        if app not in app_task_trajectories:
            app_task_trajectories[app] = []
        app_task_trajectories[app].append((task, trajectory))

    return app_task_trajectories

def single_domain(domain_dir):
    app_task_trajectories = get_app_task_trajectories(domain_dir)

    app_pairs = {}
    app_gap = {}
    for app in app_task_trajectories:
        gap, pairs = single_app(app_task_trajectories[app])
        app_gap[app] = gap
        app_pairs[app] = pairs
    
    return app_gap, app_pairs
        
    
if __name__ == '__main__':
    app_pairs = {}
    app_gap = {}
    for root, _, files in os.walk('.'):
        if 'templates.json' in files:
            domain_app_gap, domain_app_pairs = single_domain(root)
            app_pairs.update(domain_app_pairs)
            app_gap.update(domain_app_gap)

    apps = list(app_pairs.keys())
    unrelated_pairs = []
    
    for i, j in itertools.combinations(range(len(apps)), 2):
        pairs_i = app_pairs[apps[i]]
        pairs_j = app_pairs[apps[j]]
        tasks_i = list(set(task for task, _, _ in pairs_i))
        tasks_j = list(set(task for task, _, _ in pairs_j))
        for t1, t2 in itertools.product(tasks_i, tasks_j):
            unrelated_pairs.append((t1, t2, UNRELATED_SIM))

    pairs = [item for sublist in app_pairs.values() for item in sublist]
    pairs, pairs_test = train_test_split(pairs, test_size=0.1, random_state=42)
    
    if len(pairs) < len(unrelated_pairs) // 2:
        print(f"len(pairs) is {len(pairs)}, and len(unrelated_pairs) is {len(unrelated_pairs)}, balancing...")
        import math
        multiplier = math.ceil(len(unrelated_pairs) / len(pairs))
        pairs = pairs * multiplier

    # unrelated_pairs = random.sample(unrelated_pairs, int(0.5 * len(pairs)))
    unrelated_pairs, unrelated_pairs_test = train_test_split(unrelated_pairs, test_size=0.1, random_state=42)
    
    pairs.extend(unrelated_pairs)
    pairs_test.extend(unrelated_pairs_test)

    pairs = add_perturbation(pairs)
    pairs_test = add_perturbation(pairs_test)

    # fmt = "query: {}"
    fmt = "Instruct: Represent this phone-use task description for searching similar tasks\nQuery:{}" 
    pairs = [(fmt.format(t1), fmt.format(t2), sim) for t1, t2, sim in pairs]
    pairs_test = [(fmt.format(t1), fmt.format(t2), sim) for t1, t2, sim in pairs_test]
    
    df = pd.DataFrame(pairs, columns=['sentence1', 'sentence2', 'score'])
    df.to_csv('sts_mybench_data.csv', index=False, encoding='utf-8')
    df = pd.DataFrame(pairs_test, columns=['sentence1', 'sentence2', 'score'])
    df.to_csv('sts_mybench_data_test.csv', index=False, encoding='utf-8')

    with open('app_gap.json', 'w') as f:
        json.dump(app_gap, f, indent=4, ensure_ascii=False)
