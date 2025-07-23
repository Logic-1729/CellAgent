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
    #pairs = balance_pairs(pairs)
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

def random_delete_chars(s, n):
    import random
    s_list = list(s)
    if len(s_list) <= n:
        return s  # 不删
    idxs = random.sample(range(len(s_list)), n)
    for idx in sorted(idxs, reverse=True):
        del s_list[idx]
    return ''.join(s_list)

def augment_self_pairs(pairs):
    # 统计每个sim区间的数量
    from collections import defaultdict
    import numpy as np
    bins = np.arange(0.5, 1.05, 0.05)
    sim2pairs = defaultdict(list)
    for q1, q2, sim in pairs:
        # 找到sim属于哪个区间
        for i in range(len(bins)-1):
            if bins[i] <= sim < bins[i+1]:
                sim2pairs[i].append((q1, q2, sim))
                break
    avg_count = int(np.mean([len(v) for v in sim2pairs.values()]))
    aug_pairs = []
    for i, pairlist in sim2pairs.items():
        if len(pairlist) < avg_count:
            needed = avg_count - len(pairlist)
            gen = 0
            # 只对q1!=q2的pair做增强
            valid_pairs = [(q1, q2, sim) for q1, q2, sim in pairlist if q1 != q2]
            if not valid_pairs:
                continue
            while gen < needed:
                for q1, q2, sim in valid_pairs:
                    # 随机选择对q1或q2做删除
                    if random.random() < 0.5:
                        target, other = q1, q2
                        is_first = True
                    else:
                        target, other = q2, q1
                        is_first = False
                    if target.startswith('Instruct:') and 'Query:' in target:
                        prefix, query = target.split('Query:', 1)
                        n_del = random.randint(1, min(5, len(query)))
                        query_aug = random_delete_chars(query, n_del)
                        target_aug = prefix + 'Query:' + query_aug
                        if is_first:
                            aug_pairs.append((target_aug, other, sim))
                        else:
                            aug_pairs.append((other, target_aug, sim))
                        gen += 1
                        if gen >= needed:
                            break
    return aug_pairs

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
    base_dir = os.path.abspath('.')
    for root, dirs, files in os.walk('.'):
        abs_root = os.path.abspath(root)
        # 跳过CellAgent根目录本身（兼容WSL和Windows分隔符）
        if abs_root.rstrip('/\\') == base_dir.rstrip('/\\'):
            continue
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
    
    # 输出每个相似度分数区间对应的pair个数

    def print_pair_distribution(pairs, name):
        import numpy as np
        bins = np.arange(0.5, 1.05, 0.05)
        bin_labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins)-1)]
        counts = [0]*(len(bins)-1)
        for _, _, sim in pairs:
            for i in range(len(bins)-1):
                if bins[i] <= sim < bins[i+1]:
                    counts[i] += 1
                    break
        print(f"\n{name} 区间分布:")
        for label, count in zip(bin_labels, counts):
            print(f"  {label}: {count}")

    aug_pairs = augment_self_pairs(pairs)
    aug_pairs_test = augment_self_pairs(pairs_test)

    print_pair_distribution(pairs + aug_pairs, "训练集")
    print_pair_distribution(pairs_test + aug_pairs_test, "测试集")

    df = pd.DataFrame(pairs + aug_pairs, columns=['sentence1', 'sentence2', 'score'])
    df.to_csv('sts_mybench_data.csv', index=False, encoding='utf-8')
    df = pd.DataFrame(pairs_test + aug_pairs_test, columns=['sentence1', 'sentence2', 'score'])
    df.to_csv('sts_mybench_data_test.csv', index=False, encoding='utf-8')

    with open('app_gap.json', 'w') as f:
        json.dump(app_gap, f, indent=4, ensure_ascii=False)
