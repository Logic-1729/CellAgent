import json, os, itertools
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import time


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
    s_list = list(s)
    if len(s_list) <= n:
        return s  # 不删
    idxs = random.sample(range(len(s_list)), n)
    for idx in sorted(idxs, reverse=True):
        del s_list[idx]
    return ''.join(s_list)

def extract_non_task_parts(task):
    """
    根据templates.json的模板结构提取任务中的非task部分（参数部分）
    返回非task部分的位置范围列表 [(start, end), ...]
    """
    non_task_parts = []
    
    # 音乐相关模式
    if "网易云音乐播放" in task:
        if "的热门歌曲" in task:
            # 用网易云音乐播放{artist}的热门歌曲 -> artist是非task部分
            start = task.find("播放") + 2
            end = task.find("的热门歌曲")
            if start < end:
                non_task_parts.append((start, end))
        elif task.count("播放") == 1 and "的热门歌曲" not in task:
            # 用网易云音乐播放{song} -> song是非task部分
            start = task.find("播放") + 2
            if start < len(task):
                non_task_parts.append((start, len(task)))
    
    elif "打开网易云音乐" in task and "列表" in task:
        # 打开网易云音乐{list}列表 -> list是非task部分
        start = task.find("网易云音乐") + 5
        end = task.find("列表")
        if start < end:
            non_task_parts.append((start, end))
    
    # 视频相关模式
    elif "在哔哩哔哩上搜索" in task:
        if "播放" in task and "相关视频" in task:
            # 在哔哩哔哩上搜索播放{topic}相关视频 -> topic是非task部分
            start = task.find("播放") + 2
            end = task.find("相关视频")
            if start < end:
                non_task_parts.append((start, end))
        elif "相关视频" in task:
            # 在哔哩哔哩上搜索{topic}相关视频 -> topic是非task部分
            start = task.find("搜索") + 2
            end = task.find("相关视频")
            if start < end:
                non_task_parts.append((start, end))
        elif task.endswith("搜索") == False and "相关视频" not in task:
            # 在哔哩哔哩上搜索{topic} -> topic是非task部分
            start = task.find("搜索") + 2
            if start < len(task):
                non_task_parts.append((start, len(task)))
    
    elif ("关注哔哩哔哩" in task) and ("用户" in task or "UP主" in task):
        # 关注哔哩哔哩用户{user} 或 关注哔哩哔哩UP主{user} -> user是非task部分
        if "用户" in task:
            start = task.find("用户") + 2
        else:  # UP主
            start = task.find("UP主") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif ("播放哔哩哔哩" in task) and ("用户" in task or "UP主" in task) and "的最新视频" in task:
        # 播放哔哩哔哩用户{user}的最新视频 或 播放哔哩哔哩UP主{user}的最新视频 -> user是非task部分
        if "用户" in task:
            start = task.find("用户") + 2
        else:  # UP主
            start = task.find("UP主") + 3
        end = task.find("的最新视频")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "进入哔哩哔哩" in task and task.endswith("区"):
        # 进入哔哩哔哩{zone}区 -> zone是非task部分
        start = task.find("哔哩哔哩") + 4
        end = task.find("区")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "打开哔哩哔哩" in task and "列表" in task:
        # 打开哔哩哔哩{list}列表 -> list是非task部分
        start = task.find("哔哩哔哩") + 4
        end = task.find("列表")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "在抖音上搜索" in task:
        if "相关视频" in task:
            # 在抖音上搜索{topic}相关视频 -> topic是非task部分
            start = task.find("搜索") + 2
            end = task.find("相关视频")
            if start < end:
                non_task_parts.append((start, end))
        else:
            # 在抖音上搜索{topic} -> topic是非task部分
            start = task.find("搜索") + 2
            if start < len(task):
                non_task_parts.append((start, len(task)))
    
    # 浏览器相关模式
    elif "浏览器搜索" in task:
        # 用{browser}浏览器搜索{query} -> browser和query都是非task部分
        if "用" in task and task.find("用") < task.find("浏览器"):
            # 提取browser部分
            browser_start = task.find("用") + 1
            browser_end = task.find("浏览器")
            if browser_start < browser_end:
                non_task_parts.append((browser_start, browser_end))
        
        # 提取query部分
        query_start = task.find("搜索") + 2
        if query_start < len(task):
            non_task_parts.append((query_start, len(task)))
    
    # 邮件相关模式
    elif "发送一篇邮件给" in task:
        # 发送一篇邮件给{email}，主题为{subject}，内容为{content} -> email, subject, content是非task部分
        # 提取email部分
        email_start = task.find("邮件给") + 3
        email_end = task.find("，主题为")
        if email_start < email_end:
            non_task_parts.append((email_start, email_end))
        
        # 提取subject部分
        subject_start = task.find("主题为") + 3
        subject_end = task.find("，内容为")
        if subject_start < subject_end:
            non_task_parts.append((subject_start, subject_end))
        
        # 提取content部分
        content_start = task.find("内容为") + 3
        if content_start < len(task):
            non_task_parts.append((content_start, len(task)))
    
    elif "打开邮箱的" in task:
        # 打开邮箱的{list} -> list是非task部分
        start = task.find("邮箱的") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "查看邮件" in task:
        # 查看邮件{subject}的内容 或 查看邮件{subject} -> subject是非task部分
        start = task.find("查看邮件") + 4
        end = len(task)
        if "的内容" in task:
            end = task.find("的内容")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "搜索主题为" in task and "的邮件" in task:
        # 搜索主题为{subject}的邮件 -> subject是非task部分
        start = task.find("搜索主题为") + 5
        end = task.find("的邮件")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "删除邮件" in task:
        # 删除邮件{subject} -> subject是非task部分
        start = task.find("删除邮件") + 4
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "回复邮件" in task and "，内容为" in task:
        # 回复邮件{subject}，内容为{content} -> subject和content是非task部分
        # 提取subject部分
        subject_start = task.find("回复邮件") + 4
        subject_end = task.find("，内容为")
        if subject_start < subject_end:
            non_task_parts.append((subject_start, subject_end))
        
        # 提取content部分
        content_start = task.find("内容为") + 3
        if content_start < len(task):
            non_task_parts.append((content_start, len(task)))
    
    elif "将邮件" in task and "标为星标邮件" in task:
        # 将邮件{subject}标为星标邮件 -> subject是非task部分
        start = task.find("将邮件") + 3
        end = task.find("标为星标邮件")
        if start < end:
            non_task_parts.append((start, end))
    
    # 知乎相关模式
    elif "在知乎上提问" in task:
        # 在知乎上提问{question} -> question是非task部分
        start = task.find("提问") + 2
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "在知乎上搜索" in task:
        # 在知乎上搜索{query}相关内容 或 在知乎上搜索{query} -> query是非task部分
        start = task.find("搜索") + 2
        end = len(task)
        if "相关内容" in task:
            end = task.find("相关内容")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "在知乎上回答" in task:
        # 在知乎上回答{query}相关问题 或 在知乎上回答{query} -> query是非task部分
        start = task.find("回答") + 2
        end = len(task)
        if "相关问题" in task:
            end = task.find("相关问题")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "在知乎上查看" in task and "的个人主页" in task:
        # 在知乎上查看{user}的个人主页 -> user是非task部分
        start = task.find("查看") + 2
        end = task.find("的个人主页")
        if start < end:
            non_task_parts.append((start, end))
    
    # 微信相关模式
    elif "发一条微信朋友圈" in task:
        # 发一条微信朋友圈，内容为{content} -> content是非task部分
        start = task.find("内容为") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "给" in task and "发一条微信" in task:
        # 给{user}发一条微信，内容为{content} -> user和content是非task部分
        # 提取user部分
        user_start = task.find("给") + 1
        user_end = task.find("发一条微信")
        if user_start < user_end:
            non_task_parts.append((user_start, user_end))
        
        # 提取content部分
        content_start = task.find("内容为") + 3
        if content_start < len(task):
            non_task_parts.append((content_start, len(task)))
    
    elif "搜索并打开微信小程序" in task:
        # 搜索并打开微信小程序{app} -> app是非task部分
        start = task.find("小程序") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "搜索并关注微信公众号" in task:
        # 搜索并关注微信公众号{account} -> account是非task部分
        start = task.find("公众号") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "打开微信" in task and "服务" in task:
        # 打开微信{service}服务 -> service是非task部分
        start = task.find("微信") + 2
        end = task.find("服务")
        if start < end:
            non_task_parts.append((start, end))
    
    # 微博相关模式
    elif "发一条微博" in task:
        # 发一条微博，内容为{content} -> content是非task部分
        start = task.find("内容为") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "关注微博博主" in task:
        # 关注微博博主{user} -> user是非task部分
        start = task.find("博主") + 2
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "查看微博收到的" in task:
        # 查看微博收到的{notification} -> notification是非task部分
        start = task.find("收到的") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "搜索" in task and "相关微博" in task:
        # 搜索{topic}相关微博 -> topic是非task部分
        start = task.find("搜索") + 2
        end = task.find("相关微博")
        if start < end:
            non_task_parts.append((start, end))
    
    # 外卖相关模式
    elif "打开" in task and "外卖菜单" in task:
        # 打开{app}{restaurant}外卖菜单 -> app和restaurant是非task部分
        app_start = task.find("打开") + 2
        app_end = task.find("外卖菜单")
        # 需要分割app和restaurant部分，这里简化处理
        if app_start < app_end:
            non_task_parts.append((app_start, app_end))
    
    elif "用" in task and "点一份" in task and "外卖" in task:
        # 用{app}点一份{food}外卖 或 用{app}点一份{restaurant}的{food}外卖
        # 提取app部分
        app_start = task.find("用") + 1
        app_end = task.find("点一份")
        if app_start < app_end:
            non_task_parts.append((app_start, app_end))
        
        # 提取food或restaurant+food部分
        food_start = task.find("点一份") + 3
        food_end = task.find("外卖")
        if food_start < food_end:
            non_task_parts.append((food_start, food_end))
    
    # 购物相关模式
    elif "在淘宝上搜索" in task and "购买" not in task and "进入" not in task:
        # 在淘宝上搜索{item} -> item是非task部分
        start = task.find("搜索") + 2
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "查看淘宝" in task and "订单" in task:
        # 查看淘宝{category}订单 -> category是非task部分
        start = task.find("淘宝") + 2
        end = task.find("订单")
        if start < end:
            non_task_parts.append((start, end))
    
    elif "在淘宝上搜索进入" in task:
        # 在淘宝上搜索进入{shop} -> shop是非task部分
        start = task.find("进入") + 2
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    elif "在淘宝上搜索购买" in task:
        # 在淘宝上搜索购买{item} -> item是非task部分
        start = task.find("购买") + 2
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    # 出行相关模式
    elif "用" in task and "导航去" in task:
        # 用{app}导航去{place} -> app和place是非task部分
        # 提取app部分
        app_start = task.find("用") + 1
        app_end = task.find("导航去")
        if app_start < app_end:
            non_task_parts.append((app_start, app_end))
        
        # 提取place部分
        place_start = task.find("导航去") + 3
        if place_start < len(task):
            non_task_parts.append((place_start, len(task)))
    
    elif "用" in task and "搜索附近的" in task:
        # 用{app}搜索附近的{place} -> app和place是非task部分
        # 提取app部分
        app_start = task.find("用") + 1
        app_end = task.find("搜索附近的")
        if app_start < app_end:
            non_task_parts.append((app_start, app_end))
        
        # 提取place部分
        place_start = task.find("搜索附近的") + 5
        if place_start < len(task):
            non_task_parts.append((place_start, len(task)))
    
    elif "用滴滴打车到" in task:
        # 用滴滴打车到{place} -> place是非task部分
        start = task.find("打车到") + 3
        if start < len(task):
            non_task_parts.append((start, len(task)))
    
    return non_task_parts


def augment_self_pairs(pairs, query2task):
    # 统计每个sim区间的数量
    bins = np.arange(0.5, 1.05, 0.05)
    sim2pairs = defaultdict(list)
    for q1, q2, sim in pairs:
        # 找到sim属于哪个区间
        for i in range(len(bins)-1):
            if bins[i] <= sim < bins[i+1]:
                sim2pairs[i].append((q1, q2, sim))
                break
    avg_count = int(np.mean([len(v) for v in sim2pairs.values()]))
    print(f"[实时] 每区间目标数量: {avg_count}")
    aug_pairs = []
    
    for i, pairlist in sim2pairs.items():
        bin_start = bins[i]
        if bin_start < 0.80:  # 只对高分区间(>=0.80)进行增强
            continue
        
        needed = avg_count - len(pairlist)
        if needed <= 0:
            continue
            
        valid_pairs = [(q1, q2, sim) for q1, q2, sim in pairlist if q1 != q2]
        print(f"[实时] 高分区间{i} [{bin_start:.2f},{bins[i+1]:.2f}) 目标: {avg_count}, 当前: {len(pairlist)}, 可用对: {len(valid_pairs)}")
        
        if not valid_pairs:
            continue
            
        gen = 0
        max_iterations = len(valid_pairs) * 50  # 设置最大迭代次数防止死循环
        iteration_count = 0
        
        while gen < needed and iteration_count < max_iterations:
            for q1, q2, sim in valid_pairs:
                if gen >= needed:
                    break
                    
                # 随机选择对q1或q2做删除
                if random.random() < 0.5:
                    target, other = q1, q2
                    is_first = True
                else:
                    target, other = q2, q1
                    is_first = False
                    
                if target.startswith('Instruct:') and 'Query:' in target:
                    prefix, query = target.split('Query:', 1)
                    task = query2task.get(target, None)
                    if task is None:
                        print(f"[警告] 找不到映射: {target[:50]}...")
                        continue
                    
                    # 使用新的函数提取非task部分
                    non_task_parts = extract_non_task_parts(task)
                    
                    if not non_task_parts:
                        print(f"[警告] 无法找到非task部分: {task[:50]}...")
                        continue
                    
                    # 随机选择一个非task部分进行字符删除
                    start, end = random.choice(non_task_parts)
                    non_task_str = task[start:end]
                    
                    if len(non_task_str) == 0:
                        print(f"[警告] 选中的非task部分为空: {task[:50]}...")
                        continue
                    
                    # 对选中的非task部分进行随机字符删除
                    n_del = random.randint(1, min(3, len(non_task_str)))
                    non_task_aug = random_delete_chars(non_task_str, n_del)
                    
                    # 重新构建任务
                    task_aug = task[:start] + non_task_aug + task[end:]
                    query_aug = task_aug
                    target_aug = prefix + 'Query:' + query_aug
                    
                    if is_first:
                        aug_pairs.append((target_aug, other, sim))
                    else:
                        aug_pairs.append((other, target_aug, sim))
                    gen += 1
                    
                iteration_count += 1
                
        if iteration_count >= max_iterations:
            print(f"[警告] 高分区间{i} 达到最大迭代次数，可能存在问题")
        
        print(f"[实时] 高分区间{i} 实际生成增强: {gen} (目标: {needed})")
    
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
    
    # 遍历所有子目录查找templates.json文件
    for root, dirs, files in os.walk('.'):
        abs_root = os.path.abspath(root)
        # 跳过CellAgent根目录本身（兼容WSL和Windows分隔符）
        if abs_root.rstrip('/\\') == base_dir.rstrip('/\\'):
            continue
        template_path = os.path.join(abs_root, 'templates.json')
        if os.path.isfile(template_path):
            domain_app_gap, domain_app_pairs = single_domain(abs_root)
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

    if len(pairs) < len(unrelated_pairs) // 2:
        print(f"len(pairs) is {len(pairs)}, and len(unrelated_pairs) is {len(unrelated_pairs)}, balancing...")
        import math
        multiplier = math.ceil(len(unrelated_pairs) / len(pairs))
        pairs = pairs * multiplier

    pairs.extend(unrelated_pairs)
    pairs = add_perturbation(pairs)

    # 建立query->task映射表(在格式化之前)
    print("[实时] 建立 query->task 映射表")
    query2task = {}
    fmt = "Instruct: Represent this phone-use task description for searching similar tasks\nQuery:{}"
    
    # 为所有task建立映射关系
    for t1, t2, sim in pairs:
        formatted_q1 = fmt.format(t1)
        formatted_q2 = fmt.format(t2)
        query2task[formatted_q1] = t1
        query2task[formatted_q2] = t2
    
    print(f"[实时] query2task 映射表大小: {len(query2task)}")
    
    # 格式化pairs
    pairs = [(fmt.format(t1), fmt.format(t2), sim) for t1, t2, sim in pairs]

    def print_pair_distribution(pairs, name):
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

    print("[实时] 开始增强 pairs")
    t0 = time.time()
    aug_pairs = augment_self_pairs(pairs, query2task)
    print(f"[实时] 增强完成, 用时: {time.time()-t0:.2f}s, 增强数: {len(aug_pairs)}")

    print_pair_distribution(pairs + aug_pairs, "数据集")

    df = pd.DataFrame(pairs + aug_pairs, columns=['sentence1', 'sentence2', 'score'])
    df.to_csv('sts_mybench_data.csv', index=False, encoding='utf-8')

    with open('app_gap.json', 'w') as f:
        json.dump(app_gap, f, indent=4, ensure_ascii=False)