"""Stage 9 dataset: hidden labels supervise visible trajectories only."""
from __future__ import annotations
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from models.intent_gru import trajectory_to_features

TRAIN_FAMILIES = ('honest', 'scripted_1', 'scripted_2', 'scripted_3', 'scripted_4')
TEST_FAMILIES = ('scripted_5', 'delayed_deception', 'detour_deception', 'multi_switch_deception')

def validate_record(record):
    points = record['trajectory']
    if not points or any(len(p) != 3 or any(type(v) is not int for v in p) for p in points):
        raise ValueError('Expected nonempty visible [time,row,column] integer sequence')
    if any(not 0 <= p[0] <= 30 or not 0 <= p[1] <= 6 or not 0 <= p[2] <= 6 for p in points):
        raise ValueError('Trajectory coordinates/time outside GridWorld bounds')
    if any(a[0] >= b[0] for a,b in zip(points, points[1:])):
        raise ValueError('Visible timestamps must be strictly increasing')
    if record['goal_label'] not in (0,1) or record['deception_label'] not in (0,1):
        raise ValueError('Labels must be binary')

def load_stage6(training_csv, scaling_csv, family_csv, observer='ModelA'):
    """Exclude RandomPolicy and steps 6/7; do not duplicate observer variants."""
    candidates=[];test=[]
    for path,kind in [(training_csv,'train'),(scaling_csv,'scaling'),(family_csv,'family')]:
        path=Path(path)
        with path.open(newline='',encoding='utf-8') as handle:
            for index,row in enumerate(csv.DictReader(handle)):
                if kind=='train':
                    policy=row['deceiver_policy']
                    if policy=='honest_shortest_path':family='honest'
                    elif policy=='scripted_deception' and row['deception_steps'] in ('1','2','3','4'):
                        family='scripted_'+row['deception_steps']
                    else:continue
                elif row['model']!=observer:continue
                elif kind=='scaling':
                    if row['deception_steps']!='5':continue
                    family='scripted_5'
                else:
                    family=row['deception_family']
                    if family not in TEST_FAMILIES[1:]:continue
                if row['true_goal'] not in ('A','B'):raise ValueError('Unknown goal label')
                record=dict(sample_id=f'{path.name}:{index}',family=family,
                            trajectory=json.loads(row['trajectory']),goal_label=int(row['true_goal']=='B'),
                            deception_label=int(family!='honest'),source=str(path),
                            observer=row.get('model','rule_estimator'),seed=row.get('episode_seed',row.get('seed','')))
                validate_record(record)
                (candidates if kind=='train' else test).append(record)
    if set(r['family'] for r in candidates)!=set(TRAIN_FAMILIES):raise ValueError('Missing required training family')
    if set(r['family'] for r in test)!=set(TEST_FAMILIES):raise ValueError('Missing required unseen test family')
    return candidates,test

def family_split(records, validation_families=('scripted_4',)):
    """Whole variants are held out; duplicate episodes stay with their family."""
    families=set(r['family'] for r in records);held=set(validation_families)
    if not held or not held < families:raise ValueError('Validation must hold out a nonempty proper subset of families')
    train=[r for r in records if r['family'] not in held];val=[r for r in records if r['family'] in held]
    if {r['deception_label'] for r in train} != {0,1}:
        raise ValueError('Training must retain both Honest and deceptive samples')
    # Inputs repeated across different training/validation variants would leak memorized observations.
    key=lambda r:tuple(tuple(p) for p in r['trajectory'])
    if {key(r) for r in train} & {key(r) for r in val}:
        raise ValueError('Identical visible trajectories cross train/validation families')
    return train,val

def balanced_limit(records, per_family=None, seed=0):
    if per_family is None:return list(records)
    if per_family<2:raise ValueError('Per-family limit must be >=2')
    rng=random.Random(seed);groups=defaultdict(list)
    for r in records:groups[(r['family'],r['goal_label'])].append(r)
    result=[]
    for (family,goal),rows in sorted(groups.items()):
        rng.shuffle(rows);result.extend(rows[:per_family//2+(goal==0 and per_family%2)])
    return result

def audit(splits):
    report={}
    key=lambda r:json.dumps(r['trajectory'],separators=(',',':'))
    for name,rows in splits.items():
        report[name]=dict(episodes=len(rows),families=dict(Counter(r['family'] for r in rows)),
                         goals=dict(Counter(str(r['goal_label']) for r in rows)),
                         deception_labels=dict(Counter(str(r['deception_label']) for r in rows)),
                         unique_visible_trajectories=len({key(r) for r in rows}))
    conflicts=[]
    train_keys=defaultdict(list)
    for r in splits['train']:train_keys[key(r)].append(r)
    for family in sorted({r['family'] for r in splits['test']}):
        rows=[r for r in splits['test'] if r['family']==family]
        overlap=[r for r in rows if key(r) in train_keys]
        conflicts.append(dict(family=family,episodes=len(rows),observations_seen_in_train=len(overlap),
                              conflicting_goal_labels=sum(any(x['goal_label']!=r['goal_label'] for x in train_keys[key(r)]) for r in overlap),
                              conflicting_deception_labels=sum(any(x['deception_label']!=r['deception_label'] for x in train_keys[key(r)]) for r in overlap)))
    report['test_observation_aliases']=conflicts
    report['notes']=['Deception labels describe generating policy intent, not successful misdirection.',
                     'Single-class deception AUROC is undefined; return null, never 0 or 1.',
                     'Training uses rule-estimator visibility; unseen tests use fixed ModelA visibility.',
                     'Retain cross-family test observation aliases: removing them would remove scripted_5 entirely.']
    return report

def export_dataset(destination, splits):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():
        with (destination/(name+'.jsonl')).open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (destination/'audit.json').write_text(json.dumps(audit(splits),indent=2),encoding='utf-8')

class DeceptionDataset(Dataset):
    def __init__(self, records):
        if isinstance(records,(str,Path)):
            with Path(records).open(encoding='utf-8') as f:records=[json.loads(line) for line in f if line.strip()]
        self.records=list(records)
        if not self.records:raise ValueError('Dataset is empty')
        for r in self.records:validate_record(r)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        return trajectory_to_features(r['trajectory']),r['goal_label'],r['deception_label']

def collate(batch):
    sequences,goals,deceptions=zip(*batch)
    return (pad_sequence(sequences,batch_first=True),torch.tensor([len(s) for s in sequences]),
            torch.tensor(goals,dtype=torch.long),torch.tensor(deceptions,dtype=torch.long))
