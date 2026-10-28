from copy import deepcopy
import pytest
from .hunting_replay import check_replay, hunting_summary, RATE_METRICS
from .hunting_metrics import TOTALS


def test_replay_allows_only_new_hunting_field_and_spell_order():
    reference={'seed':102,'persistence':{'fitness':{'offspring':3},'spells':{'spell_records':[[1,2],[3,4]]}}}
    row=deepcopy(reference)
    row['persistence']['hunting']={'captures':4}
    row['persistence']['spells']['spell_records'].reverse()
    check_replay(row,reference)
    row['persistence']['fitness']['offspring']=4
    with pytest.raises(RuntimeError,match='previous saved fields'):check_replay(row,reference)


def test_hunting_summary_uses_paired_episode_rates_and_separate_raw_totals():
    episodes=[]
    for seed in (102,103):
        groups={}
        for value in (-1,0,1):
            row={key:0 for key in TOTALS}
            row.update(captures=2+value,predator_ticks=1000)
            row.update({key:2.+value for key in RATE_METRICS})
            groups[str(value)]={cohort:dict(row) for cohort in ('all','founders','born')}
        episodes.append({'seed':seed,'persistence':{'physics_ticks':8000,'hunting':{'groups':groups,'max_abs_energy_balance_error':1e-12}}})
    result=hunting_summary(episodes)
    entry=result['summary']['metrics']['all_captures_per_1000_predator_ticks']
    assert entry['paired_episode_count']==2
    assert entry['differences_from_neutral']['attraction']['mean']==1.
    assert result['totals']['attraction']['all']['captures']==6
    assert result['max_abs_individual_energy_balance_error']==1e-12
    assert result['early_episode_seeds']==[]
    with pytest.raises(ValueError,match='duplicate seeds'):hunting_summary(episodes+[deepcopy(episodes[0])])
