from types import SimpleNamespace
import numpy as np
import pytest

from .individual_preferences import IndividualPreferences
from .preference_metrics import PreferencePersistenceObserver
from .test_persistence import world


def test_balanced_fixed_and_order_independent_without_simulation_rng():
    env = world(0, {f'predator_{i}': (i,0) for i in range(6)})
    rng_before = np.random.get_state()
    a = IndividualPreferences(); a.start_episode(env, 82)
    b = IndividualPreferences(); b.start_episode(env, 82)
    for block in range(10):
        ids = [f'predator_{3*block+i}' for i in range(3)]
        assert sorted(a.value_for_id(agent) for agent in ids) == [-1,0,1]
        assert [a.value_for_id(agent) for agent in ids] == [b.value_for_id(agent) for agent in ids]
    assert a.result(env)['founder_counts'] == {'-1': 2,'0': 2,'1': 2}
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    original = dict(a.assignments)
    a.register(world(1, {'predator_0':(0,0), 'predator_6':(1,0)}))
    assert all(a.assignments[k] == v for k,v in original.items())
    assert a.assignments['predator_6']['founder'] is False
    assert a.result(env)['inherited'] is False


def test_seed_changes_assignment_and_restart_is_reproducible():
    env = world(0, {f'predator_{i}': (i,0) for i in range(30)})
    p = IndividualPreferences(); p.start_episode(env, 82)
    first = [p.get(a.id) for a in env.predators]
    p.start_episode(env, 83)
    assert first != [p.get(a.id) for a in env.predators]
    p.start_episode(env, 82)
    assert first == [p.get(a.id) for a in env.predators]
    with pytest.raises(ValueError): p.value_for_id('prey_0')


def test_uniform_modes_and_birth_not_copied_from_parent():
    env = world(0, {'predator_0':(0,0)})
    for mode,value in [('neutral',0),('avoidance',-1),('attraction2',1)]:
        p=IndividualPreferences(mode);p.start_episode(env,82)
        assert p.get('predator_0')==value
        p.register(world(1,{'predator_0':(0,0),'predator_3':(1,0)}))
        assert p.get('predator_3')==value
    p=IndividualPreferences();p.start_episode(env,82)
    # Over many potential children the assignments differ from the focal value;
    # assigning a child's ID never reads the parent or its preference.
    assert any(p.value_for_id(f'predator_{i}')!=p.get('predator_0') for i in range(6,30))


def test_category_exposures_partition_global_measurements():
    env=world(0,{f'predator_{i}':(i*2,0) for i in range(6)})
    p=IndividualPreferences();p.start_episode(env,82)
    observer=PreferencePersistenceObserver(p)
    observer.reset(env,82)
    for tick in (1,2):
        env=world(tick,{f'predator_{i}':(i*2,0) for i in range(6)})
        p.register(env);observer.sample(env)
    result=observer.finish(env)
    groups=result['preference_groups'];e=groups['exposure']
    assert sum(e[f'predator_ticks_{v}'] for v in (-1,0,1))==result['exposure']['predator_ticks']
    assert sum(value for key,value in e.items() if key.startswith('close_pair_ticks_'))==result['exposure']['close_pair_ticks_100']
    assert sum(value for key,value in e.items() if key.startswith('potential_pair_ticks_'))==result['exposure']['potential_pair_ticks']
    assert sum(row['spells']['left_censored_spells'] for row in groups['pairs'].values())==15


@pytest.mark.parametrize('mode', ['neutral','mixed'])
def test_evaluator_registers_newborn_and_uses_each_fixed_value(monkeypatch, mode):
    import importlib
    import torch
    from ray.rllib.core.columns import Columns
    evaluator=importlib.import_module('.evaluate',__package__)
    instances=[]
    class FakeEnv:
        width=height=800
        distance_scale=400
        def __init__(self,config): instances.append(self)
        def reset(self,seed):
            self.time_step=0;self.history=[]
            self.predators=[SimpleNamespace(id=f'predator_{i}',position=SimpleNamespace(x=20*i,y=0)) for i in range(3)]
            return self.observations(),{}
        def observations(self):
            result={}
            for a in self.predators:
                obs=np.zeros(36,dtype=np.float32)
                from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import SELF_SIZE
                obs[SELF_SIZE:SELF_SIZE+4]=[1,0,-.5,.5]
                result[a.id]=obs
            return result
        def _physics_step(self,actions):
            self.history.append(dict(actions));self.time_step+=1
            if self.time_step==1:self.predators.append(SimpleNamespace(id='predator_3',position=SimpleNamespace(x=60,y=0)))
        def step(self,actions):
            self._physics_step(actions)
            ended=self.time_step==2
            return {} if ended else self.observations(),{}, {'__all__':ended},{},{}
        def close(self):pass
    class Module:
        def forward_inference(self,batch):
            return {Columns.ACTION_DIST_INPUTS:torch.zeros((len(batch[Columns.OBS]),48))}
    monkeypatch.setattr(evaluator,'PredPreyGrassContinuous',FakeEnv)
    monkeypatch.setattr(evaluator,'checkpoint_env_config',lambda _: {'predator_observe_count':1})
    monkeypatch.setattr(evaluator,'load_gene_pool',lambda _:None)
    monkeypatch.setattr(evaluator,'use_gene_pool',lambda *_:None)
    monkeypatch.setattr(evaluator,'capture',lambda env:({a.id:np.array([a.position.x,a.position.y]) for a in env.predators},{}))
    seen=[];original=evaluator.biased_logits
    def bias(logits,obs,pref,strength,slots):
        seen.append(pref);return original(logits,obs,pref,strength,slots)
    monkeypatch.setattr(evaluator,'biased_logits',bias)
    preferences=IndividualPreferences(mode)
    row=evaluator.evaluate('unused',0,1,2,82,{'predator':Module()},individual_preferences=preferences)[0]
    assert row['individual_preferences']['assignments']['predator_3']['first_seen_tick']==1
    assert row['neutral_original_actions_verified']==(mode=='neutral')
    # First step: 3 actual inputs then 9 counterfactual diagnostic calls.
    # Second step begins with 4 actual inputs, including the newborn.
    assert seen[:3]==[preferences.get(f'predator_{i}') for i in range(3)]
    assert seen[12:16]==[preferences.get(f'predator_{i}') for i in range(4)]
    assert row['individual_preferences']['inherited'] is False
    if mode=='neutral':
        evaluator.evaluate('unused',0,1,2,82,{'predator':Module()})
        assert instances[0].history==instances[1].history
    else:
        assert set(preferences.get(f'predator_{i}') for i in range(3))=={-1,0,1}
