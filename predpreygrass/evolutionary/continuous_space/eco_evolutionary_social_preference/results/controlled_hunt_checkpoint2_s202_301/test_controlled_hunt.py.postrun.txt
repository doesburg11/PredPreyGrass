import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns
from . import controlled_hunt as hunt


class ConstantModule:
    def __init__(self, stopped=False):self.stopped=stopped
    def forward_inference(self,batch):
        logits=torch.zeros((len(batch[Columns.OBS]),48))
        if self.stopped:
            logits[:]=-100.;logits[:,32]=100.
        return {Columns.ACTION_DIST_INPUTS:logits}


@pytest.mark.parametrize('seed',[202,203,204])
def test_matched_starting_states_genomes_grass_rng_and_no_initial_contacts(seed):
    rows={};envs=[]
    for condition in hunt.CONDITIONS:
        pool={'predator':np.array([[.1,0,0],[0,.1,0],[0,0,.1]]),'prey':np.array([[0,0,0],[.1,.1,0]])}
        env,_=hunt.setup_trial({'grass_count':12},pool,seed,condition);envs.append(env)
        rows[condition]={'initial_state':hunt.initial_state(env)}
        animals=env.predators+env.prey
        for i,a in enumerate(animals):
            for b in animals[i+1:]:assert env.torus.distance(a.position,b.position)>a.radius+b.radius
    hunt.check_matched(rows)
    assert len(envs[0].predators)==1 and len(envs[1].predators)==2
    assert len(envs[0].prey)==len(envs[1].prey)==1
    rows['two_attraction']['initial_state']['animals']['prey_0']['energy']+=1
    with pytest.raises(RuntimeError,match='starting states'):hunt.check_matched(rows)
    for env in envs:env.close()


def test_per_agent_action_noise_not_shifted_by_companion_and_zero_bias_exact():
    modules={'predator':ConstantModule(),'prey':ConstantModule()}
    obs={a:np.zeros(36,dtype=np.float32) for a in ('predator_0','prey_0')}
    generators={a:hunt.generator_for(202,a) for a in obs}
    expanded={**obs,'predator_1':np.zeros(36,dtype=np.float32)}
    more={a:hunt.generator_for(202,a) for a in expanded}
    attracted={a:hunt.generator_for(202,a) for a in obs}
    for _ in range(8):
        base,_,_=hunt.actions_for(modules,obs,generators,0,1)
        with_other,_,_=hunt.actions_for(modules,expanded,more,0,1)
        no_visible_bias,_,_=hunt.actions_for(modules,obs,attracted,1,1)
        assert base=={a:with_other[a] for a in obs}==no_visible_bias
        for a in obs:
            assert torch.equal(generators[a].get_state(),more[a].get_state())
            assert torch.equal(generators[a].get_state(),attracted[a].get_state())


def test_held_actions_manual_physics_matches_original_step():
    a,_=hunt.setup_trial({'grass_count':0},None,204,'two_neutral')
    b,_=hunt.setup_trial({'grass_count':0},None,204,'two_neutral')
    for _ in range(3):
        actions={agent:32 for agent in a.active}
        a.step(actions)
        for _ in range(b.action_repeat):b._physics_step(actions)
        assert hunt.initial_state(a)==hunt.initial_state(b)
        assert a.active==b.active and a.family.keys()==b.family.keys()
        for agent,obs in a._observations().items():np.testing.assert_array_equal(obs,b._observations()[agent])
    a.close();b.close()


@pytest.mark.parametrize('event',['capture','birth','timeout','target_starved','predators_starved'])
def test_exact_tick_stopping_preserves_original_birth_rule_and_energy_accounting(monkeypatch,event):
    original=hunt.setup_trial
    def setup(config,pool,seed,condition):
        env,d=original({**config,'grass_count':0},pool,seed,condition)
        if event=='capture':env.prey[0].position=env.predators[0].position.copy()
        if event=='birth':env.prey[0].energy=env.threshold['prey']+1.
        if event=='target_starved':env.prey[0].energy=1e-9
        if event=='predators_starved':
            for a in env.predators:a.energy=1e-9
        return env,d
    monkeypatch.setattr(hunt,'setup_trial',setup)
    row=hunt.trial({},None,{'predator':ConstantModule(True),'prey':ConstantModule(True)},202,'two_neutral')
    assert row['hunting']['max_abs_energy_balance_error']<1e-8
    if event=='capture':
        assert row['outcome']=='captured' and row['capture_tick']==row['observed_ticks']==1
    elif event=='birth':
        assert row['outcome']=='birth_stop' and row['births']['prey']==1 and row['observed_ticks']==1
        assert row['capture_success']==0 and row['penalized_capture_time']==256
    elif event in ('target_starved','predators_starved'):
        assert row['outcome']==event and row['observed_ticks']==1
        assert row['capture_success']==0 and row['penalized_capture_time']==256
    else:
        assert row['outcome']=='timeout' and row['observed_ticks']==256 and row['capture_tick'] is None


def test_success_rate_uncertainty_is_not_zero_at_all_failure_or_success():
    assert hunt.wilson_interval(0,100)[1]>.03
    assert hunt.wilson_interval(100,100)[0]<.97
    lo,hi=hunt.wilson_interval(50,100)
    assert lo<.5<hi


def test_summary_compares_against_two_neutral_and_keeps_failures():
    rows=[]
    for seed in (202,203):
        for c in hunt.CONDITIONS:
            success=int(c=='two_attraction' or (c=='two_neutral' and seed==202))
            r={m:1. for m in hunt.METRICS}
            r.update(seed=seed,condition=c,capture_success=success,capture_tick=32 if success else None,
                     penalized_capture_time=32 if success else 256,outcome='captured' if success else 'timeout',
                     initial_socially_visible_predators=0 if c=='one_neutral' else 2,
                     predator_decisions=10,social_visible_decisions=0 if c=='one_neutral' else 5)
            rows.append(r)
    s=hunt.trial_summary(rows)
    assert s['successes']=={'one_neutral':0,'two_neutral':1,'two_attraction':2}
    contrast=s['metrics']['capture_success']['differences_from_neutral']
    assert contrast['two_attraction']['mean']==.5 and contrast['one_neutral']['mean']==-.5
    assert s['discordant_capture_pairs']['two_attraction']==1
    assert s['metrics']['penalized_capture_time']['condition_means']['one_neutral']==256
    assert s['mean_capture_tick_successes_only']['one_neutral'] is None


def test_neutral_nonuniform_actions_match_original_with_visible_observations():
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.eval import select_actions
    class Nonuniform:
        def forward_inference(self,batch):
            return {Columns.ACTION_DIST_INPUTS:torch.linspace(-2.,2.,48)[None,:].repeat(len(batch[Columns.OBS]),1)}
    env,_=hunt.setup_trial({'grass_count':12},None,202,'two_neutral')
    obs=env._observations();modules={s:Nonuniform() for s in ('predator','prey')}
    actual={a:hunt.generator_for(202,a) for a in obs}
    expected={a:hunt.generator_for(202,a) for a in obs}
    for _ in range(6):
        chosen,_,_=hunt.actions_for(modules,obs,actual,0,env.observe_count['predator'])
        for a in obs:
            assert chosen[a]==select_actions(modules,{a:obs[a]},expected[a],True)[a]
            assert torch.equal(actual[a].get_state(),expected[a].get_state())
    env.close()


def test_core_source_guard_rejects_deleted_added_or_changed_files(tmp_path):
    a=tmp_path/'a.py';b=tmp_path/'b.py'
    a.write_text('a=1\n');b.write_text('b=2\n')
    expected={str(p):hunt.digest(p) for p in (a,b)}
    hunt.verify_core_sources(tmp_path,expected)
    b.unlink()
    with pytest.raises(RuntimeError,match='set/hashes'):hunt.verify_core_sources(tmp_path,expected)
    b.write_text('b=2\n');c=tmp_path/'c.py';c.write_text('c=3\n')
    with pytest.raises(RuntimeError,match='set/hashes'):hunt.verify_core_sources(tmp_path,expected)
    c.unlink();a.write_text('a=4\n')
    with pytest.raises(RuntimeError,match='set/hashes'):hunt.verify_core_sources(tmp_path,expected)
