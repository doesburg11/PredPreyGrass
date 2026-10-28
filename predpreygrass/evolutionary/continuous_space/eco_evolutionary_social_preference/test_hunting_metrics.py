from types import MethodType, SimpleNamespace
import pytest
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous
from .hunting_metrics import HuntingObserver
from .individual_preferences import IndividualPreferences
from .test_persistence import world


def setup(energy=9.):
    env = world(0, {'predator_0':(0,0)})
    animal = env.predators[0]
    animal.energy=energy;animal.species='predator';animal.alive=True;animal.cost_multiplier=1.
    animal.velocity=SimpleNamespace(mag=lambda:2.)
    env.entities=[animal];env.active=[animal.id]
    env.family={animal.id:['predator',None,0]}
    env.max_energy={'predator':10.};env.initial_energy={'predator':2.}
    env.speed_cost={'predator':.2};env.resting_cost={'predator':.1};env.catch_efficiency=.5
    env._gain=MethodType(PredPreyGrassContinuous._gain,env)
    env._energy=MethodType(PredPreyGrassContinuous._energy,env)
    return env, animal


def run_energy(env, catches):
    env.time_step+=1
    terms={};truncs={};infos={}
    env._energy(catches,{},terms,truncs,infos)
    return terms,truncs,infos


def test_real_energy_code_caps_multiple_catches_and_hooks_do_not_change_state():
    env,a=setup();plain,b=setup()
    p=IndividualPreferences('neutral');p.start_episode(env,1)
    observer=HuntingObserver(p,trace_seeds=());observer.reset(env,1)
    actual=run_energy(env,[(a,4.),(a,6.)])
    expected=run_energy(plain,[(b,4.),(b,6.)])
    observer.sample(env)
    assert actual==expected and a.energy==b.energy==9.5
    hunting=observer.finish(env)['hunting']
    row=hunting['individuals'][0]
    assert row['captures']==2 and row['offered_hunting_energy']==5.
    assert row['retained_hunting_energy']==1.
    assert row['movement_energy']==.4 and row['resting_energy']==.1
    assert row['predator_ticks']==1 and abs(row['energy_balance_error'])<1e-12
    assert hunting['groups']['0']['all']['storage_cap_loss']==4.
    assert hunting['groups']['1']['all']['captures_per_1000_predator_ticks'] is None
    # Reset on the same environment must not double-wrap energy/gain.
    energy_hook=env._energy;gain_hook=env._gain
    env.time_step=0;p.start_episode(env,2);observer.reset(env,2)
    assert env._energy is energy_hook and env._gain is gain_hook
    run_energy(env,[]);observer.sample(env)
    assert observer.finish(env)['hunting']['individuals'][0]['predator_ticks']==1


def test_death_interval_cost_and_negative_terminal_energy_balance():
    env,a=setup(.3)
    p=IndividualPreferences('neutral');p.start_episode(env,1)
    observer=HuntingObserver(p,trace_seeds=());observer.reset(env,1)
    run_energy(env,[]);observer.sample(env)
    assert not env.predators
    row=observer.finish(env)['hunting']['individuals'][0]
    assert row['predator_ticks']==1 and row['final_energy']==pytest.approx(-.2)
    assert abs(row['energy_balance_error'])<1e-12


def test_birth_cost_and_final_tick_newborn_with_zero_exposure():
    env,a=setup()
    p=IndividualPreferences('neutral');p.start_episode(env,1)
    observer=HuntingObserver(p,trace_seeds=());observer.reset(env,1)
    run_energy(env,[(a,4.)])
    a.energy-=2.;env.family[a.id][2]=1
    child=SimpleNamespace(id='predator_1',energy=2.,species='predator',alive=True,position=SimpleNamespace(x=1,y=0))
    env.predators.append(child);env.family[child.id]=['predator',None,0]
    p.register(env);observer.sample(env)
    result=observer.finish(env)['hunting']
    parent,born=result['individuals']
    assert parent['reproduction_energy']==2. and parent['final_energy']==7.5
    assert born['predator_ticks']==0 and born['initial_energy']==born['final_energy']==2.
    assert result['max_abs_energy_balance_error']<1e-12
    # Corrupting the ledger must fail rather than report fabricated efficiency.
    observer.budgets[a.id]['retained_hunting_energy']+=1.
    with pytest.raises(RuntimeError,match='ledger'):observer.finish(env)
