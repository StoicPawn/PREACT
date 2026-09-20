import numpy as np
import pandas as pd

from preact.models.scenario_dynamics import ScenarioShock,StochasticVARScenarioModel

def test_stochastic_var_scenario_is_reproducible_and_distributed():
    rng=np.random.default_rng(1)
    x=[0.0]
    y=[1.0]
    for _ in range(79):
        x.append(0.8*x[-1]+0.15*y[-1]+rng.normal(0,0.05))
        y.append(0.7*y[-1]+rng.normal(0,0.05))
    frame=pd.DataFrame({"x":x,"y":y})
    model=StochasticVARScenarioModel(alpha=0.5).fit(frame)
    a=model.simulate(
        initial_state={"x":x[-1],"y":y[-1]},
        steps=6,runs=100,seed=7,
        shocks=[ScenarioShock("x",2,"add",1.0)],
    )
    b=model.simulate(
        initial_state={"x":x[-1],"y":y[-1]},
        steps=6,runs=100,seed=7,
        shocks=[ScenarioShock("x",2,"add",1.0)],
    )
    assert a.median.equals(b.median)
    assert a.lower.shape==(7,2)
    assert (a.lower<=a.upper).all().all()
    assert a.median.loc[2,"x"]>a.median.loc[1,"x"]
