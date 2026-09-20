"""Transparent stochastic dynamics baseline for Scenario Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping,Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ScenarioShock:
    variable:str
    step:int
    operation:str
    value:float

    def __post_init__(self):
        if self.operation not in {"add","multiply","replace"}:
            raise ValueError("operation must be add, multiply or replace")
        if self.step<0:
            raise ValueError("step must be non-negative")


@dataclass(frozen=True)
class ScenarioDistribution:
    variables:tuple[str,...]
    median:pd.DataFrame
    lower:pd.DataFrame
    upper:pd.DataFrame
    runs:int
    lower_quantile:float
    upper_quantile:float


class StochasticVARScenarioModel:
    """Ridge-regularized VAR(1) with residual Monte-Carlo uncertainty.

    This is a conditional dynamics model, not a causal estimator. A user shock
    changes the simulated state transition by construction; the resulting delta
    must not be interpreted as an identified causal effect.
    """

    def __init__(self,alpha:float=1.0)->None:
        self.alpha=float(alpha)
        self.scaler=StandardScaler()
        self.model=Ridge(alpha=self.alpha)
        self.variables:tuple[str,...]=()
        self.residual_covariance:np.ndarray|None=None
        self._fitted=False

    def fit(self,frame:pd.DataFrame)->"StochasticVARScenarioModel":
        numeric=frame.select_dtypes(include=[np.number]).dropna()
        if len(numeric)<10 or numeric.shape[1]<1:
            raise ValueError("at least 10 complete numeric observations are required")
        self.variables=tuple(str(c) for c in numeric.columns)
        z=self.scaler.fit_transform(numeric)
        x=z[:-1]
        y=z[1:]
        self.model.fit(x,y)
        residuals=y-self.model.predict(x)
        if residuals.shape[0]>1:
            cov=np.cov(residuals,rowvar=False)
        else:
            cov=np.eye(len(self.variables))*1e-6
        cov=np.atleast_2d(cov).astype(float)
        cov+=np.eye(cov.shape[0])*1e-8
        self.residual_covariance=cov
        self._fitted=True
        return self

    def _apply_shock(self,state:np.ndarray,shock:ScenarioShock)->np.ndarray:
        index=self.variables.index(shock.variable)
        out=state.copy()
        if shock.operation=="add":
            out[index]+=shock.value
        elif shock.operation=="multiply":
            out[index]*=shock.value
        else:
            out[index]=shock.value
        return out

    def simulate(
        self,
        *,
        initial_state:Mapping[str,float],
        steps:int,
        runs:int=1000,
        shocks:Sequence[ScenarioShock]=(),
        seed:int=42,
        lower_quantile:float=0.10,
        upper_quantile:float=0.90,
    )->ScenarioDistribution:
        if not self._fitted or self.residual_covariance is None:
            raise RuntimeError("fit the model before simulation")
        if steps<1 or runs<1:
            raise ValueError("steps and runs must be positive")
        if not 0<lower_quantile<upper_quantile<1:
            raise ValueError("invalid quantiles")

        initial=np.array([float(initial_state[v]) for v in self.variables],dtype=float)
        shock_map:dict[int,list[ScenarioShock]]={}
        for shock in shocks:
            if shock.variable not in self.variables:
                raise KeyError(shock.variable)
            shock_map.setdefault(shock.step,[]).append(shock)

        rng=np.random.default_rng(seed)
        paths=np.zeros((runs,steps+1,len(self.variables)),dtype=float)
        paths[:,0,:]=initial

        for run in range(runs):
            state=initial.copy()
            for step in range(1,steps+1):
                z=self.scaler.transform(pd.DataFrame([state],columns=self.variables))[0]
                predicted_z=np.asarray(self.model.predict(z.reshape(1,-1))[0],dtype=float)
                noise=rng.multivariate_normal(
                    np.zeros(len(self.variables)),
                    self.residual_covariance,
                )
                state=self.scaler.inverse_transform((predicted_z+noise).reshape(1,-1))[0]
                for shock in shock_map.get(step,()):
                    state=self._apply_shock(state,shock)
                paths[run,step,:]=state

        index=pd.RangeIndex(0,steps+1,name="step")
        def frame(q:float)->pd.DataFrame:
            return pd.DataFrame(
                np.quantile(paths,q,axis=0),
                index=index,
                columns=self.variables,
            )
        return ScenarioDistribution(
            variables=self.variables,
            median=frame(0.5),
            lower=frame(lower_quantile),
            upper=frame(upper_quantile),
            runs=runs,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
