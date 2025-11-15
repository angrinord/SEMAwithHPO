import copy
import torch
from torch import nn
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensifier.successive_halving import SuccessiveHalving
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
import logging
from backbone.sema_components import Adapter


def hpo4adapter(model, dataset, n_samples=256, inner_steps=100, n_trials=100, device="cuda"):
    # subsample data
    n_samples = min(n_samples, len(dataset))
    subset_indices = np.random.choice(len(dataset), n_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    # define hp space
    cs = ConfigurationSpace()
    rank = UniformIntegerHyperparameter("rank", lower=16, upper=inner_steps)
    alpha = UniformFloatHyperparameter("alpha", lower=1.0, upper=64.0, log=True)
    dropout = UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5)
    # nonlinearity = CategoricalHyperparameter("adapter_activation", choices=["relu", "gelu"])
    learn_rate = UniformFloatHyperparameter("learn_rate", lower=1e-5, upper=1e-3, log=True)
    weight_decay = UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-3, log=True)
    # cs.add([rank, alpha, dropout, nonlinearity, learn_rate, weight_decay])
    cs.add([rank, alpha, dropout, learn_rate, weight_decay])

    def train_and_eval(cfg, seed: int = 0, budget: float = 0.0, instance: str = None, rank_penalty: float = 0.01):
        try:
            local_model = copy.deepcopy(model).to(device)
            local_model.train()

            # Apply hyperparameters
            for module in local_model.modules():
                if isinstance(module, Adapter) and getattr(module, "is_new", False):
                    module.rank = int(cfg["rank"])
                    module.dropout = float(cfg["dropout"])
                    module.non_linear_func = getattr(torch.nn, cfg["activation"].capitalize())()
                    module.scale = float(cfg["alpha"])
                    nn.init.zeros_(module.up_proj.weight)

            optimizer = torch.optim.AdamW(
                local_model.parameters(),
                lr=float(cfg["learn_rate"]),
                weight_decay=float(cfg["weight_decay"])
            )

            losses = []
            for i, (_, x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                out = local_model(x)
                logits = out["logits"][:, :y.max().item() + 1]
                loss = torch.nn.functional.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if i >= inner_steps:
                    break

            avg_loss = float(np.mean(losses))
            penalty = rank_penalty * int(cfg["rank"])
            return avg_loss + penalty
        except Exception as e:
            logging.warning(f"[HPO] Trial failed: {e}")
            return float("inf")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    od = f"./smac_output/task_{timestamp}"
    scenario = Scenario(
        cs,
        name=f"adapter_hpo_task",
        deterministic=True,
        n_trials=n_trials,
        seed=0,
        output_directory=od,
        min_budget=10,
        max_budget=inner_steps,
    )

    initial_design = SobolInitialDesign(scenario, n_configs=4)
    intensifier = SuccessiveHalving(scenario)
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train_and_eval,
        initial_design=initial_design,
        intensifier=intensifier,
    )

    incumbent = smac.optimize()
    best_value = smac.runhistory.get_cost(incumbent)
    logging.info(f"[HPO] Best config={incumbent} with penalized loss={best_value:.4f}")
    return dict(incumbent)
