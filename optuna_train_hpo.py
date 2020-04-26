"""

An example of hyperparameter optimization using AllenNLPExecutor.
It requires sqlite3 to run this example.

"""

import optuna


config_file = "training_config/rc/transformer_qa_optuna.jsonnet"
result_path = "result_hpo/trial_{}"
metric = "best_validation_per_instance_em"


def objective(trial: optuna.Trial) -> float:
    # begin hyperparameter space
    trial.suggest_float("grad_clipping", 0.0, 5.0)
    trial.suggest_float("lr", 1e-5, 3e-5, log=True)
    # end   hyperparameter space

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial,  # trial object
        config_file,  # jsonnet path
        result_path.format(trial.number),  # directory for snapshots and logs
        metric,  # metric which you want to track
        include_package="allennlp_models"  # same as `--include-package` in allennlp
    )
    return executor.run()


if __name__ == '__main__':
    study = optuna.create_study(
        storage="sqlite:///examle.db",  # save results in DB
        study_name="optuna_allennlp_demo",
        direction="maximize",
    )

    timeout = 60 * 60 * 5  # timeout (sec): 60*60*5 sec => 5hours
    study.optimize(
        objective,
        n_jobs=1,  # number of processes in parallel execution
        n_trials=10,  # number of trials to train a model
        timeout=timeout,  # threshold for executing time (sec)
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
