from post_training_analysis import run_heldout_subject_finetuning_from_config

config = {
    "source_run": {
        "model_dir": "/data/result-gru-ffc255a1",
        "checkpoint_policy": "best_eval",
    },
    "heldout_subjects": {
        "test_subject_start": 10,
        "test_subject_end": 15
    },
    "heldout_finetuning": {
        "n_steps": 500,
        "lr": 1e-3,
        "checkpoint_every_n_steps": 100,
        "batch_size": 1024,
        "batch_mode": "random",
        "checkpoint_plot_split_examples_every_n": 100,
        "checkpoint_save_output_df_every_n": 0,
        "train_example_sessions_per_subject": 1,
        "eval_example_sessions_per_subject": 1,
        "example_max_subjects": 1,
        "keep_media_files": False,
    },
    "output": {
        "output_root": "/results/heldout_subject_finetuning",
        "run_name_suffix": None,
    },
    "wandb": {
        "project": "fine_tuning"
    },
    "seed": 17,
}

result = run_heldout_subject_finetuning_from_config(config)
print(result["output_dir"])
