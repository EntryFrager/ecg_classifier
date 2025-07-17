import subprocess

datas = ["ds", "ds_with_metadata"]
models = ["model", "model_with_metadata"]
keys_target_label = ["sinus", "arrit", "tach", "brad", "afib"]

learning_rates = [0.01, 0.001, 0.0001]

alphas = [0.9, 0.8, 0.7]

for model, data in zip(models, datas):
    for key_target_label in keys_target_label:
        for lr in learning_rates:
            for alpha in alphas:
                if key_target_label == "sinus":
                    alpha = 1 - alpha

                cmd = f"ecg_classifier data={data} data.key_target_label={key_target_label} model={model} train.alpha={alpha} optimizer.lr={lr}"
                print(
                    f"Running setup:\n"
                    f"\tdata={data}\n"
                    f"\ttarget_labels={key_target_label}\n"
                    f"\tmodel={model}\n"
                    f"\ttrain.alpha={alpha}\n"
                    f"\toptimizer.lr={lr}\n",
                    flush=True,
                )
                print(f"Command is: {cmd}\n", flush=True)
                subprocess.run(cmd, shell=True, check=True)
