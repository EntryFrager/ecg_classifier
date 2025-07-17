import subprocess

datas = ["ds", "ds_with_metadata"]
models = ["model", "model_with_metadata"]
target_labels = ["sinus", "arrit", "tach", "brad", "afib"]

learning_rates = [0.01, 0.001, 0.0001]

alpha = [0.9, 0.8, 0.7]
beta = [0.1, 0.2, 0.3]


for model, data in zip(models, datas):
    for target_label in target_labels:
        for lr in learning_rates:
            for a, b in zip(alpha, beta):
                if target_label == "sinus":
                    a = b
                    b = 1 - a

                cmd = f"ecg_classifier data={data} target_labels={target_label} model={model} train.alpha={a} train.beta={b} optimizer.lr={lr}"
                print(
                    f"Running setup:\n"
                    f"\tdata={data}\n"
                    f"\ttarget_labels={target_label}\n"
                    f"\tmodel={model}\n"
                    f"\ttrain.alpha={a}\n"
                    f"\ttrain.beta={b}\n"
                    f"\toptimizer.lr={lr}\n",
                    flush=True,
                )
                print(f"Command is: {cmd}\n", flush=True)
                subprocess.run(cmd, shell=True, check=True)
