import subprocess
import time
import re
import sys
import numpy as np


# -------------------------
# Settings
# -------------------------

TRAINING_SCRIPT = "prog3.py"  # Change to your training filename

TRAIN_X = "data\\multi_train_x.npy"
TRAIN_Y = "data\\multi_train_y.npy"
DEV_X = "data\\multi_dev_x.npy"
DEV_Y = "data\\multi_dev_y.npy"

LEARNING_RATE = 0.01
MAX_STEPS = 1000
PATIENCE = 5
DEV_FREQ = 50

MINIBATCH_SIZES = [8, 16, 32, 64]
REPEATS = 3


# -------------------------
# Run one experiment
# -------------------------

def run_one_test(batch_size, repeat):
    model_file = f"testing_mb{batch_size}_run{repeat}.npz"

    # Keep the random seed consistent across batch sizes.
    np.random.seed(1000 + repeat)

    command = [
        sys.executable,
        TRAINING_SCRIPT,
        "--train",
        "--trainx", TRAIN_X,
        "--trainy", TRAIN_Y,
        "--devx", DEV_X,
        "--devy", DEV_Y,
        "--model", model_file,
        "--lr", str(LEARNING_RATE),
        "--max-steps", str(MAX_STEPS),
        "--mb", str(batch_size),
        "--patience", str(PATIENCE),
        "--dev_freq", str(DEV_FREQ)
    ]

    print(f"\nTesting batch size {batch_size}, run {repeat}")
    print("-" * 50)

    start_time = time.time()

    # Run the training program and collect its printed output.
    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    total_time = time.time() - start_time
    output = process.stdout + "\n" + process.stderr

    # Show the training output.
    print(output)

    if process.returncode != 0:
        print("This experiment failed. Check the error above.")
        return None

    # Find development losses and accuracies in the output.
    results = []

    for line in output.splitlines():
        loss_match = re.search(
            r"dev loss\s*[=:]\s*([0-9.eE+-]+)", line, re.IGNORECASE
        )
        accuracy_match = re.search(
            r"(?:acc|accuracy)\s*[=:]\s*([0-9.eE+-]+)",
            line,
            re.IGNORECASE
        )

        if loss_match:
            loss = float(loss_match.group(1))
            accuracy = (
                float(accuracy_match.group(1))
                if accuracy_match else None
            )
            results.append({
                "loss": loss,
                "accuracy": accuracy,
                "time": time.time() - start_time
            })

    if not results:
        print(
            "Could not recognize the development results in the output. "
            "The experiment ran, but check the format of your dev output."
        )
        return {
            "batch": batch_size,
            "run": repeat,
            "loss": None,
            "accuracy": None,
            "time_to_best": None,
            "total_time": total_time
        }

    best_result = min(results, key=lambda result: result["loss"])

    return {
        "batch": batch_size,
        "run": repeat,
        "loss": best_result["loss"],
        "accuracy": best_result["accuracy"],
        "time_to_best": best_result["time"],
        "total_time": total_time
    }


# -------------------------
# Compare all batch sizes
# -------------------------

def main():
    all_results = []

    for batch_size in MINIBATCH_SIZES:
        for repeat in range(1, REPEATS + 1):
            result = run_one_test(batch_size, repeat)

            if result is not None:
                all_results.append(result)

    print("\n========== INDIVIDUAL RESULTS ==========")
    print(
        f"{'Batch':>6} | {'Run':>3} | {'Best Loss':>10} | "
        f"{'Accuracy':>9} | {'Time to Best':>12} | {'Total Time':>11}"
    )
    print("-" * 75)

    for result in all_results:
        loss = result["loss"]
        accuracy = result["accuracy"]
        best_time = result["time_to_best"]

        loss_text = f"{loss:.4f}" if loss is not None else "N/A"
        accuracy_text = (
            f"{accuracy:.4f}"
            if accuracy is not None else "N/A"
        )
        best_time_text = (
            f"{best_time:.2f}s"
            if best_time is not None else "N/A"
        )

        print(
            f"{result['batch']:>6} | "
            f"{result['run']:>3} | "
            f"{loss_text:>10} | "
            f"{accuracy_text:>9} | "
            f"{best_time_text:>12} | "
            f"{result['total_time']:>10.2f}s"
        )

    print("\n========== AVERAGES BY BATCH SIZE ==========")

    for batch_size in MINIBATCH_SIZES:
        results = [
            r for r in all_results
            if r["batch"] == batch_size and r["loss"] is not None
        ]

        if not results:
            print(f"Batch size {batch_size}: no usable results.")
            continue

        average_loss = np.mean([r["loss"] for r in results])

        accuracies = [
            r["accuracy"] for r in results
            if r["accuracy"] is not None
        ]

        times = [
            r["time_to_best"] for r in results
            if r["time_to_best"] is not None
        ]

        average_accuracy = (
            np.mean(accuracies) if accuracies else None
        )
        average_best_time = np.mean(times) if times else None
        average_total_time = np.mean(
            [r["total_time"] for r in results]
        )

        accuracy_text = (
            f"{average_accuracy:.4f}"
            if average_accuracy is not None else "N/A"
        )
        best_time_text = (
            f"{average_best_time:.2f}s"
            if average_best_time is not None else "N/A"
        )

        print(
            f"Batch {batch_size}: "
            f"average loss = {average_loss:.4f}, "
            f"average accuracy = {accuracy_text}, "
            f"average time to best = {best_time_text}, "
            f"average total time = {average_total_time:.2f}s"
        )


if __name__ == "__main__":
    main()