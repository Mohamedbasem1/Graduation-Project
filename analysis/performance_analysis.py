import io
import os
import sys
from collections import defaultdict
from zipfile import ZipFile

import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())

from analysis.retrieval_ensemble import retrieval_ensemble, optimal_retrieval_ensemble
from data_scripts import read_qrels_file
from metrics.Custom_TaskA_eval import evaluate_task_a
from metrics.helpers import apply_threshold, apply_cutoff

CUT_OFF = 10

def ensure_directory_exists(directory):
    """Ensure the specified directory exists, create it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_from_zip(zip_file, file_to_read):
    """Extract a specific file from a zip and read it into a pandas DataFrame."""
    with io.BytesIO() as eval_dump_file_io:
        eval_dump_file_io.write(zip_file.read(file_to_read))
        eval_dump_file_io.seek(0)
        df_run = pd.read_csv(eval_dump_file_io, sep="\t", names=["qid", "Q0", "docid", "rank", "score", "tag"])
        df_run["docid"] = df_run["docid"].astype(str)
        df_run["qid"] = df_run["qid"].astype(str)
    return df_run

def evaluate_steps(raw_run_df, no_answer_threshold):
    """Evaluate the run using various thresholds and cutoffs."""
    raw_split_results, _ = evaluate_task_a(df_qrels, raw_run_df)
    rets = apply_threshold(raw_run_df, no_answer_threshold=no_answer_threshold)

    if no_answer_threshold is None:
        raw_run_df, quantile_threshold = rets
    else:
        raw_run_df = rets
        quantile_threshold = None

    thre_run_df = apply_cutoff(raw_run_df, cutoff=CUT_OFF)
    thre_split_results, thre_per_sample_results = evaluate_task_a(df_qrels, thre_run_df)

    # Propagate best values from the raw run to the thresholded run
    for key in ["best_map", "best_map_thresh", "best_recip", "best_recip_thresh"]:
        thre_split_results["overall"][key] = raw_split_results["overall"][key]
    thre_split_results["overall"]["quantile threshold"] = quantile_threshold
    
    return thre_split_results, thre_per_sample_results

def process_model_runs(root, files):
    """Process zip files in a directory and return the evaluation results."""
    model_runs = []
    summary_results = {}

    for dump_file in files:
        dump_file_path = os.path.join(root, dump_file)
        if dump_file.endswith(".zip"):
            with ZipFile(dump_file_path, "r") as zip_file:
                eval_file = [f.filename for f in zip_file.filelist if "eval" in f.filename]
                if len(eval_file) != 1:
                    continue  # Skip if there is not exactly one eval file
                run_df = read_from_zip(zip_file, file_to_read=eval_file[0])
                model_runs.append(run_df)
                split_results, per_sample_results = evaluate_steps(run_df, None)
                summary_results[dump_file] = (split_results, per_sample_results)

    return model_runs, summary_results

if __name__ == "__main__":
    all_results_df = []
    summary_dump = {}
    ensemble_runs = defaultdict(list)

    df_qrels = read_qrels_file("data/QQA23_TaskA_qrels_dev.gold")
    walk = list(os.walk("artifacts/dumps"))
    
    # Ensure the summary directory exists
    summary_dir = "artifacts/summary"
    ensure_directory_exists(summary_dir)

    for root, folders, files in tqdm(walk):
        if len(folders) == 0:  # Only process leaf directories
            model_runs, summary_results = process_model_runs(root, files)
            if model_runs:
                ensemble_runs[root].extend(model_runs)
                summary_dump[root] = summary_results

    # Process ensemble results
    ensemble_results = {}
    for model_dir, model_runs in ensemble_runs.items():
        ensemble_df = retrieval_ensemble(model_runs, cutoff=CUT_OFF)
        ensemble_split_results, ensemble_per_sample_results = evaluate_steps(ensemble_df, None)
        print(model_dir, ensemble_split_results["overall"]["best_map_thresh"])
        ensemble_results[model_dir] = (ensemble_split_results, ensemble_per_sample_results)

    # Collect all results and write to Excel
    ensemble_final_results = []
    for model_dir, model_results in summary_dump.items():
        model_results_df = pd.concat([pd.json_normalize(result[0]) for result in model_results.values()])
        model_results_df["file_name"] = model_results.keys()
        model_results_df["model_root"] = model_dir

        model_results_df.to_excel(os.path.join(summary_dir, f"{os.path.split(model_dir)[-1]}.xlsx"))
        all_results_df.append(model_results_df)

        optimal_ensemble = optimal_retrieval_ensemble([result[1] for result in model_results.values()])
        optimal_ensemble_map_cut_10 = sum([v["map_cut_10"] for v in optimal_ensemble.values()]) / len(optimal_ensemble)

        ensemble_final_results.append({
            "model_dir": model_dir,
            "original_avg_performance (map_cut_10)": model_results_df["overall.map_cut_10"].mean(),
            "original_min_performance (map_cut_10)": model_results_df["overall.map_cut_10"].min(),
            "actual_ensemble_result (map_cut_10)": ensemble_results[model_dir][0]["overall"]["map_cut_10"],
            "best_original_avg_performance (best_map)": model_results_df["overall.best_map"].mean(),
            "best_actual_ensemble_result (best_map)": ensemble_results[model_dir][0]["overall"]["best_map"],
            "optimal_ensemble (map_cut_10)": optimal_ensemble_map_cut_10,
            "original_avg_performance (single map_cut_10 )": model_results_df["single_answer.map_cut_10"].mean(),
            "original_avg_performance (multi map_cut_10)": model_results_df["multi_answer.map_cut_10"].mean(),
            "original_avg_performance (no answer map_cut_10)": model_results_df["no_answer.map_cut_10"].mean(),
            "original_avg_performance (recip_rank)": model_results_df["overall.recip_rank"].mean(),
            "best_original_avg_performance (best_recip_rank)": model_results_df["overall.best_recip"].mean(),
            "actual_ensemble_result (recip_rank)": ensemble_results[model_dir][0]["overall"]["recip_rank"],
            "best_actual_ensemble_result (best_recip_rank)": ensemble_results[model_dir][0]["overall"]["best_recip"],
            "original_avg_performance (recall_10)": model_results_df["overall.recall_10"].mean(),
            "original_avg_performance (recall_100)": model_results_df["overall.recall_100"].mean(),
            "num_models": len(model_results)
        })

    # Write final performance summary
    performance_df = pd.DataFrame(ensemble_final_results)
    performance_df.to_excel(os.path.join(summary_dir, f"ensemble_performance_df.{CUT_OFF}.xlsx"))

    # Write all model results
    pd.concat(all_results_df).to_excel(os.path.join(summary_dir, f"all_models_results.{CUT_OFF}.xlsx"))