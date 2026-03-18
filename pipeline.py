from pathlib import Path

from reddit_parser import parse_folder_to_json
from sentiment_classify_reddit_comments import (
    classify_json_files_to_csvs,
    combine_csv_files,
)
from sentiment_plots_reddit import generate_all_plots
from report_generator import generate_pdf_report


def run_category_pipeline(category_folder: str | Path, logger=None):
    category_folder = Path(category_folder)
    category_name = category_folder.name

    outputs_dir = category_folder / "outputs"
    json_dir = outputs_dir / "json"
    csv_dir = outputs_dir / "csv"
    plots_dir = outputs_dir / "plots"

    def log(msg):
        if logger:
            logger(msg)

    txt_files = sorted(category_folder.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in folder: {category_folder}")

    log(f"Category selected: {category_name}")
    log(f"Found {len(txt_files)} txt files")

    json_files = parse_folder_to_json(category_folder, json_dir)
    log(f"Created {len(json_files)} JSON files")

    csv_files = classify_json_files_to_csvs(json_files, csv_dir)
    log(f"Created {len(csv_files)} CSV files")

    combined_csv = csv_dir / f"{category_name}_combined_sentiment.csv"
    combine_csv_files(csv_files, combined_csv)
    log(f"Created combined CSV: {combined_csv.name}")

    plot_paths = generate_all_plots(combined_csv, plots_dir)
    log(f"Generated {len(plot_paths)} plots")

    report_dir = outputs_dir / "report"
    report_pdf = report_dir / f"{category_name}_report.pdf"

    report_pdf = generate_pdf_report(
        category_name=category_name,
        combined_csv_path=combined_csv,
        plot_paths=plot_paths,
        output_pdf_path=report_pdf,
        txt_count=len(txt_files),
        json_count=len(json_files),
        csv_count=len(csv_files),
    )

    log(f"Generated PDF report: {report_pdf.name}")

    return {
        "category_name": category_name,
        "json_files": json_files,
        "csv_files": csv_files,
        "combined_csv": combined_csv,
        "plot_paths": plot_paths,
        "report_pdf": report_pdf,
        "output_dir": outputs_dir,
    }