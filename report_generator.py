from pathlib import Path
from datetime import datetime
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate_pdf_report(
    category_name: str,
    combined_csv_path: str | Path,
    plot_paths,
    output_pdf_path: str | Path,
    txt_count: int,
    json_count: int,
    csv_count: int,
) -> Path:
    combined_csv_path = Path(combined_csv_path)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(combined_csv_path)

    total_comments = len(df)
    sentiment_counts = df["sentiment"].value_counts().to_dict()

    positive_count = sentiment_counts.get("positive", 0)
    neutral_count = sentiment_counts.get("neutral", 0)
    negative_count = sentiment_counts.get("negative", 0)

    if total_comments > 0:
        positive_pct = positive_count / total_comments * 100
        neutral_pct = neutral_count / total_comments * 100
        negative_pct = negative_count / total_comments * 100
    else:
        positive_pct = neutral_pct = negative_pct = 0

    c = canvas.Canvas(str(output_pdf_path), pagesize=A4)
    page_width, page_height = A4

    def draw_title_page():
        y = page_height - 3 * cm
        c.setFont("Helvetica-Bold", 20)
        c.drawString(2 * cm, y, "Reddit Sentiment Analysis Report")

        y -= 1.5 * cm
        c.setFont("Helvetica", 13)
        c.drawString(2 * cm, y, f"Category: {category_name}")

        y -= 0.8 * cm
        c.drawString(2 * cm, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y -= 1.5 * cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y, "Summary")

        y -= 1.0 * cm
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, y, f"TXT files processed: {txt_count}")
        y -= 0.7 * cm
        c.drawString(2 * cm, y, f"JSON files created: {json_count}")
        y -= 0.7 * cm
        c.drawString(2 * cm, y, f"CSV files created: {csv_count}")
        y -= 0.7 * cm
        c.drawString(2 * cm, y, f"Total comments analyzed: {total_comments}")

        y -= 1.2 * cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y, "Sentiment Results")

        y -= 1.0 * cm
        c.setFont("Helvetica", 12)
        c.drawString(2 * cm, y, f"Positive: {positive_count} ({positive_pct:.1f}%)")
        y -= 0.7 * cm
        c.drawString(2 * cm, y, f"Neutral: {neutral_count} ({neutral_pct:.1f}%)")
        y -= 0.7 * cm
        c.drawString(2 * cm, y, f"Negative: {negative_count} ({negative_pct:.1f}%)")

        c.showPage()

    def draw_image_page(image_path, title):
        image_path = Path(image_path)
        if not image_path.exists():
            return

        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, page_height - 2 * cm, title)

        max_width = page_width - 4 * cm
        max_height = page_height - 5 * cm

        img = ImageReader(str(image_path))
        img_width, img_height = img.getSize()

        scale = min(max_width / img_width, max_height / img_height)
        draw_width = img_width * scale
        draw_height = img_height * scale

        x = (page_width - draw_width) / 2
        y = (page_height - draw_height) / 2 - 0.5 * cm

        c.drawImage(str(image_path), x, y, width=draw_width, height=draw_height, preserveAspectRatio=True)
        c.showPage()

    draw_title_page()

    title_map = {
        "sentiment_percentages_bar.png": "Sentiment Distribution Bar Chart",
        "sentiment_percentages_pie.png": "Sentiment Distribution Pie Chart",
        "upvotes_downvotes_by_sentiment.png": "Votes by Sentiment",
        "sentiment_combined.png": "Combined Sentiment Figure",
        "sentiment_trend_by_year.png": "Sentiment Trend by Year",
    }

    for plot_path in plot_paths:
        plot_name = Path(plot_path).name
        plot_title = title_map.get(plot_name, plot_name)
        draw_image_page(plot_path, plot_title)

    c.save()
    return output_pdf_path