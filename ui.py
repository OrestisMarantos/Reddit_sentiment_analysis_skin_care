import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QMessageBox
)

from pipeline import run_category_pipeline


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reddit Beauty Sentiment Pipeline")
        self.setGeometry(200, 200, 800, 500)

        self.category_folder = None
        self.category_name = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        self.folder_label = QLabel("No category folder selected")
        self.category_label = QLabel("Analyzed word: -")

        self.select_folder_btn = QPushButton("Select Category Folder")
        self.select_folder_btn.clicked.connect(self.select_category_folder)

        self.go_btn = QPushButton("Go")
        self.go_btn.clicked.connect(self.run_pipeline)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(self.folder_label)
        layout.addWidget(self.category_label)
        layout.addWidget(self.select_folder_btn)
        layout.addWidget(self.go_btn)
        layout.addWidget(self.log_box)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def log(self, message):
        self.log_box.append(message)

    def select_category_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Category Folder")
        if folder:
            self.category_folder = folder
            self.category_name = os.path.basename(folder)

            self.folder_label.setText(f"Folder: {folder}")
            self.category_label.setText(f"Analyzed word: {self.category_name}")

            self.log(f"Selected folder: {folder}")
            self.log(f"Detected analyzed word: {self.category_name}")

    def run_pipeline(self):
        if not self.category_folder:
            QMessageBox.warning(self, "Missing folder", "Please select a category folder.")
            return

        try:
            self.go_btn.setEnabled(False)
            self.log("Starting pipeline...")

            results = run_category_pipeline(self.category_folder, logger=self.log)

            self.log("Pipeline finished successfully.")
            self.log(f"Combined CSV: {results['combined_csv']}")
            self.log(f"Plots folder: {results['output_dir'] / 'plots'}")
            self.log(f"PDF report: {results['report_pdf']}")

            QMessageBox.information(
                self,
                "Success",
                f"Pipeline finished successfully for '{results['category_name']}'."
            )

        except Exception as e:
            self.log(f"Error: {str(e)}")
            QMessageBox.critical(self, "Pipeline error", str(e))

        finally:
            self.go_btn.setEnabled(True)