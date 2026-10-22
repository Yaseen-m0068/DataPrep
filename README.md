# DataPrep

### Explainable Automated Data Preprocessing Assistant

DataPrep is an interactive data preprocessing tool built with Python and Streamlit. It helps users explore, clean, and transform structured datasets before using them for data analysis or machine learning.

Instead of silently modifying data, DataPrep allows users to review potential data quality issues, preview transformations, choose preprocessing strategies, and apply changes when they are ready.

The goal is to make data preprocessing **more accessible, customizable, transparent, and reproducible**.

## Features

### Dataset Ingestion and Profiling
- Upload supported CSV, Excel, and Parquet datasets.
- View dataset dimensions and column information.
- Inspect data types, missing values, duplicate rows, and other data quality indicators.
- Review detected column types and preprocessing recommendations.

### Data Cleaning and Transformation
- **Missing values:** Preview and apply missing-value handling strategies.
- **Duplicate rows:** Identify and remove duplicate records.
- **Categorical standardization:** Standardize inconsistent categorical values.
- **Categorical encoding:** Apply one-hot or ordinal encoding.
- **Numerical outliers:** Detect and apply supported outlier-handling strategies.
- **Numerical scaling:** Scale numerical features using supported methods.

### Interactive Workflow
- Preview changes before applying transformations.
- Compare the original and cleaned dataset metrics.
- Maintain transformation history.
- Undo and redo transformations.
- Reset the dataset to its original state.
- Export processed data to CSV or Excel.

> DataPrep is under active development. Available functionality may change as new preprocessing workflows are added.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing utilities |
| SciPy | Scientific computing |
| Streamlit | Interactive web application |
| Plotly | Data visualization |
| PyArrow | Columnar data processing and Parquet support |
| OpenPyXL | Excel file handling |
| Pytest | Automated testing |

## Project Architecture

The project follows a modular structure to separate dataset ingestion, profiling, detection, transformation, and pipeline management.

```text
DataPrep/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── ingestion/
│   ├── profiling/
│   ├── detection/
│   ├── recommendations/
│   ├── transformations/
│   ├── validation/
│   └── pipeline/
├── tests/
└── data/
    └── sample/
```

*Note: The tree above is a high-level overview. Some folders may contain additional modules, and the exact structure can evolve during development.*

## Getting Started

Follow the instructions for your operating system to download, install, and run DataPrep locally.

### Prerequisites

Before starting, install:

- **Python 3.11** — the version used for development and testing.
- **Git** — to clone the repository.
- A terminal such as Windows PowerShell, macOS Terminal, or a Linux shell.
- A modern web browser.

Check whether Python and Git are installed:

```bash
python --version
git --version
```

On systems where Python 3 is invoked as `python3`, use:

```bash
python3 --version
```

## 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/Yaseen-m0068/DataPrep.git
cd DataPrep
```

## 2. Create a Virtual Environment

A virtual environment keeps DataPrep's Python dependencies separate from other projects.

### Windows — PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not available, install Python 3.11 and ensure it is accessible from your terminal.

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Debian or Ubuntu, if virtual environment creation is unavailable, you may need to install the Python 3.11 venv package using your distribution's package manager.

After activation, verify the Python version:

```bash
python --version
```

The command should report Python 3.11.x.

## 3. Install Dependencies

With the virtual environment activated, upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Wait for installation to complete before proceeding.

If installation fails, check that you are using the correct Python version and that the virtual environment is activated.

## 4. Run the Application

From the project's root directory, run:

```bash
python -m streamlit run app.py
```

Streamlit will start a local web server and display a local URL in your terminal, usually:

```text
http://localhost:8501
```

Open the displayed URL in your browser to use DataPrep.

To stop the application, press `Ctrl + C` in the terminal.

## 5. Run the Tests

DataPrep uses Pytest to test its preprocessing modules and related functionality.

To run the complete test suite, execute this command from the project root with the virtual environment activated:

```bash
python -m pytest -q
```

To see detailed test results:

```bash
python -m pytest -v
```

To run tests for a particular module, for example categorical encoding:

```bash
python -m pytest tests/test_encoding.py -v
```

A successful test run should report that all executed tests passed. The number of tests may increase as the project develops.

## How to Use DataPrep

1. **Launch the application** using the command above.
2. **Upload a dataset** in one of the supported file formats.
3. **Explore the profile** to understand the dataset's structure and data quality.
4. **Review preprocessing options** and select the columns and strategies you want to use.
5. **Preview the transformation** and inspect the proposed changes.
6. **Apply the transformation** when the result is suitable.
7. **Review the cleaned data** and use the available history controls if you need to undo or redo a change.
8. **Export the processed dataset** when you are satisfied with the result.

Exact options depend on the selected transformation and the dataset.

## Development Workflow

If you want to contribute or experiment with new features:

1. Create a feature branch.
2. Implement the change in the appropriate module.
3. Add or update tests.
4. Run the complete test suite.
5. Review the changes before committing.

Example:

```bash
git checkout -b feature/your-feature-name
python -m pytest -q
git status
git diff --check
```

Once the changes have been reviewed and tested, commit and push your branch.

## Roadmap

Potential future enhancements include:

- Expanded date and text preprocessing.
- More intelligent data type detection.
- Additional missing-value and outlier-handling strategies.
- Improved preprocessing recommendations.
- Enhanced before-and-after data validation.
- Reproducible preprocessing pipeline export.
- Preprocessing code generation.
- Additional supported data formats and transformations.

These items are planned enhancements and should not be considered implemented unless they are available in the current version.

## Project Goal

The goal of DataPrep is to build a flexible and explainable preprocessing assistant that supports different structured datasets while keeping users in control of their data preparation decisions.

By combining dataset profiling, configurable transformations, previews, and transformation history, DataPrep aims to make data preparation easier to understand and more reliable for data analysis and machine learning workflows.

## Author

**Mohamed Yaseen**

GitHub: [Yaseen-m0068](https://github.com/Yaseen-m0068)