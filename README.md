# Data Quality Detective — An AI-Powered Data Reliability Platform

An end-to-end data reliability platform designed to detect, evaluate, investigate, and report data-quality issues and distribution shifts in analytical datasets.

## 1. Project Overview

Data-quality problems such as missing values, invalid records, unexpected outliers, and changes in data distributions can compromise analytical results and downstream business decisions.

**Data Quality Detective** addresses these challenges through an integrated Python and PostgreSQL solution that combines rule-based validation, statistical profiling, anomaly detection, data drift monitoring, and root-cause analysis.

The project uses NYC Yellow Taxi trip data and controlled synthetic data-quality incidents to develop and evaluate its detection capabilities.

## 2. Key Features

### Data Ingestion
- Load and process NYC Taxi trip datasets.
- Support batch-oriented data ingestion.
- Track ingestion metadata and batch status in PostgreSQL.
- Simulate controlled data-quality incidents for evaluation.

### Data Quality Validation
- Detect missing and invalid values.
- Identify negative fares and trip distances.
- Validate passenger counts and trip durations.
- Check fare-component relationships.
- Persist validation results for analysis.

### Statistical Profiling
- Generate descriptive statistics for dataset features.
- Analyze distributions and statistical characteristics.
- Support comparisons across data batches.

### Outlier Detection

Implements and evaluates four anomaly-detection approaches:

1. Interquartile Range (IQR)
2. Z-Score
3. Modified Z-Score
4. Isolation Forest

The evaluation framework measures detection performance against available ground-truth incident labels.

### Data Drift Detection

Monitors changes between reference and current datasets using:

| Metric | Purpose |
|---|---|
| Population Stability Index (PSI) | Measures distribution shifts |
| Kolmogorov-Smirnov (KS) test | Detects differences between numerical distributions |
| Wasserstein distance | Measures distribution distance |
| Jensen-Shannon divergence | Measures categorical distribution differences |

Drift results are classified by severity and persisted in PostgreSQL.

### Root-Cause Analysis

Investigates detected anomalies and quality issues through:
- Statistical evidence.
- Business-level patterns.
- Vendor-level comparisons.
- Pickup-location analysis.
- Supporting evidence stored for investigation.

Generates consolidated findings and reports in Markdown and PDF formats.

### Interactive Dashboard

Built with Streamlit and Plotly.

Provides visual analysis of:
- Data drift results.
- Feature distributions.
- Batch-level comparisons.
- Root-cause investigation reports.

### Database Integration

Uses PostgreSQL for structured storage of:
- Ingestion batch metadata.
- Taxi trip records.
- Data-quality validation results.
- Drift detection results.

SQLAlchemy provides database connectivity.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Statistical analysis | SciPy |
| Database | PostgreSQL |
| Database connectivity | SQLAlchemy, psycopg2 |
| Dashboard | Streamlit, Plotly |
| Reporting | ReportLab |
| Testing | Pytest |
| Continuous integration | GitHub Actions |
| Data formats | Parquet, CSV |

## 4. Project Structure

```text
Data Quality Detective/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── dashboard/
│   └── app.py
│
├── scripts/
│   └── inspect_taxi_schema.py
│
├── src/
│   ├── database/
│   │   └── schema.sql
│   │
│   ├── ingestion/
│   │   ├── taxi_loader.py
│   │   ├── monthly_loader.py
│   │   └── incident_loader.py
│   │
│   ├── quality/
│   │   ├── quality_rules.py
│   │   ├── validation_engine.py
│   │   ├── incident_simulator.py
│   │   └── incident_validator.py
│   │
│   ├── profiling/
│   │   └── statistical_profiler.py
│   │
│   ├── outlier_detection/
│   │   ├── iqr_detector.py
│   │   ├── zscore_detector.py
│   │   ├── modified_zscore_detector.py
│   │   └── isolation_forest_detector.py
│   │
│   ├── drift/
│   │   └── drift_detector.py
│   │
│   ├── evaluation/
│   │   ├── anomaly_evaluator.py
│   │   └── incident_evaluator.py
│   │
│   └── root_cause/
│       ├── rca_engine.py
│       ├── report_generator.py
│       └── pdf_report_generator.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── requirements.txt
├── pytest.ini
└── README.md
```

## 5. Installation and Setup

### Prerequisites

- Python 3.14
- PostgreSQL 18
- Git

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Data Quality Detective"
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure PostgreSQL

Create a PostgreSQL database for the application.

Configure the following application environment variables in a local `.env` file:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
```

Apply the base database schema:

```bash
psql -U your_database_user -d your_database_name -f src/database/schema.sql
```

The integration tests require a dedicated PostgreSQL test database named
`data_quality_detective_test`. Do not use your production or application
database for integration tests.

For local testing, create a `.env.test` file in the project root:

```env
TEST_DB_HOST=localhost
TEST_DB_PORT=5432
TEST_DB_NAME=data_quality_detective_test
TEST_DB_USER=your_postgres_user
TEST_DB_PASSWORD=your_postgres_password
```

### Configure the datasets

Obtain the required NYC Yellow Taxi Parquet datasets separately and place them in the expected local data directories.

Dataset files and generated reports are excluded from Git to keep the repository lightweight and avoid committing raw or generated data.

## 6. Running the Dashboard

After configuring the application database and preparing the required database tables and data:

```bash
streamlit run dashboard/app.py
```

The dashboard opens in your browser.

## 7. Running Automated Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Run tests with coverage:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

### Current Test Results

| Metric | Result |
|---|---:|
| Total automated tests | 192 |
| Passed | 192 |
| Failed | 0 |
| Last measured overall code coverage | 98% |

PostgreSQL integration tests use a dedicated test database to validate schema configuration and drift-result persistence.

## 8. Continuous Integration

GitHub Actions automatically executes the test suite on pushes and pull requests targeting the `main` or `master` branches.

The workflow:
1. Checks out the repository.
2. Sets up Python 3.14.
3. Starts a PostgreSQL 18 service.
4. Installs project dependencies.
5. Executes the automated test suite.

Workflow configuration:

`.github/workflows/tests.yml`

## 9. Evaluation and Engineering Considerations

The project includes controlled synthetic incidents to evaluate anomaly and validation approaches against known injected issues.

Detection performance is evaluated using classification metrics such as precision, recall, and F1-score.

Synthetic evaluation results should not be interpreted as proof of equivalent performance on all real-world datasets. Detection thresholds and root-cause hypotheses require validation against the relevant business context.

## 10. Future Improvements

- Introduce database migrations using Alembic.
- Add automated data-quality monitoring schedules.
- Improve alerting and incident notification.
- Expand drift monitoring to additional datasets.
- Add deployment and containerization support.
- Introduce production-grade observability and monitoring.

---

**Project:** Data Quality Detective — An AI-Powered Data Reliability Platform

**Focus Areas:** Data Engineering | Data Quality | Statistical Analysis | Anomaly Detection | Data Drift | Root-Cause Analysis | PostgreSQL | Machine Learning