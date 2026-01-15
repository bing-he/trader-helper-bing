Criterion ETL Pipeline for TraderHelper
Mission

Deliver clean, analysis-ready U.S. natural gas fundamentals for automated analysis and trading models.

Maintainer

Patrick

Status

Active / Production

Primary Entry Point

CriterionOrchestrator.py

1. Project Abstract
Criterion is the data engineering nerve center of the TraderHelper project. It functions as a robust, automated ETL (Extract, Transform, Load) pipeline that systematically retrieves raw energy data from a central PostgreSQL database, applies a series of transformations and calculations, and loads the resulting analysis-ready datasets into a centralized CSV data vault.

The pipeline is designed for high reliability, maintainability, and extensibility, enabling both human analysts and AI agents to monitor market signals, train predictive models, and automate analytical workflows with confidence in the data's integrity.

2. Core Design Principles
This codebase was refactored to adhere strictly to modern software engineering best practices to ensure stability and clarity. An AI agent maintaining this system can assume the following principles are fundamental to its design:

Modularity (SOLID Principles): Each script and function adheres to the Single Responsibility Principle. For example, one function fetches data, another transforms it, and a third saves it. This isolates logic and simplifies debugging.

Don't Repeat Yourself (DRY): Shared logic (e.g., database connections, logging setup, saving data) is encapsulated in reusable helper functions. The CriterionOrchestrator.py script centralizes the execution flow.

Simplicity (KISS): Complex, monolithic scripts have been broken down into clear, sequential pipelines. The main() function in each script acts as a high-level orchestrator, making the operational flow easy to understand.

Code Standards (PEP 8, 20, 257): All code is formatted to PEP 8 standards for readability. It strives to be "Pythonic" (PEP 20), favoring clarity over complexity. All functions are documented with PEP 257 docstrings and include type hints for static analysis.

Portability: No absolute file paths are used. All paths are relative to the script's location using pathlib, ensuring the project runs on any machine without modification.

Robust Logging: All scripts use the logging module instead of print(). The orchestrator produces a timestamped log of the entire run, including the captured output of each sub-script, creating a clear audit trail.

3. System Architecture & Data Flow
The system follows a classic ETL pattern orchestrated by a main controller script.

flowchart TD
    subgraph A[Source Layer]
        A1[PostgreSQL Database]
        A2[Mapping & Config CSVs]
    end

    subgraph B[Processing Layer: Python Scripts]
        B1[CriterionOrchestrator.py]
        B2[Update Scripts]
    end

    subgraph C[Storage Layer: Data Vault]
        C1["INFO/ Directory"]
    end

    subgraph D[Consumption Layer]
        D1[Jupyter Notebooks]
        D2[ML Models]
        D3[Dashboards & Alerts]
    end

    A1 --> B2
    A2 --> B2
    B1 -- runs in sequence --> B2
    B2 -- write results --> C1
    C1 -- provide data --> D1
    C1 -- provide data --> D2
    C1 -- provide data --> D3

Source Layer: The process begins with the PostgreSQL Database (for live data) and a set of Mapping CSVs that define which data series to process.

Processing Layer: The CriterionOrchestrator.py script executes each Update*.py script in a predefined order. Each script is responsible for a specific data domain (e.g., Storage, LNG).

Storage Layer: All processed data is saved as clean, standardized CSV files in the INFO/ directory, which serves as the "data vault" or source of truth for all downstream applications.

Consumption Layer: The CSVs in the INFO/ vault are consumed by Jupyter notebooks for research, machine learning pipelines for model training, and dashboards for visualization.

4. Component Breakdown: Scripts
Each script is a self-contained ETL pipeline for a specific data domain.

Script

Responsibility

Key Inputs

Main Outputs (→ INFO/)

CriterionOrchestrator.py

Sequentially executes all other update scripts; halts on any failure.

SCRIPTS_TO_RUN list

Logs to stdout

UpdateAndForecastFundy.py

Primary Engine: Fetches all core series, calculates regional/CONUS balances, and generates forecasts.

database_tables_list.csv, CriterionExtra_tables_list.csv

Fundy.csv, FundyForecast.csv, CriterionExtra.csv, CriterionExtraForecast.csv

UpdateCriterionStorage.py

Calculates the net daily storage change (injection/withdrawal) for each facility.

Database nomination_points

CriterionStorageChange.csv

UpdateCriterionLNG.py

Fetches historical LNG feed gas data and generates a 60-day forward forecast.

Hard-coded ticker lists

CriterionLNGHist.csv, CriterionLNGForecast.csv

UpdateCriterionHenryFlows.py

Fetches scheduled vs. available capacity for all pipelines connected to Henry Hub.

locs_list.csv (to identify Henry tickers)

CriterionHenryFlows.csv

UpdateCriterionNuclear.py

Fetches historical nuclear generation data and associated forward forecasts.

NuclearPairs.csv

CriterionNuclearHist.csv, CriterionNuclearForecast.csv

UpdateCriterionLocs.py

Syncs Metadata: Intelligently updates the local location mapping file from the database, preserving manual edits.

Database metadata, locs_list.csv

locs_list.csv (updated in-place)

5. Detailed Script Logic
This section provides a deeper look into the internal workings of each key script.

UpdateAndForecastFundy.py
This is the most complex script and acts as the primary data processing engine.

Dual-Source Processing: It systematically processes two mapping files: database_tables_list.csv (core) and CriterionExtra_tables_list.csv (extended).

Dual-Mode Operation: For each source, it runs in two modes:

ACTUALS: Performs an incremental update on the historical data file (e.g., Fundy.csv). It fetches the last 60 days of data and merges it, preserving older records.

FORECAST: Performs a full overwrite of the forecast file (e.g., FundyForecast.csv), replacing it entirely with the latest forecast data.

Derived Series Calculation: After fetching all base series, it pivots the data into a wide-format DataFrame (dates as index, items as columns) to perform vectorized calculations. This is where it computes regional and CONUS-level balances (e.g., Northeast - Balance = Prod - (Ind + ResCom + Power)).

Finalization: The data is unpivoted back into a long format, region information is applied, and the final, clean CSVs are saved.

UpdateCriterionLocs.py
This script is critical for maintaining the master location mapping file (locs_list.csv). Its logic is designed to be non-destructive to manual work.

Vectorized Reconciliation: It avoids slow row-by-row loops by performing a single pandas.merge operation between the local CSV and the live database table.

Three-Way Merge Logic: The result of the merge is split into three distinct categories:

New Records: Locations present in the database but not the local file. These are added with a blank market_component.

Orphaned Records: Locations in the local file but no longer in the database. Their loc_name is appended with " - ORPHANED", but the row is preserved.

Updated Records: Locations present in both. The script updates all database-sourced columns (loc_name, ticker, etc.) but explicitly preserves the value from the market_component column of the local file.

Schema Safety: It automatically adds any new columns from the database to the local CSV to prevent schema drift errors.

UpdateCriterionStorage.py
This script calculates a single, crucial metric: the daily net change in natural gas storage.

Incremental Fetch: It checks for an existing CriterionStorageChange.csv. If found, it only fetches the last 60 days of data to perform a fast, incremental update.

Core Calculation: The key logic is in its SQL query, which groups flows by storage_name and eff_gas_day. The final value is derived from SUM(scheduled_quantity * rec_del_sign) * -1 to correctly represent injections as positive and withdrawals as negative.

6. Component Breakdown: Data Artifacts
These CSV files control and are produced by the system.

File

Type

Purpose

database_tables_list.csv

Mapping

Defines the core fundamental series to be processed by UpdateAndForecastFundy.py.

CriterionExtra_tables_list.csv

Mapping

Defines a secondary set of series for UpdateAndForecastFundy.py.

NuclearPairs.csv

Mapping

Maps nuclear plant names to their respective database tickers.

locs_list.csv

Mapping & Output

Master location file. Contains metadata for all pipeline locations. It is both an input for other scripts and the output of UpdateCriterionLocs.py. The market_component column is manually maintained.

Fundy.csv / CriterionExtra.csv

Output

The primary historical datasets containing all fetched and calculated series.

FundyForecast.csv / etc.

Output

Forecast datasets generated by the corresponding scripts.

CriterionStorageChange.csv

Output

Clean, daily time-series of net storage changes by facility.

7. Setup and Execution
Prerequisites
Python 3.10+

Access to the private Git repository

Database credentials (if running in live mode)

1. Environment Setup
Execute these commands from the TraderHelper/Criterion directory.

# Set up and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all required packages from the repository root
pip install -r ../requirements.txt

2. Configuration (.env file)
Create a file named .env inside the Criterion/ directory. This file stores the database credentials.

# .env
DB_USER=your_username
DB_PASSWORD=your_password

If these values are left blank or the file does not exist, scripts that can fall back to local CSVs will attempt to do so.

3. Execution
Run the entire data pipeline with a single command:

python CriterionOrchestrator.py

The orchestrator will execute each script in sequence. Fresh CSV files will be generated in the INFO/ directory.

8. Operational Procedures for an AI Agent
As an AI agent maintaining this system, your primary responsibilities are automation and validation.

Scheduled Execution: Run CriterionOrchestrator.py on a schedule (e.g., daily at 05:00 ET).

Output Validation: After each run, verify the contents of the INFO/ directory.

File Existence: Confirm that all expected output CSVs have been created or updated.

File Freshness: Check that the "Date Modified" timestamp of each file corresponds to the latest run.

Content Sanity Check: For critical files like Fundy.csv, ensure the file is not empty and contains data for the most recent dates.

Failure Analysis: If the orchestrator exits with a non-zero status code (indicated by a CRITICAL log message), parse the log output. The traceback will identify the failed script and the specific error, which should be reported for human review.

Anomaly Detection (Future Capability): Implement logic to scan the output files for anomalies after a successful run. Examples:

CriterionStorageChange.csv: Flag any daily change that is more than 3 standard deviations from its 30-day rolling average.

Fundy.csv: Ensure values for demand components (ResCom, Power) are non-negative.