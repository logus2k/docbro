# Section 3 -- Dataset and Problem Definition

This document explains the goals, concepts, and implementation details of Section 3 of the notebook, which covers loading the Jena Climate dataset, cleaning it, resampling it to hourly resolution, and formally defining the forecasting problem.

---

## Table of Contents

1. [Section Goal](#1-section-goal)
2. [The Jena Climate Dataset](#2-the-jena-climate-dataset)
3. [Loading Data from CSV](#3-loading-data-from-csv)
4. [Initial Dataset Inspection](#4-initial-dataset-inspection)
5. [Duplicated Rows](#5-duplicated-rows)
6. [Missing Values](#6-missing-values)
7. [Datetime Parsing](#7-datetime-parsing)
8. [Temporal Ordering](#8-temporal-ordering)
9. [Time Delta Distribution](#9-time-delta-distribution)
10. [Hourly Resampling](#10-hourly-resampling)
11. [Post-Resampling Quality Check](#11-post-resampling-quality-check)
12. [Handling NaN After Resampling](#12-handling-nan-after-resampling)
13. [Variable Selection](#13-variable-selection)
14. [The Forecasting Problem Definition](#14-the-forecasting-problem-definition)
15. [Why These Choices Matter Downstream](#15-why-these-choices-matter-downstream)

---

## 1. Section Goal

The purpose of this section is to transform the raw dataset into a clean, hourly, well-understood dataframe ready for feature engineering and modelling. By the end of Section 3, the notebook has:

- Loaded and verified the raw data
- Removed duplicates and handled missing values
- Resampled from 10-minute to 1-hour intervals
- Selected 6 meteorological variables from the original 15
- Formally defined the forecasting task

Every decision here directly affects every model trained later. A mistake at this stage (e.g., keeping duplicates, leaking test data, choosing the wrong resampling method) would propagate through the entire pipeline.

---

## 2. The Jena Climate Dataset

### What it is

A publicly available dataset of meteorological measurements collected by the Max Planck Institute for Biogeochemistry at a weather station in Jena, Germany.

### Key properties

| Property | Value |
|---|---|
| Time span | January 2009 -- December 2016 |
| Original sampling | Every ~10 minutes |
| Raw rows | ~420,224 observations |
| Original columns | 15 meteorological variables + 1 datetime |
| Source file | `jena_climate_2009_2016.csv` |

### Why this dataset?

It is a standard benchmark in time series deep learning. It has enough volume (8 years), enough complexity (multiple correlated weather variables), and a clear, physically meaningful prediction target (temperature). It is also well-studied, making it easier to compare results against published work.

---

## 3. Loading Data from CSV

### What happens

```python
DATASET_PATH = DATA_DIR / "jena_climate_2009_2016.csv"
df = pd.read_csv(DATASET_PATH)
df.head()
```

### Concepts

**CSV (Comma-Separated Values):** A plain-text file format where each row is a line and columns are separated by commas. It is the most common way to store and share tabular data.

**`pd.read_csv()`:** A Pandas function that reads a CSV file into a **DataFrame** -- a 2D table with labelled columns and indexed rows. This is the primary data structure used throughout the notebook.

**`df.head()`:** Shows the first 5 rows. This is a sanity check -- you confirm the file loaded correctly, columns have sensible names, and values look reasonable before doing anything else.

**`Path` (from `pathlib`):** A Python object representing a file system path. Using `Path.cwd().parent / "src" / "data"` constructs the path in a platform-independent way (works on Windows, Linux, Mac) instead of hardcoding slashes.

---

## 4. Initial Dataset Inspection

### What happens

```python
print("Shape:", df.shape)
print(df.columns.tolist())
df.info()
```

### Concepts

**`df.shape`:** Returns a tuple `(rows, columns)`. For the raw dataset this is approximately `(420224, 16)` -- 420,224 observations and 16 columns (15 variables + 1 datetime).

**`df.columns.tolist()`:** Lists all column names. This lets you see exactly what variables are available.

**`df.info()`:** Shows each column's data type, non-null count, and memory usage. This is important because:
- If a numeric column has type `object`, it means there are non-numeric values mixed in (a data quality problem)
- If non-null counts differ between columns, some columns have missing values
- Memory usage tells you if the dataset fits comfortably in RAM

### Why inspect first?

Before transforming data, you need to understand its current state. Inspecting shape, types, and nulls prevents surprises later. This is a fundamental step in any data science workflow, sometimes called **Exploratory Data Analysis (EDA)**.

---

## 5. Duplicated Rows

### What happens

```python
print("Duplicated rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()
```

### Concepts

**Duplicated row:** A row where every column value is identical to another row in the dataset. In a time series, this typically means the same observation was logged twice due to a sensor or logging artifact.

**`df.duplicated()`:** Returns a boolean Series -- `True` for each row that is a duplicate of a previous row. `.sum()` counts how many.

**`df.drop_duplicates()`:** Removes all duplicated rows, keeping the first occurrence.

**`.copy()`:** Creates an independent copy of the dataframe. Without this, Pandas may create a "view" that shares memory with the original -- modifying it later could produce unexpected warnings or side effects.

### Why remove duplicates?

Duplicates would cause two problems:
1. **Biased resampling:** When computing hourly means, a duplicated 10-minute observation would be counted twice, skewing the average
2. **Inflated dataset size:** The model would see the same information twice, which does not help learning but does increase computation time

---

## 6. Missing Values

### What happens

```python
print(df.isna().sum())
```

### Concepts

**Missing value (NaN):** "Not a Number" -- a placeholder indicating that no data was recorded for that cell. In a weather station, this could happen if a sensor malfunctions or communication drops temporarily.

**`df.isna()`:** Returns a boolean DataFrame -- `True` wherever a value is missing.

**`.sum()`:** Counts `True` values per column, giving you the number of missing values in each variable.

### Why check for missing values?

- Most machine learning models (including neural networks) cannot process NaN values -- they cause errors or undefined behavior
- The strategy for handling them (drop, interpolate, fill with a constant) affects data quality
- Knowing how many are missing, and in which columns, informs the right strategy

At this stage, the raw dataset has **no missing values** -- they only appear later after resampling.

---

## 7. Datetime Parsing

### What happens

```python
df["Date Time"] = pd.to_datetime(df["Date Time"], dayfirst=True)
```

### Concepts

**Datetime parsing:** Converting a text string like `"01.01.2009 00:10:00"` into a proper datetime object that Python understands as a point in time. Without parsing, the `Date Time` column is just text -- you cannot sort by time, compute time differences, or resample.

**`pd.to_datetime()`:** The Pandas function that performs this conversion. It reads the string and produces a `Timestamp` object.

**`dayfirst=True`:** Tells the parser that the date format is `DD.MM.YYYY` (day first, as used in Germany), not `MM/DD/YYYY` (month first, as used in the US). Without this flag, `01.02.2009` would be parsed as January 2nd instead of February 1st -- a subtle but critical bug.

### Why this matters

Everything downstream depends on correct temporal ordering:
- The train/val/test split is chronological
- Resampling groups observations by hour
- Feature engineering extracts hour-of-day and day-of-year
- If dates are wrong, all of this breaks silently

---

## 8. Temporal Ordering

### What happens

```python
df = df.sort_values("Date Time").reset_index(drop=True)
```

### Concepts

**`sort_values("Date Time")`:** Sorts all rows by the datetime column in ascending order (oldest first). This guarantees chronological ordering.

**`reset_index(drop=True)`:** After sorting, the original row indices are scrambled (e.g., row 5000 might now be in position 3). This resets the index to a clean 0, 1, 2, ... sequence. `drop=True` means "don't keep the old index as a column."

### Why sort?

Although the raw data is likely already in order, **you cannot assume this**. Data files can be written out of order due to logging buffering, time zone issues, or file merging. Sorting explicitly is a defensive practice -- it costs almost nothing and prevents a class of hard-to-detect bugs.

---

## 9. Time Delta Distribution

### What happens

```python
df["Date Time"].diff().value_counts().head(10)
```

### Concepts

**`diff()`:** Computes the difference between each consecutive pair of timestamps. For a perfectly regular 10-minute series, every difference should be `0 days 00:10:00`.

**`value_counts()`:** Counts how many times each unique difference appears and sorts by frequency.

### What to look for

| Observation | Meaning |
|---|---|
| Almost all deltas = 10 min | The series is regular, as expected |
| A few deltas > 10 min | There are gaps (missing observations) |
| Deltas < 10 min | Possible duplicates or sub-minute logging |
| Deltas = 0 | Exact duplicate timestamps (should have been removed already) |

This diagnostic step reveals whether the time series is clean enough for resampling. If there were large gaps (e.g., a whole month missing), you would need a different strategy.

---

## 10. Hourly Resampling

### What happens

```python
df_hourly = (
    df.set_index("Date Time")
      .resample("1h")
      .mean()
      .reset_index()
)
```

### Concepts

**Resampling:** Changing the frequency of a time series. Going from higher frequency (10 min) to lower frequency (1 hour) is called **downsampling**. The opposite (e.g., hourly to 5-minute via interpolation) is **upsampling**.

**`set_index("Date Time")`:** Makes the datetime column the row index. Pandas requires a datetime index for the `.resample()` method.

**`resample("1h")`:** Groups all observations falling within each 1-hour bin. For example, the 6 readings between 14:00 and 14:50 are grouped into the 14:00 bin.

**`.mean()`:** Within each hourly bin, computes the arithmetic average of all observations. This is the **aggregation function** -- it defines how multiple 10-minute readings are collapsed into one hourly value.

**`reset_index()`:** Moves the datetime back from the index to a regular column (easier to work with later).

### Why resample to hourly?

| Reason | Explanation |
|---|---|
| **Matches the forecasting task** | Predicting 24 hours ahead at 10-minute resolution would mean 144 output steps instead of 24, making the problem much harder |
| **Reduces computational cost** | 70,000 rows instead of 420,000 means faster training, less memory |
| **Smooths sub-hourly noise** | 10-minute fluctuations (e.g., a brief wind gust) are less relevant for day-ahead temperature forecasting |
| **Preserves meaningful patterns** | Diurnal (day/night) and seasonal temperature cycles operate on hourly to daily timescales |

### Why `.mean()` specifically?

| Alternative | Problem |
|---|---|
| `.first()` or `.last()` | Takes a single reading, discarding information from the other 5 readings in the hour |
| `.median()` | More robust to outliers but less representative of the full hour's conditions |
| `.mean()` | Preserves the central tendency of the entire hourly window and is the standard approach in climate data processing |

### Result

| | Rows | Resolution |
|---|---:|---|
| Before | ~420,224 | ~10 minutes |
| After | ~70,129 | 1 hour |

The sixfold reduction is expected: 6 ten-minute intervals per hour.

---

## 11. Post-Resampling Quality Check

### What happens

```python
print(df_hourly.isna().sum())
print(df_hourly["Date Time"].duplicated().sum())
print(df_hourly["Date Time"].diff().value_counts().head(10))
```

### What is checked

| Check | Expected result | Actual result |
|---|---|---|
| Missing values | Possibly a few (from incomplete hourly bins) | 88 NaN per meteorological column |
| Duplicated timestamps | None | 0 |
| Time delta distribution | Almost all = 1 hour | Confirmed |

### Why 88 NaN values appeared

When resampling with `.mean()`, an hourly bin that contains **no observations at all** produces NaN. This happens when there is a gap in the original 10-minute data -- if no readings were recorded between, say, 03:00 and 03:59, the 03:00 bin has nothing to average. With 8 years of data and only 88 affected bins, coverage is excellent (99.87%).

### Why re-check after resampling?

Resampling is a **transformation** -- it can introduce new issues that did not exist in the raw data:
- NaN values from empty bins (as seen here)
- Duplicated timestamps from timezone issues
- Irregular time deltas if the original data had large gaps

You should always validate data after any transformation, not just at the beginning.

---

## 12. Handling NaN After Resampling

### What happens

```python
df_hourly = df_hourly.dropna().reset_index(drop=True)
```

### Concepts

**`dropna()`:** Removes every row that contains at least one NaN value. This is the simplest missing-value strategy.

### Why drop instead of impute?

**Imputation** means filling missing values with estimated values (e.g., linear interpolation, forward fill, mean). The notebook chose dropping because:

| Factor | Reasoning |
|---|---|
| **Few affected rows** | Only 88 out of ~70,129 (0.13%) -- negligible data loss |
| **Cause is data gaps** | These bins had no readings at all, so interpolation would be pure guessing |
| **Risk of artifacts** | Interpolated values could introduce fake patterns that the model learns as real |
| **Simplicity** | Dropping is the safest and most transparent approach when the loss is small |

### Result

70,129 - 88 = **70,041 clean hourly observations**, with zero missing values.

---

## 13. Variable Selection

### What happens

```python
selected_columns = [
    "Date Time",
    "T (degC)",     # Air temperature (TARGET)
    "p (mbar)",     # Atmospheric pressure
    "rh (%)",       # Relative humidity
    "wv (m/s)",     # Wind speed
    "max. wv (m/s)",# Maximum wind speed
    "wd (deg)"      # Wind direction
]

df_model = df_hourly[selected_columns].copy()
```

### Why these 6 variables?

The selection follows the **project specification** (assignment requirements). But the choices are also physically motivated:

| Variable | Why it matters for temperature forecasting |
|---|---|
| **T (degC)** | The target itself -- past temperature is the strongest predictor of future temperature (autoregressive signal) |
| **p (mbar)** | Atmospheric pressure changes indicate weather system movements (fronts, storms) that drive temperature changes |
| **rh (%)** | Humidity affects how temperature evolves -- humid air changes temperature more slowly (higher heat capacity) |
| **wv (m/s)** | Wind transports air masses of different temperatures; strong wind can bring rapid changes |
| **max. wv (m/s)** | Captures gust intensity, which may signal turbulent mixing or incoming fronts |
| **wd (deg)** | Wind direction determines which air mass is arriving -- northerly wind brings cold air, southerly brings warm in the Northern Hemisphere |

### What was excluded?

The original dataset has 15 variables. The 9 excluded ones include variables like `Tdew (degC)` (dew point), `VPmax (mbar)` (saturation vapor pressure), `VPact (mbar)` (actual vapor pressure), `VPdef (mbar)` (vapor pressure deficit), `sh (g/kg)` (specific humidity), `H2OC (mmol/mol)` (water vapor concentration), `rho (g/m**3)` (air density), `aH (g/m**3)` (absolute humidity), and `Tpot (K)` (potential temperature).

Many of these are **highly correlated** with the retained variables (e.g., dew point and vapor pressure are derived from temperature and humidity). Keeping all of them would add redundancy without adding much new information, while increasing model complexity and training time.

---

## 14. The Forecasting Problem Definition

By the end of Section 3, the problem is formally defined:

### Input

A **window of past hourly observations** across 6 meteorological variables (later expanded to 16 via feature engineering). The window length (lookback) is 120 hours by default (5 days).

### Output

A **sequence of 24 future temperature values** -- one per hour for the next day.

### Problem type

**Multivariate-input, univariate-output, multi-step prediction**

```
                  6 variables (later 16)
                  |
Input tensor:     (120 time steps, 16 features)
                  |
                  v
    [  GRU Model  ]
                  |
                  v
Output vector:    (24 future temperatures)
```

### Cross-variable dependencies

The multivariate input allows the model to exploit physical relationships. For example:
- A **rapid pressure drop** often precedes a temperature change (approaching storm front)
- A **shift in wind direction** from south to north signals incoming colder air
- **High humidity** moderates temperature swings (the air holds more thermal energy)

A univariate model (temperature only) would miss these signals entirely.

---

## 15. Why These Choices Matter Downstream

Every decision in Section 3 has consequences for the rest of the pipeline:

| Decision | Downstream impact |
|---|---|
| **Hourly resolution** | Defines the unit of the lookback window (120 hours, not 120 ten-minute intervals) and the forecast horizon (24 hours = 24 steps, not 144 steps) |
| **Drop duplicates** | Ensures resampling produces correct hourly means |
| **Drop NaN (not impute)** | Avoids fake patterns in training data but introduces tiny gaps in the time series |
| **6 selected variables** | Determines the base input dimensionality (expanded to 16 after feature engineering in Section 5) |
| **Datetime parsing with `dayfirst=True`** | Ensures correct hour-of-day and day-of-year extraction in feature engineering |
| **Chronological ordering** | Guarantees that the train/val/test split in Section 6 respects temporal order (no future leakage) |

### Data flow summary

```
Raw CSV (420,224 rows x 16 cols, 10-min)
    |
    v  drop duplicates
    |
    v  parse datetime, sort chronologically
    |
    v  resample to hourly (.mean())
    |
    v  drop NaN rows
    |
    v  select 6 variables + datetime
    |
df_model (70,041 rows x 7 cols, 1-hour)
    |
    v  [continues to Section 4: Data Preparation]
```
