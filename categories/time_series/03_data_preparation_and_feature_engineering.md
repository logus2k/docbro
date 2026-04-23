# Sections 4 & 5 -- Data Preparation and Feature Engineering

This document explains the goals, concepts, and implementation details of Sections 4 (Data Initial Preparation) and 5 (Feature Engineering) of the notebook. Section 4 defines the target variable and establishes the base modelling dataframe. Section 5 expands the 6 raw meteorological variables into 16 model-ready features using physically motivated transformations.

---

## Table of Contents

1. [Section Goals](#1-section-goals)
2. [Target Variable and Exogenous Inputs](#2-target-variable-and-exogenous-inputs)
3. [Why Date Time Is Not a Direct Input](#3-why-date-time-is-not-a-direct-input)
4. [The Feature Engineering Problem](#4-the-feature-engineering-problem)
5. [Cyclical Encoding -- The Core Idea](#5-cyclical-encoding----the-core-idea)
6. [Cyclical Time Features (4 features)](#6-cyclical-time-features-4-features)
7. [Wind-Derived Features (6 features)](#7-wind-derived-features-6-features)
8. [The Final 16-Feature Vector](#8-the-final-16-feature-vector)
9. [Implementation Details](#9-implementation-details)
10. [Why These Transformations Matter for the GRU](#10-why-these-transformations-matter-for-the-gru)

---

## 1. Section Goals

| Section | Input | Output | Purpose |
|---|---|---|---|
| **4** | `df_model` (70,041 rows x 7 cols) | Defined `TARGET_COL`, `TIME_COL`, `feature_cols` | Establish what we're predicting and what we're using as input |
| **5** | `df_model` with 6 raw variables | `df_feat` with 16 engineered features | Transform raw measurements into representations the neural network can learn from effectively |

The transition from 6 to 16 features is not about adding more information -- all the information is already in the original 6 variables. It is about **re-representing** that information in forms that are easier for a neural network to exploit.

---

## 2. Target Variable and Exogenous Inputs

### Target variable

```python
TARGET_COL = "T (degC)"
```

**Air temperature** is the only quantity the model must forecast. The output is a vector of 24 future temperature values.

### Exogenous inputs (covariates)

The other 5 meteorological variables are **exogenous inputs** -- they help predict the target but are not themselves predicted:

| Variable | Role |
|---|---|
| `p (mbar)` | Atmospheric pressure -- signals weather system changes |
| `rh (%)` | Relative humidity -- affects how temperature evolves |
| `wv (m/s)` | Wind speed -- transports air masses |
| `max. wv (m/s)` | Maximum wind speed -- captures gusts and turbulence |
| `wd (deg)` | Wind direction -- determines which air mass is arriving |

### Concept: exogenous vs. endogenous

- **Endogenous** variable: the thing you're modelling (temperature). Its future values are unknown and must be predicted.
- **Exogenous** variables: external factors that influence the endogenous variable. In this setup, their **past** values are available as input, but the model does not need to predict their future.

The model uses past values of all 6 variables (including past temperature) to predict only future temperature.

---

## 3. Why Date Time Is Not a Direct Input

```python
TIME_COL = "Date Time"
# Excluded from feature_cols -- NOT fed to the model
```

A neural network operates on **numeric tensors**. A datetime object like `2015-07-03 14:00:00` is not a number -- it is a structured object. You cannot feed it directly into a GRU.

But the datetime contains valuable information:
- **Hour of day** tells the model where we are in the diurnal cycle (noon is warm, 3am is cold)
- **Day of year** tells the model where we are in the seasonal cycle (July is warm, January is cold)

The solution: **extract** the temporal information as numbers, then **encode** those numbers in a form that respects their cyclic nature (Section 5.1). The raw `Date Time` column is used only for this extraction and then dropped from the model input.

---

## 4. The Feature Engineering Problem

Feature engineering is the process of **transforming raw data into representations** that make it easier for a model to learn useful patterns. The question is: why can't we just feed the 6 raw variables directly?

We could -- the model would still work. But certain raw representations create unnecessary difficulties:

### Problem 1: Cyclic discontinuities

Hour-of-day as a raw integer (0, 1, 2, ... 23) creates a **discontinuity**: hour 23 and hour 0 are adjacent in time but maximally distant numerically (23 vs 0). The network would need to learn that 23 and 0 are actually neighbors, which wastes capacity.

The same problem applies to:
- Day-of-year: day 365 and day 1 are adjacent
- Wind direction: 359 degrees and 1 degree are almost identical

### Problem 2: Polar vs. Cartesian representation

Wind is naturally described in **polar coordinates** -- a speed (magnitude) and a direction (angle). But neural networks work much better with **Cartesian coordinates** (x, y components) because:
- Cartesian addition is linear: two winds from different directions sum naturally
- Polar angles wrap around (360 = 0), which is non-linear and hard to learn

### Solution approach

Transform cyclic quantities with **sine-cosine encoding** and decompose polar quantities into **Cartesian components**. This is what Section 5 implements.

---

## 5. Cyclical Encoding -- The Core Idea

This is the most important concept in Section 5, used for both time features and wind direction.

### The problem with raw values

Imagine encoding hour-of-day as 0 to 23:

```
Hour:     0   6   12   18   23   0    <-- 23 and 0 are adjacent in reality
Value:    0   6   12   18   23   0    <-- but 23 units apart numerically
```

A neural network measuring distance between inputs would think hour 23 and hour 0 are very different, when they are actually 1 hour apart.

### The sine-cosine solution

Map each value to a point on the **unit circle** using sine and cosine:

```python
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
```

Now each hour is represented by two coordinates (sin, cos) that trace a circle:

```
              12:00
            (0, -1)
               |
  06:00  (-1, 0) --- (1, 0)  18:00
               |
            (0, 1)
              00:00
```

Key properties:
- **Hour 23 and hour 0 are now close** -- they are neighboring points on the circle
- **Hour 0 and hour 12 are maximally distant** -- opposite sides of the circle, reflecting the real-world diurnal difference
- **Smooth transitions everywhere** -- no discontinuities

### Why two features (sin AND cos)?

Using only sine would make **different hours look identical**: sin(6h) = sin(18h) = 0. The pair (sin, cos) gives a **unique** 2D position for every hour:

| Hour | sin | cos | Unique? |
|---:|---:|---:|---|
| 0 | 0.00 | 1.00 | Yes |
| 6 | -1.00 | 0.00 | Yes |
| 12 | 0.00 | -1.00 | Yes |
| 18 | 1.00 | 0.00 | Yes |

Sine alone cannot distinguish 6 from 18 (both give sin ~ 0 or ~+/-1 depending on convention). The pair resolves the ambiguity.

---

## 6. Cyclical Time Features (4 features)

### Implementation

```python
def add_time_features(df, time_col="Date Time"):
    df["hour"]      = df[time_col].dt.hour        # 0-23
    df["dayofyear"] = df[time_col].dt.dayofyear   # 1-366

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"]  = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"]  = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    return df
```

### Feature-by-feature explanation

| Feature | Formula | Period | What it captures |
|---|---|---|---|
| `hour_sin` | sin(2*pi * hour / 24) | 24 hours | Diurnal cycle -- sine component |
| `hour_cos` | cos(2*pi * hour / 24) | 24 hours | Diurnal cycle -- cosine component |
| `doy_sin` | sin(2*pi * dayofyear / 365.25) | ~365 days | Seasonal cycle -- sine component |
| `doy_cos` | cos(2*pi * dayofyear / 365.25) | ~365 days | Seasonal cycle -- cosine component |

### Why 365.25?

A year is not exactly 365 days -- leap years add a day every 4 years. Using 365.25 as the period keeps the encoding aligned across years. Without this, the seasonal encoding would drift slightly each year.

### The intermediate columns `hour` and `dayofyear`

These are extracted as integers first (0-23 and 1-366) to compute the sine/cosine. They are **not included** in the final 16-feature model input -- only the sine/cosine pairs are kept. The raw integers served as intermediate calculation steps.

### What the model learns from these features

- **Diurnal pair (hour_sin, hour_cos):** The model can learn that temperature peaks in the afternoon and drops at night, and that the rate of change differs between sunrise (rapid warming) and sunset (gradual cooling)
- **Seasonal pair (doy_sin, doy_cos):** The model can learn that summer temperatures are higher, winter temperatures lower, and that the transition rates differ between spring (fast warming) and autumn (slower cooling)

---

## 7. Wind-Derived Features (6 features)

### Implementation

```python
def add_wind_features(df):
    wd_rad = np.deg2rad(df["wd (deg)"])        # degrees -> radians

    df["wd_sin"] = np.sin(wd_rad)              # cyclic direction (sin)
    df["wd_cos"] = np.cos(wd_rad)              # cyclic direction (cos)

    df["wx"] = df["wv (m/s)"] * np.cos(wd_rad) # Cartesian x-component
    df["wy"] = df["wv (m/s)"] * np.sin(wd_rad) # Cartesian y-component

    df["wind_gap"]   = df["max. wv (m/s)"] - df["wv (m/s)"]
    df["gust_ratio"] = df["max. wv (m/s)"] / (df["wv (m/s)"] + 1e-6)

    return df
```

### Feature-by-feature explanation

#### Cyclic direction encoding: `wd_sin`, `wd_cos`

Same principle as the time features. Wind direction in degrees (0-360) is cyclic: 359 degrees and 1 degree are almost identical. Sine-cosine encoding removes the discontinuity.

```
            N (0/360 deg)
            (sin=0, cos=1)
               |
W (270)  ------|------  E (90)
(sin=-1,cos=0) | (sin=1,cos=0)
               |
            S (180 deg)
            (sin=0, cos=-1)
```

#### Cartesian wind vector: `wx`, `wy`

The raw data represents wind as **polar coordinates**: speed + direction. This is converted to **Cartesian components**:

```python
wx = speed * cos(direction)   # east-west component
wy = speed * sin(direction)   # north-south component
```

**Why Cartesian?** Consider two scenarios:
- Wind from the east at 5 m/s: direction = 90 deg, speed = 5
- Wind from the west at 5 m/s: direction = 270 deg, speed = 5

In polar form, these have the same speed (5) and very different angles. In Cartesian form:
- East wind: wx = 0, wy = 5 (positive y)
- West wind: wx = 0, wy = -5 (negative y)

The Cartesian form makes it **linearly obvious** that these are opposite winds. A neural network can learn this with a simple weight, while learning it from polar coordinates requires discovering a non-linear relationship.

#### Gust metrics: `wind_gap`, `gust_ratio`

These capture the relationship between sustained wind and gusts:

| Feature | Formula | What it measures |
|---|---|---|
| `wind_gap` | max_wv - wv | Absolute gust intensity in m/s. A gap of 10 m/s means gusts are 10 m/s stronger than sustained wind. |
| `gust_ratio` | max_wv / (wv + 1e-6) | Relative gust strength. A ratio of 3.0 means gusts are 3x the sustained wind. |

**Why `1e-6` (epsilon)?** When sustained wind speed is exactly 0, dividing by it produces infinity. Adding a tiny epsilon (0.000001) to the denominator prevents division by zero while having negligible effect on all other values.

**Physical relevance:** Strong gusts relative to sustained wind can indicate atmospheric instability (turbulence, convective activity), which often precedes temperature changes. These features give the model a direct signal for this.

### Note: the original `wd (deg)` is kept

Even though wind direction is re-encoded as `wd_sin` and `wd_cos`, the original `wd (deg)` column remains in the final 16 features. This is slightly redundant -- the XAI analysis later confirms that `wd (deg)` is among the 5 least important features and removes it during pruning.

---

## 8. The Final 16-Feature Vector

After feature engineering, the input to the model has 16 dimensions:

| # | Feature | Source | Category |
|---:|---|---|---|
| 1 | `T (degC)` | Original | Meteorological (target) |
| 2 | `p (mbar)` | Original | Meteorological |
| 3 | `rh (%)` | Original | Meteorological |
| 4 | `wv (m/s)` | Original | Meteorological |
| 5 | `max. wv (m/s)` | Original | Meteorological |
| 6 | `wd (deg)` | Original | Meteorological |
| 7 | `hour_sin` | Derived from Date Time | Cyclical temporal |
| 8 | `hour_cos` | Derived from Date Time | Cyclical temporal |
| 9 | `doy_sin` | Derived from Date Time | Cyclical temporal |
| 10 | `doy_cos` | Derived from Date Time | Cyclical temporal |
| 11 | `wd_sin` | Derived from wd (deg) | Wind cyclic |
| 12 | `wd_cos` | Derived from wd (deg) | Wind cyclic |
| 13 | `wx` | Derived from wv + wd | Wind Cartesian |
| 14 | `wy` | Derived from wv + wd | Wind Cartesian |
| 15 | `wind_gap` | Derived from max.wv - wv | Wind gust |
| 16 | `gust_ratio` | Derived from max.wv / wv | Wind gust |

### Information accounting

No new external information was added. The 16 features contain exactly the same information as the original 6 variables plus the datetime column -- just re-represented:

| Original source | Features derived | New info? |
|---|---|---|
| Date Time | hour_sin, hour_cos, doy_sin, doy_cos | No -- just re-encoded |
| wd (deg) | wd_sin, wd_cos | No -- just re-encoded |
| wv (m/s) + wd (deg) | wx, wy | No -- polar to Cartesian |
| max. wv + wv | wind_gap, gust_ratio | No -- arithmetic combinations |

### What the XAI later reveals

In Section 11, permutation importance ranks these 16 features. The 5 least important (later pruned) are:

| Pruned feature | Why low importance |
|---|---|
| `doy_sin` | Redundant with `doy_cos` for this model |
| `gust_ratio` | Redundant with `wind_gap` |
| `wd (deg)` | Redundant with `wd_sin` and `wd_cos` |
| `wd_sin` | Less informative than `wd_cos` for this dataset |
| `wy` | Less informative than `wx` for this location |

This confirms that some of the 16 features carry overlapping information. The feature engineering cast a wide net intentionally; XAI later prunes the redundancy.

---

## 9. Implementation Details

### Code organization

All feature engineering lives in a single module: `src/features/engineering.py`. Three functions:

| Function | Input | Output |
|---|---|---|
| `add_time_features(df, time_col)` | DataFrame with datetime column | Same DataFrame + 6 new columns (hour, dayofyear, hour_sin, hour_cos, doy_sin, doy_cos) |
| `add_wind_features(df)` | DataFrame with wv, max.wv, wd columns | Same DataFrame + 6 new columns (wd_sin, wd_cos, wx, wy, wind_gap, gust_ratio) |
| `get_final_feature_columns()` | Nothing | List of 16 feature names used for modelling |

### Defensive copying

Both functions start with `df = df.copy()`. This prevents **in-place modification** of the original dataframe. Without this, calling `add_time_features(df_model)` would alter `df_model` itself, which could cause bugs if the original is needed later.

### Degree-to-radian conversion

```python
wd_rad = np.deg2rad(df["wd (deg)"])
```

Python's `sin()` and `cos()` expect **radians**, not degrees. `deg2rad` converts: `radians = degrees * pi / 180`. Forgetting this conversion is a common bug -- the code would run without errors but produce wrong values.

### Notebook usage

```python
df_feat = add_time_features(df_model, time_col=TIME_COL)
df_feat = add_wind_features(df_feat)
final_feature_cols = get_final_feature_columns()  # the 16 names
```

The two functions are called sequentially (time features first, then wind features on the result). The final feature list is retrieved from `get_final_feature_columns()` to ensure consistency -- every part of the pipeline (baseline, EA, TimeGAN, XAI) uses the same 16 features in the same order.

---

## 10. Why These Transformations Matter for the GRU

### Without feature engineering

The GRU would receive 6 raw features per time step. It could still learn to forecast temperature, but it would need to:
- Discover that hour 23 and hour 0 are neighbors (non-trivial with raw integers)
- Learn the non-linear relationship between wind direction in degrees and its effect on temperature
- Figure out that wind speed + direction together form a vector (a complex interaction)

This is possible in theory but wastes model capacity on learning representations that we can provide for free.

### With feature engineering

The GRU receives 16 features that:
- Represent cyclic quantities on smooth circles (no discontinuities to learn)
- Decompose wind into Cartesian components (linear relationships are easy to learn)
- Pre-compute gust intensity metrics (the model doesn't need to learn subtraction/division)

The model can focus its capacity on learning **temporal dynamics** (how these features evolve over time and predict future temperature) rather than on learning basic feature representations.

### Analogy

It is like giving a student a problem in two ways:
- **Raw:** "A wind blows at 5 m/s from 270 degrees. What is the east-west component?" (requires knowing trigonometry)
- **Engineered:** "wx = -5.0 m/s, wy = 0.0 m/s." (the answer is already there)

Feature engineering does the trigonometry homework so the neural network can focus on the harder question: given this history, what will the temperature be tomorrow?
