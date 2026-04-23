# Notebook Q&A -- Study Questions with Answers

This document contains 10 questions for each of the 12 study documents, totalling 120 questions. Each question has 4 options with the correct answer marked.

---

## 00 -- Notebook Report (Overview)

**Q1.** What is the core forecasting task in this project?
- A) Univariate-input, multivariate-output, single-step prediction
- B) Multivariate-input, univariate-output, multi-step prediction
- C) Multivariate-input, multivariate-output, multi-step prediction
- D) Univariate-input, univariate-output, single-step prediction
> **Answer: B** -- The model receives multiple meteorological variables (multivariate-input), predicts only temperature (univariate-output), for the next 24 hours (multi-step).

**Q2.** What is the final resolution of the dataset after preprocessing?
- A) 10 minutes, ~420,000 rows
- B) 1 hour, ~70,041 rows
- C) 30 minutes, ~140,000 rows
- D) 1 day, ~2,920 rows
> **Answer: B** -- The original 10-minute data is resampled to hourly, producing ~70,041 observations.

**Q3.** How many input features does the model receive after feature engineering?
- A) 6 (original meteorological variables only)
- B) 10 (6 original + 4 temporal)
- C) 16 (6 original + 4 temporal + 6 wind-derived)
- D) 15 (all original Jena Climate columns)
> **Answer: C** -- 6 original + 4 cyclical time features + 6 wind-derived features = 16.

**Q4.** Which of the following is NOT one of the five advanced techniques in the pipeline?
- A) Evolutionary Optimization (GA)
- B) Transfer Learning from a pre-trained weather model
- C) Synthetic Data Generation (TimeGAN)
- D) Explainable AI (XAI)
> **Answer: B** -- The five techniques are: GRU baseline, evolutionary optimization, TimeGAN, XAI, and efficiency analysis. Transfer learning is not used.

**Q5.** What role does the persistence baseline play?
- A) It is the final selected model
- B) It provides a no-skill benchmark that any learned model must beat
- C) It generates synthetic data for augmentation
- D) It serves as the teacher model for knowledge distillation
> **Answer: B** -- Persistence (repeating the last observed value) is the minimum bar. Any useful model must outperform it.

**Q6.** What was the outcome of the TimeGAN experiment?
- A) It was used to augment training data, improving accuracy by 5%
- B) It achieved strong reconstruction but insufficient generative realism, and was not used in the final pipeline
- C) It replaced the GRU baseline as the primary forecasting model
- D) It was not implemented due to time constraints
> **Answer: B** -- The autoencoder reconstruction was good, but fully generated sequences lacked realism. TimeGAN remained a proof-of-concept.

**Q7.** What is the final selected model and its MAE?
- A) GRU Baseline Official, 1.650 degC
- B) Best Evolutionary GRU, 1.598 degC
- C) Pruned Evolutionary GRU, 1.575 degC
- D) TimeGAN-augmented GRU, 1.520 degC
> **Answer: C** -- The Pruned Evolutionary GRU achieved 1.575 degC MAE (best single seed: 1.569 degC).

**Q8.** How many total evaluations did the EA perform?
- A) 15 (one per generation)
- B) 100 (10 population x 10 generations)
- C) 300 (20 population x 15 generations)
- D) 1,000 (50 population x 20 generations)
> **Answer: C** -- Population size 20, 15 generations = 300 individual pipeline evaluations.

**Q9.** What did XAI-guided feature pruning do?
- A) Removed 5 least important features (16 -> 11), improving accuracy
- B) Added 5 new features based on saliency analysis
- C) Removed all wind features, degrading accuracy slightly
- D) Doubled the number of features using polynomial expansion
> **Answer: A** -- Permutation importance identified 5 weak features. Removing them improved MAE from 1.598 to 1.575 degC.

**Q10.** Which best summarizes the project's main conclusion?
- A) Larger models always produce better forecasts
- B) Combining evolutionary optimization with XAI-guided pruning produces a model that is more accurate, compact, and efficient than the hand-tuned baseline
- C) TimeGAN synthetic data was essential for achieving the best results
- D) The persistence baseline was surprisingly competitive with deep learning models
> **Answer: B** -- The closed-loop optimize -> explain -> prune -> validate methodology produced a model better on all fronts.

---

## 01 -- Project Goal (Foundational Concepts)

**Q1.** What does "multivariate-input" mean in this project?
- A) The model predicts multiple variables simultaneously
- B) The model receives multiple meteorological measurements at each time step
- C) The model uses multiple loss functions during training
- D) The model trains on multiple datasets
> **Answer: B** -- At each hour, the model receives 16 features (temperature, pressure, humidity, wind, etc.).

**Q2.** What problem does a GRU solve that a standard RNN cannot?
- A) GRUs can process images, standard RNNs cannot
- B) GRUs handle the vanishing/exploding gradient problem through gating mechanisms
- C) GRUs run faster because they have no recurrent connections
- D) GRUs can output variable-length sequences
> **Answer: B** -- Standard RNNs lose gradient signal over long sequences. GRU gates (reset and update) control information flow to maintain long-range dependencies.

**Q3.** What does the reset gate in a GRU control?
- A) How much of the previous hidden state to forget
- B) The learning rate during training
- C) Whether to use dropout or not
- D) The number of layers in the model
> **Answer: A** -- The reset gate decides how much past memory to discard. When near 0, the GRU ignores the past and reacts mainly to current input.

**Q4.** Which of the following is a hyperparameter (not a learned parameter)?
- A) The weights of the first GRU layer
- B) The bias values in the dense layer
- C) The learning rate (2e-4)
- D) The output of the recovery network
> **Answer: C** -- Learning rate is set before training and controls the training process. Weights and biases are learned automatically during training.

**Q5.** What is the purpose of regularization?
- A) To make the model train faster
- B) To prevent overfitting by discouraging the model from memorizing noise
- C) To increase the number of trainable parameters
- D) To convert the data from hourly to daily resolution
> **Answer: B** -- Regularization (dropout, L2, early stopping) prevents the model from performing well on training data but poorly on unseen data.

**Q6.** Why is gradient clipping needed for recurrent networks?
- A) It speeds up training by 50%
- B) It prevents exploding gradients, which cause unstable training with wild weight swings
- C) It reduces the number of parameters
- D) It converts gradients from float32 to float16
> **Answer: B** -- Recurrent networks are prone to gradient explosion. Clipping caps the gradient norm at a threshold (e.g., 2.0 or 5.0).

**Q7.** How does AdamW differ from standard Adam?
- A) AdamW uses a fixed learning rate, Adam uses an adaptive one
- B) AdamW decouples weight decay from the adaptive learning rate, applying it directly to weights
- C) AdamW only works with GRU models, Adam works with any model
- D) AdamW does not use momentum, Adam does
> **Answer: B** -- In Adam, weight decay gets entangled with adaptive learning rates. AdamW fixes this by applying weight decay directly to the weights.

**Q8.** What is the advantage of Huber loss over MSE?
- A) Huber loss is always faster to compute
- B) Huber loss behaves like MSE for small errors (smooth) and like MAE for large errors (robust to outliers)
- C) Huber loss works only with time series data
- D) Huber loss requires no hyperparameters
> **Answer: B** -- Huber combines the smooth gradients of MSE near zero with the outlier robustness of MAE beyond the delta threshold.

**Q9.** In the GA, what does elitism ensure?
- A) All individuals are mutated every generation
- B) The worst individuals are always removed
- C) The best individual(s) survive unchanged into the next generation
- D) Crossover produces identical children
> **Answer: C** -- Elitism preserves the top 2 individuals, guaranteeing the best solution found so far is never lost to random crossover/mutation.

**Q10.** Why is it significant that the EA optimizes the entire pipeline, not just model architecture?
- A) It reduces the number of hyperparameters to search
- B) It captures interactions between preprocessing, windowing, training, and architecture that individual optimization would miss
- C) It eliminates the need for a validation set
- D) It guarantees finding the global optimum
> **Answer: B** -- A different scaler may work better with a different architecture, and a different lookback may require a different learning rate. Joint optimization captures these interactions.

---

## 02 -- Dataset and Problem Definition (Section 3)

**Q1.** What is the original sampling frequency of the Jena Climate dataset?
- A) Every 1 minute
- B) Every 10 minutes
- C) Every 1 hour
- D) Every 6 hours
> **Answer: B** -- The raw dataset has ~420,224 observations at approximately 10-minute intervals.

**Q2.** Why is `dayfirst=True` critical when parsing dates?
- A) It makes parsing 10x faster
- B) Without it, `01.02.2009` would be parsed as January 2nd instead of February 1st (German date format)
- C) It converts dates to UTC timezone
- D) It enables leap year support
> **Answer: B** -- The dataset uses DD.MM.YYYY (European/German format). Without `dayfirst=True`, the parser assumes MM/DD/YYYY (US format), silently corrupting all dates.

**Q3.** Why must duplicates be removed before resampling?
- A) Duplicates cause the CSV file to be larger
- B) A duplicated 10-minute observation would be counted twice in the hourly mean, skewing the average
- C) Pandas cannot resample data with duplicates
- D) Duplicates indicate sensor malfunction and all data from that day should be discarded
> **Answer: B** -- If 14:20 is duplicated, the hourly mean for 14:00 is biased toward the conditions at 14:20.

**Q4.** Why was `.mean()` chosen for hourly resampling?
- A) It is the only aggregation function Pandas supports
- B) It preserves the central tendency of each hourly period while smoothing sub-hourly noise
- C) It is faster than `.median()` or `.first()`
- D) It produces fewer NaN values than other methods
> **Answer: B** -- Mean aggregation uses all 6 readings per hour, preserving information. `.first()` or `.last()` would discard 5 of 6 readings.

**Q5.** After resampling, 88 NaN values appeared. What caused them?
- A) A bug in the resampling code
- B) Some hourly bins had no 10-minute observations to average (gaps in the raw data)
- C) The mean function produces NaN for negative values
- D) The datetime parsing failed for 88 rows
> **Answer: B** -- When an entire hour has no raw observations (a gap in recording), `.mean()` produces NaN because there is nothing to average.

**Q6.** Why was wind direction (`wd (deg)`) included if neural networks can't interpret degrees directly?
- A) It was included by mistake
- B) It provides directional information that is later re-encoded as cyclic and Cartesian features in Section 5
- C) The GRU can natively understand angular measurements
- D) It is only used for visualization, not as model input
> **Answer: B** -- Wind direction is retained for feature engineering (sin/cos encoding and Cartesian decomposition). The raw degrees column also stays in the final 16 features, though XAI later shows it is redundant.

**Q7.** What does the forecasting task look like in terms of tensor shapes?
- A) Input: (24,), Output: (120, 16)
- B) Input: (120, 16), Output: (24,)
- C) Input: (120, 6), Output: (120, 6)
- D) Input: (144, 16), Output: (144, 16)
> **Answer: B** -- Input is 120 time steps x 16 features; output is 24 future temperature values.

**Q8.** Why is chronological ordering critical before splitting?
- A) Pandas requires sorted data for the `.iloc[]` operation
- B) Without temporal order, the train/val/test split could mix past and future data, causing leakage
- C) It makes the data load 50% faster
- D) It is required by the GRU architecture
> **Answer: B** -- The split must respect time so the model never trains on data from after the test period.

**Q9.** Which of the following is an example of data leakage in this project?
- A) Using early stopping during training
- B) Fitting the scaler on the entire dataset (including test data) before splitting
- C) Using RobustScaler instead of StandardScaler
- D) Resampling to hourly resolution
> **Answer: B** -- If the scaler's statistics include test-period data, the training data is normalized using future information.

**Q10.** What is the shape of the dataset at the end of Section 3?
- A) 420,224 rows x 16 columns
- B) 70,041 rows x 7 columns (datetime + 6 meteorological variables)
- C) 70,041 rows x 16 columns
- D) 48,885 rows x 7 columns
> **Answer: B** -- After resampling, cleaning, and variable selection: ~70,041 hourly rows with 7 columns. Feature engineering (Section 5) later expands to 16.

---

## 03 -- Data Preparation and Feature Engineering (Sections 4 & 5)

**Q1.** Why is `Date Time` not fed directly to the neural network?
- A) It would make the model too slow
- B) Neural networks operate on numeric tensors, and datetime objects are not numbers
- C) The datetime column contains missing values
- D) It would cause data leakage
> **Answer: B** -- A datetime like `2015-07-03 14:00:00` is a structured object. Its temporal information is extracted as numbers (hour, day-of-year) and then cyclically encoded.

**Q2.** What is the cyclic discontinuity problem with raw hour-of-day encoding?
- A) Hours 0-11 are always warmer than hours 12-23
- B) Hour 23 and hour 0 are adjacent in time but numerically distant (23 vs 0), creating an artificial gap
- C) Hour values are too small for neural networks to detect
- D) The hour column has missing values at midnight
> **Answer: B** -- A network measuring distance would think hours 23 and 0 are 23 units apart, when they are actually 1 hour apart.

**Q3.** Why are both sine AND cosine needed for cyclic encoding?
- A) Sine alone is computationally unstable
- B) Cosine alone cannot distinguish morning from evening
- C) Sine alone creates ambiguities (e.g., hour 3 and hour 9 have the same sine value), which cosine resolves
- D) It doubles the number of features, which always improves accuracy
> **Answer: C** -- sin(3h) = sin(9h) = 0.707, so sine alone cannot distinguish them. cos(3h) = 0.707 and cos(9h) = -0.707, breaking the ambiguity.

**Q4.** What does the formula `hour_sin = sin(2*pi*hour/24)` compute?
- A) The temperature at that hour
- B) The position of the hour on a unit circle with period 24, giving the sine coordinate
- C) The number of minutes past midnight
- D) A random value between -1 and 1
> **Answer: B** -- `2*pi*hour/24` converts the hour to an angle (radians) that completes a full circle over 24 hours. `sin()` gives the y-coordinate on that circle.

**Q5.** Why is 365.25 used instead of 365 for day-of-year encoding?
- A) It accounts for leap years (one extra day every 4 years), keeping the seasonal encoding aligned across years
- B) It is a more precise value of pi
- C) It compensates for daylight saving time
- D) It was chosen arbitrarily
> **Answer: A** -- (365+365+365+366)/4 = 365.25. Without this, day 366 in a leap year would exceed a full rotation.

**Q6.** What is the key difference between `wd_sin/wd_cos` and `wx/wy`?
- A) They encode different wind variables
- B) `wd_sin/wd_cos` encode direction only; `wx/wy` combine direction with speed into a vector
- C) `wx/wy` are more accurate measurements
- D) There is no difference; they are the same values
> **Answer: B** -- Cyclic encoding captures "which direction?" while Cartesian decomposition captures "how much wind is going where?" (speed baked into direction).

**Q7.** Why does the `gust_ratio` formula include `1e-6`?
- A) To make the ratio always greater than 1
- B) To prevent division by zero when sustained wind speed is exactly 0
- C) To convert from m/s to km/h
- D) To add Gaussian noise for regularization
> **Answer: B** -- `max_wv / (wv + 1e-6)` avoids infinity when `wv = 0`. The epsilon (0.000001) has negligible effect on non-zero values.

**Q8.** How are the 16 final features categorized?
- A) 6 original + 4 cyclical temporal + 6 wind-derived
- B) 8 original + 4 cyclical temporal + 4 wind-derived
- C) 10 original + 6 derived
- D) 16 original meteorological variables
> **Answer: A** -- 6 raw meteorological + 4 time features (hour_sin, hour_cos, doy_sin, doy_cos) + 6 wind features (wd_sin, wd_cos, wx, wy, wind_gap, gust_ratio).

**Q9.** Which pruned feature was redundant because its cyclic pair partner already captured the same cycle?
- A) `T (degC)` -- redundant with `p (mbar)`
- B) `doy_sin` -- redundant with `doy_cos` (both encode the seasonal cycle)
- C) `hour_sin` -- redundant with `hour_cos`
- D) `wind_gap` -- redundant with `gust_ratio`
> **Answer: B** -- The model only needed one component of the seasonal pair. `doy_sin` was the least informative of the two and was pruned.

**Q10.** Does feature engineering add new information to the dataset?
- A) Yes, it adds external weather forecast data
- B) Yes, it adds satellite imagery features
- C) No, it re-represents existing information in forms easier for the neural network to learn from
- D) No, it actually removes information by reducing dimensionality
> **Answer: C** -- All 16 features are derived from the original 6 variables + datetime. No external data is added. The transformations respect physical structure (cyclic, Cartesian).

---

## 04 -- Split, Scaling and Windowing (Section 6)

**Q1.** What is the train/validation/test split ratio?
- A) 80% / 10% / 10%
- B) 70% / 15% / 15%
- C) 60% / 20% / 20%
- D) 50% / 25% / 25%
> **Answer: B** -- 70% training (~2009-mid 2014), 15% validation (~mid 2014-early 2016), 15% test (~early 2016-end 2016).

**Q2.** How is the validation set used differently from the test set?
- A) The validation set is used for final reporting; the test set is used during training
- B) The validation set is used repeatedly for tuning and EA fitness; the test set is used only once at the end
- C) They are identical and interchangeable
- D) The validation set contains synthetic data; the test set contains real data
> **Answer: B** -- Validation is used for early stopping, LR reduction, and EA fitness (repeated use). Test is evaluated only once for final unbiased results.

**Q3.** What would go wrong if you shuffled data before splitting for time series?
- A) Training would be slower
- B) The training set could contain data from after the test period, leaking future information
- C) The model would learn faster
- D) Nothing; shuffling is always beneficial
> **Answer: B** -- Shuffled splits mix past and future data. The model could train on December 2016 data and be tested on March 2016, having "seen the future."

**Q4.** Why do raw features need scaling before training?
- A) Neural networks require all inputs to be between 0 and 1
- B) Features with larger numeric ranges (e.g., pressure ~1000) would dominate gradient updates, overshadowing smaller-scale features
- C) Scaling converts the data from Celsius to Fahrenheit
- D) Scaling removes outliers from the data
> **Answer: B** -- Without scaling, pressure (~1000) produces much larger gradients than hour_sin (~1), causing the model to learn disproportionately from high-magnitude features.

**Q5.** Which scaler is most robust to outliers, and why?
- A) StandardScaler, because it uses the mean
- B) MinMaxScaler, because it uses the full range
- C) RobustScaler, because it uses median and IQR, which are resistant to extreme values
- D) All three are equally robust
> **Answer: C** -- Median and IQR are not affected by outliers. A single extreme temperature reading shifts the mean/std (Standard) or stretches the range (MinMax), but barely moves the median/IQR (Robust).

**Q6.** What goes wrong if you fit the scaler on the entire dataset?
- A) Training becomes 10x slower
- B) The scaler's statistics (mean, std) incorporate future data from the test set, causing data leakage
- C) The scaler produces NaN values
- D) Nothing; it is the recommended approach
> **Answer: B** -- The test set's statistics subtly influence how training data is normalized, giving the model implicit knowledge of future conditions.

**Q7.** How is a single supervised (X, y) sample created from the time series?
- A) X = one random row, y = the next row
- B) X = a window of LOOKBACK consecutive hours (all 16 features), y = the next HORIZON hours of temperature only
- C) X = the full training set, y = the full test set
- D) X = one feature, y = all other features
> **Answer: B** -- The sliding window takes 120 hours of all 16 features as input and the next 24 hours of temperature as the target.

**Q8.** With LOOKBACK=120 and HORIZON=24, what is the shape of one sample?
- A) X: (120, 16), y: (24,)
- B) X: (16, 120), y: (1, 24)
- C) X: (24, 16), y: (120,)
- D) X: (144, 16), y: (144,)
> **Answer: A** -- X has 120 time steps x 16 features. y has 24 future temperature values (univariate output).

**Q9.** How many samples does windowing produce from a time series of length T?
- A) T
- B) T - LOOKBACK
- C) T - LOOKBACK - HORIZON + 1
- D) T / (LOOKBACK + HORIZON)
> **Answer: C** -- Each valid starting position needs enough room for both the lookback window and the forecast horizon.

**Q10.** What does each dimension of `X_train` shape `(48885, 120, 16)` represent?
- A) 48,885 features, 120 samples, 16 time steps
- B) 48,885 samples, 120 time steps per sample, 16 features per time step
- C) 48,885 epochs, 120 batches, 16 layers
- D) 48,885 hours, 120 features, 16 targets
> **Answer: B** -- First dim = number of training samples, second = lookback window length, third = number of input features.

---

## 05 -- Baseline Models (Section 7)

**Q1.** Why is the persistence baseline useful despite being a terrible forecast?
- A) It provides training data for the GRU
- B) It sets a minimum performance bar -- any model that can't beat "repeat the last value" is useless
- C) It identifies outliers in the dataset
- D) It pre-trains the GRU weights
> **Answer: B** -- Without persistence as a reference, you can't tell whether a model's MAE is genuinely good or just looks impressive in isolation.

**Q2.** How does the persistence forecast work?
- A) It predicts the average temperature of the input window
- B) It repeats the last observed temperature value across all 24 forecast hours
- C) It uses linear extrapolation from the last 3 hours
- D) It predicts zero for all hours
> **Answer: B** -- `X[:, -1, target_idx]` extracts the last time step's temperature, then `np.repeat` copies it 24 times.

**Q3.** What optimizer and loss function does the official GRU baseline use?
- A) Adam with MSE loss
- B) SGD with MAE loss
- C) AdamW with Huber loss (delta=1.0)
- D) AdamW with MAE loss
> **Answer: C** -- The baseline uses AdamW optimizer and Huber loss with delta=1.0 ("huber1"). The EA later found MAE loss works better.

**Q4.** How many parameters does the first GRU layer (96 units, input=16) contribute?
- A) 96 * 16 = 1,536
- B) 3 * (16*96 + 96*96 + 2*96) = 32,832
- C) 16 + 96 = 112
- D) 96 * 96 = 9,216
> **Answer: B** -- 3 gates, each with input weights (16*96), recurrent weights (96*96), and 2 biases (2*96).

**Q5.** What does `return_sequences=True` mean on a GRU layer?
- A) The layer returns only the last hidden state
- B) The layer returns a hidden state for every time step in the input sequence
- C) The layer processes the sequence in reverse order
- D) The layer returns the input sequence unchanged
> **Answer: B** -- `True` outputs shape (batch, 120, units) -- needed when another GRU follows. `False` outputs only the last state (batch, units).

**Q6.** How do early stopping and LR reduction interact?
- A) They cannot be used together
- B) LR reduction triggers first (patience=3) to give the model a chance; if no improvement after 3 more epochs, early stopping (patience=6) activates
- C) Early stopping always triggers before LR reduction
- D) They are identical -- just different names for the same callback
> **Answer: B** -- LR reduction halves the learning rate after 3 stale epochs. If 6 total stale epochs pass, early stopping ends training and restores the best weights.

**Q7.** Why are both scaled and original-scale metrics reported?
- A) Scaled metrics are always more accurate
- B) Scaled metrics are useful for comparing models under the same scaler; original-scale (degC) provides physical interpretation
- C) Original-scale metrics are only used for visualization
- D) They always give the same ranking of models
> **Answer: B** -- Scaled metrics monitor training; original-scale (degC) is the primary metric for decision-making and reporting.

**Q8.** How is inverse scaling performed for the baseline (StandardScaler)?
- A) `x_original = x_scaled / std + mean`
- B) `x_original = x_scaled * std + mean`
- C) `x_original = (x_scaled - mean) * std`
- D) `x_original = x_scaled * mean + std`
> **Answer: B** -- StandardScaler forward: `(x - mean) / std`. Inverse: `x * std + mean`.

**Q9.** What are the baseline MAE results in degrees Celsius?
- A) Persistence: 1.650, GRU: 3.144
- B) Persistence: 3.144, GRU: 1.650
- C) Persistence: 0.36, GRU: 0.19
- D) Persistence: 4.254, GRU: 2.193
> **Answer: B** -- Persistence: 3.144 degC MAE, GRU Baseline: 1.650 degC MAE. Option D shows RMSE, not MAE.

**Q10.** Why does the GRU forecast appear smoother than the true temperature signal?
- A) The model applies a smoothing filter to its output
- B) Uncertainty compounds over the 24-hour horizon, so the model hedges toward the mean instead of predicting sharp oscillations
- C) The training data was smoothed before use
- D) The GRU architecture cannot produce non-smooth outputs
> **Answer: B** -- Multi-step forecasts become increasingly uncertain at later steps. The model learns to produce conservative (smooth) predictions that minimize average error.

---

## 06 -- TimeGAN Synthetic Data (Section 8)

**Q1.** What is the purpose of generating synthetic data?
- A) To replace the test set with generated data
- B) To augment the training set, potentially improving model robustness or generalization
- C) To create a validation set when one is unavailable
- D) To compress the dataset for faster loading
> **Answer: B** -- Synthetic data could expand the training set with realistic examples, helping the model generalize better.

**Q2.** Which TimeGAN sub-network learns temporal dynamics in latent space?
- A) Embedder
- B) Recovery
- C) Supervisor
- D) Discriminator
> **Answer: C** -- The Supervisor is trained to predict the next latent time step from the current one, capturing how the latent representation evolves over time.

**Q3.** What does TimeGAN add over a standard GAN for time series?
- A) More discriminator layers
- B) An embedding space with a supervisor network that enforces temporal coherence
- C) A larger generator network
- D) Pre-training on ImageNet
> **Answer: B** -- Standard GANs have no mechanism for temporal dynamics. TimeGAN adds an embedder/recovery (latent space) and a supervisor (temporal transitions).

**Q4.** Why is the TimeGAN sequence length set to 144?
- A) It is the maximum sequence length GRUs can process
- B) It matches LOOKBACK + HORIZON (120 + 24), so generated sequences can be split directly into input-output pairs
- C) 144 is a power of 2 minus 1
- D) It was chosen to match the batch size
> **Answer: B** -- Each generated 144-step sequence can be split into 120-step input and 24-step target, matching the forecasting setup.

**Q5.** What happens during Phase 1 (autoencoder pretraining)?
- A) The generator and discriminator compete adversarially
- B) The embedder and recovery networks are jointly trained to compress and reconstruct real sequences
- C) The supervisor learns temporal transitions
- D) Synthetic sequences are generated and evaluated
> **Answer: B** -- The autoencoder (embedder + recovery) minimizes reconstruction MSE on real data, learning a meaningful latent representation.

**Q6.** In the generator loss `g_loss = g_loss_u + 100*g_loss_s + gamma*g_loss_v`, why is the supervised loss weighted 100x?
- A) It is a bug in the implementation
- B) Temporal coherence is critical; without strong weighting, the adversarial loss could destroy temporal structure
- C) It compensates for the supervisor having fewer parameters
- D) The value 100 was found through grid search
> **Answer: B** -- The 100x weight ensures the generator prioritizes producing temporally coherent sequences, even as the adversarial game pushes it toward fooling the discriminator.

**Q7.** What does the discriminator threshold (`d_loss > 0.15`) prevent?
- A) The generator from becoming too strong
- B) The discriminator from becoming too strong too early, which would give the generator uninformative gradients
- C) Training from exceeding 15 epochs
- D) The loss from becoming negative
> **Answer: B** -- If the discriminator dominates, the generator receives gradients saying "everything is obviously fake" with no useful direction. The threshold pauses the discriminator to let the generator catch up.

**Q8.** What is the generation path from noise to synthetic data?
- A) Noise -> Discriminator -> Recovery -> Data
- B) Noise -> Generator -> Supervisor -> Recovery -> Data
- C) Noise -> Embedder -> Generator -> Data
- D) Noise -> Recovery -> Supervisor -> Data
> **Answer: B** -- Random noise is transformed by the Generator into raw latent sequences, refined by the Supervisor for temporal coherence, then decoded by Recovery into data space.

**Q9.** The autoencoder reconstruction MSE was excellent (~0.0012). Why was TimeGAN still not used?
- A) Reconstruction tests the embedder-recovery path (real data in, real data out). Generation uses the full path (noise in), which produced sequences with insufficient realism and variability.
- B) The MSE was too high for practical use
- C) Reconstruction is the same as generation
- D) The project ran out of time to integrate it
> **Answer: A** -- Reconstruction is easy (structured input). Generation is hard (unstructured noise). The generator failed to produce the full diversity and dynamics of real weather.

**Q10.** What would help TimeGAN produce more realistic sequences?
- A) Shorter training (fewer epochs)
- B) Smaller latent dimension
- C) Longer adversarial training, larger latent dimension, or alternative architectures like diffusion models
- D) Removing the supervisor network
> **Answer: C** -- 15 adversarial epochs was short, and hidden_dim=24 may be too small. Newer approaches (diffusion models) are often more stable than GANs for generation.

---

## 07 -- Evolutionary Optimization (Section 9)

**Q1.** Why was a GA chosen instead of Bayesian optimization (e.g., Optuna)?
- A) GAs are always faster than Bayesian methods
- B) The course requires evolutionary/nature-inspired methods
- C) Bayesian optimization cannot handle 17 dimensions
- D) GAs guarantee finding the global optimum
> **Answer: B** -- The GA satisfies the course requirement for evolutionary computation. Optuna would be more sample-efficient but wouldn't demonstrate evolutionary concepts.

**Q2.** How many optimization dimensions does the search space have, and what are the five categories?
- A) 10 dimensions: architecture, training, loss, optimizer, batch size
- B) 17 dimensions: architecture, regularization, training, preprocessing, windowing
- C) 5 dimensions: one per category
- D) 20 dimensions: architecture, regularization, training, preprocessing, windowing
> **Answer: B** -- 6 architecture + 4 regularization + 5 training + 1 preprocessing + 1 windowing = 17 genes.

**Q3.** What is a genotype in this context?
- A) The DNA sequence of the model developer
- B) A dictionary mapping 17 gene names to specific values from the search space, encoding a complete pipeline configuration
- C) The model's trained weights
- D) The training loss curve
> **Answer: B** -- Example: `{"n_layers": 2, "units1": 64, "loss_name": "mae", "scaler_name": "robust", "lookback": 144, ...}`.

**Q4.** Why is the constraint `units2 <= units1` enforced?
- A) It reduces the total number of possible configurations by 50%
- B) GRU layers that widen after the first layer are architecturally unusual and unproductive; funneling (wide -> narrow) is standard
- C) Keras requires it for technical reasons
- D) It ensures the model has exactly 86,744 parameters
> **Answer: B** -- A widening architecture wastes the EA's budget exploring configurations that rarely work well. Funneling focuses computation on competitive designs.

**Q5.** Why is the fitness metric computed in degC rather than scaled space?
- A) degC values are always smaller and easier to compare
- B) Different individuals may use different scalers, making scaled metrics incomparable; degC provides a common physical scale
- C) Scaled metrics cannot be computed on the validation set
- D) degC metrics are faster to compute
> **Answer: B** -- An MAE of 0.13 under RobustScaler and 0.19 under StandardScaler are in different units. Converting to degC makes all individuals comparable regardless of scaler choice.

**Q6.** In tournament selection with k=3, what happens?
- A) The top 3 individuals from the entire population are selected
- B) 3 random individuals are picked, and the one with the lowest fitness (best MAE) wins
- C) All individuals compete, and the bottom 3 are eliminated
- D) 3 individuals are merged into one
> **Answer: B** -- Random selection of 3, keep the best. This balances exploitation (good individuals are likely to win) with exploration (mediocre individuals still have a chance).

**Q7.** Can uniform crossover produce invalid genotypes?
- A) No, crossover always produces valid configurations
- B) Yes, because combining genes from different parents can violate constraints (e.g., units2 > units1); constraints are re-applied afterward
- C) Yes, and they are discarded
- D) Crossover is not used in this GA
> **Answer: B** -- Parent 1 with units1=64 and Parent 2 with units2=128 could produce a child with units1=64, units2=128. `apply_genotype_constraints()` fixes this.

**Q8.** With mutation rate 0.2 and 17 genes, how many genes are expected to mutate per individual?
- A) 0.2 genes
- B) 2 genes
- C) ~3.4 genes (17 * 0.2)
- D) 17 genes (all of them)
> **Answer: C** -- Each gene has a 20% independent chance. Expected value: 17 * 0.2 = 3.4 mutations per child.

**Q9.** What would happen without elitism?
- A) Training would be faster
- B) The best solution found so far could be lost to random crossover and mutation
- C) The population would converge faster
- D) Nothing; elitism has no effect
> **Answer: B** -- Without elitism, the best genotype could be modified by crossover/mutation into a worse one, and its original form would be lost forever.

**Q10.** Name the three most impactful differences the EA discovered vs. the baseline.
- A) More layers, higher dropout, smaller batch size
- B) RobustScaler (vs Standard), MAE loss (vs Huber), 144h lookback (vs 120h)
- C) Larger units (128->128), MSE loss, MinMaxScaler
- D) 3 GRU layers, Gaussian noise, Adam optimizer
> **Answer: B** -- RobustScaler handles outliers better, MAE aligns directly with the evaluation metric, and 144h lookback provides more temporal context. All three contributed to the improvement.

---

## 08 -- Retraining and Model Selection (Section 10)

**Q1.** Why are the final models retrained from scratch?
- A) The EA weights are stored in a format Keras cannot load
- B) The EA used a limited budget (20 epochs); retraining with 50 epochs provides a fairer, more reliable comparison
- C) The EA models were accidentally deleted
- D) Retraining is required by the Keras API
> **Answer: B** -- EA training was kept short (20 epochs) to evaluate 300 individuals. Retraining with 50 epochs lets each model reach its full potential.

**Q2.** Why does each candidate need its own data preparation?
- A) They use different random seeds
- B) They use different scalers (Standard vs Robust) and different lookback windows (120 vs 144), so the input tensors differ
- C) One uses synthetic data and the other doesn't
- D) They were developed by different team members
> **Answer: B** -- Different scalers produce different normalizations; different lookbacks produce different tensor shapes. Each model needs data prepared according to its own pipeline configuration.

**Q3.** The baseline converged at epoch ~14 while the EA model converged at epoch ~43. Why?
- A) The baseline is a better model
- B) The EA model uses MAE loss (smoother gradients, slower convergence), larger batches (fewer updates per epoch), and longer lookback (more context to learn from)
- C) The EA model has more parameters
- D) The baseline was pre-trained
> **Answer: B** -- MAE loss, batch_size=256, and lookback=144 all contribute to slower but steadier learning. The EA model benefits from the full 50-epoch budget.

**Q4.** Why are scaled metrics not directly comparable between the two models?
- A) They use different loss functions
- B) They use different scalers (StandardScaler vs RobustScaler), so the numeric ranges of "scaled" values differ
- C) One model is larger than the other
- D) Scaled metrics are always identical
> **Answer: B** -- An MAE of 0.19 under StandardScaler and 0.13 under RobustScaler are measured in different units. Only degC metrics are on the same physical scale.

**Q5.** What are the final test MAE results in degrees Celsius?
- A) Baseline: 1.598, EA: 1.650
- B) Baseline: 1.650, EA: 1.598
- C) Baseline: 0.1909, EA: 0.1295
- D) Baseline: 3.144, EA: 1.575
> **Answer: B** -- Baseline: 1.650 degC, EA: 1.598 degC. Option C shows scaled metrics. Option D compares persistence with the pruned model.

**Q6.** What does multi-seed robustness testing verify?
- A) That the model works on different datasets
- B) That the EA's best configuration generalizes across different random weight initializations, not just seed 42
- C) That the model can handle different scaler types
- D) That the model predicts correctly for different seasons
> **Answer: B** -- If the result only works with seed 42, it is not genuinely better -- it was a lucky initialization. Multi-seed testing rules this out.

**Q7.** What is the mean and std of the EA model's MAE across seeds 7, 21, 42?
- A) 1.650 +/- 0.050
- B) 1.593 +/- 0.016
- C) 1.575 +/- 0.009
- D) 3.144 +/- 0.500
> **Answer: B** -- The full EA model (16 features) achieved 1.593 +/- 0.016. Option C is the pruned model's result.

**Q8.** Does the worst seed for the EA model still beat the baseline?
- A) No, the worst seed (1.610) is worse than the baseline (1.650)
- B) Yes, even the worst seed (~1.610) is better than the baseline (1.650)
- C) They are exactly the same
- D) It depends on the scaler
> **Answer: B** -- 1.610 < 1.650, so even the worst EA seed outperforms the baseline. This confirms the improvement is robust.

**Q9.** Which model is selected at the end of Section 10?
- A) The persistence baseline
- B) The GRU Baseline Official
- C) The Best Evolutionary GRU, which then goes to XAI analysis and potential pruning
- D) A new transformer model
> **Answer: C** -- The EA model is selected for subsequent XAI analysis (Section 11), where pruning further improves it.

**Q10.** Why is the EA model described as being both "more accurate AND more compact"?
- A) It uses a trick to count parameters differently
- B) The baseline was over-parameterized (86,744 params, 96->64 GRU); the EA found a leaner architecture (63,512 params, 64->64 GRU) that generalizes better
- C) All evolutionary models are always smaller
- D) The pruned features were counted as parameters
> **Answer: B** -- The baseline's extra 23,000+ parameters were not contributing to better predictions. The EA discovered that a smaller model generalizes better.

---

## 09 -- Explainable AI (Section 11)

**Q1.** What is the difference between global and local explainability?
- A) Global explains one prediction; local explains all predictions
- B) Global reveals which features matter across all predictions; local reveals which inputs matter for one specific prediction
- C) Global uses gradient saliency; local uses permutation importance
- D) They are the same thing
> **Answer: B** -- Global = permutation importance (all test samples). Local = gradient saliency (one specific forecast).

**Q2.** What are the three steps of permutation importance?
- A) Train, predict, evaluate
- B) Compute baseline MAE, shuffle one feature across samples, measure MAE increase
- C) Compute gradients, take absolute values, rank features
- D) Split data, scale features, create windows
> **Answer: B** -- (1) Get baseline MAE with intact data, (2) shuffle one feature to break its relationship with the target, (3) measure how much MAE increases.

**Q3.** Why shuffle features across samples rather than setting them to zero?
- A) Shuffling is faster to compute
- B) Zeroing changes the input distribution in ways the model never saw; shuffling preserves realistic values while only breaking the sample-specific relationship
- C) Zero values cause division by zero errors
- D) There is no difference between the two approaches
> **Answer: B** -- The model was trained on realistic feature values. Shuffling keeps the marginal distribution intact. Zeroing introduces out-of-distribution inputs that could cause unpredictable behavior.

**Q4.** Which feature is most important according to permutation importance?
- A) `p (mbar)` (atmospheric pressure)
- B) `hour_sin` (diurnal cycle)
- C) `T (degC)` (past temperature)
- D) `gust_ratio` (relative gust strength)
> **Answer: C** -- Past temperature is by far the dominant predictor. Temperature has strong autocorrelation -- today's temperature is the best predictor of tomorrow's.

**Q5.** What does a high absolute gradient at position (t=118, feature=T) in the saliency map mean?
- A) The temperature at time step 118 is very high
- B) The model's prediction is highly sensitive to the temperature value at time step 118 -- a small change there would significantly alter the forecast
- C) The model ignores that input
- D) There is an error in the data at that position
> **Answer: B** -- High saliency = the model pays close attention to that input value. It's "important" for this specific prediction.

**Q6.** What does the temporal saliency profile (Figure 11) reveal?
- A) All time steps are equally important
- B) The earliest time steps are most important
- C) Importance is concentrated near the most recent observations, with a sharp spike in the final segment
- D) Saliency is random across time steps
> **Answer: C** -- The model relies primarily on recent history (last ~10-20 hours). Distant past contributes little, consistent with short-term weather forecasting.

**Q7.** What do the five pruned features have in common?
- A) They are all temperature-related
- B) They are all redundant -- each is either a duplicate encoding of information already captured by another feature, or carries minimal predictive signal
- C) They are all wind features
- D) They were all added during feature engineering
> **Answer: B** -- doy_sin (redundant with doy_cos), gust_ratio (redundant with wind_gap), wd(deg) (redundant with wd_sin/cos), wd_sin (less informative than wd_cos), wy (less informative than wx).

**Q8.** Why did removing 5 features improve accuracy?
- A) Fewer features always improve accuracy
- B) The removed features added noise and redundancy, wasting model capacity; removing them let the model focus on genuinely informative inputs
- C) The pruned model was trained for more epochs
- D) The features were corrupted in the original dataset
> **Answer: B** -- The features were not just uninformative but slightly detrimental -- they added 5 extra noise dimensions the model had to process.

**Q9.** How does the pruned model's seed stability compare to the full model?
- A) The pruned model is less stable (higher std)
- B) They have identical stability
- C) The pruned model is more stable (std=0.009 vs 0.016), confirming that removing noisy features reduced initialization-dependent variance
- D) Stability was not measured for the pruned model
> **Answer: C** -- Lower std means the pruned model's performance varies less across random seeds, making it more reliable.

**Q10.** What is the closed-loop methodology?
- A) Training -> Testing -> Deployment
- B) Optimize (EA) -> Explain (XAI) -> Prune (remove weak features) -> Validate (multi-seed testing)
- C) Load data -> Clean data -> Train model
- D) Forward pass -> Backward pass -> Weight update
> **Answer: B** -- The project's key contribution: EA finds a good model, XAI reveals what's important, pruning removes noise, multi-seed testing confirms robustness.

---

## 10 -- Efficiency Analysis (Section 12)

**Q1.** Which of the following is NOT an efficiency metric measured in this section?
- A) Training time
- B) Inference latency
- C) F1 score
- D) Trainable parameter count
> **Answer: C** -- F1 score is a classification metric. The section measures training time, memory, parameters, and inference latency.

**Q2.** Why does the EA model train ~46% faster despite longer input sequences (144 vs 120)?
- A) It uses a GPU while the baseline uses CPU
- B) Fewer parameters (63,512 vs 86,744) and larger batch size (256 vs 128) more than compensate for the longer input
- C) It skips early stopping
- D) It uses half-precision (float16) arithmetic
> **Answer: B** -- Smaller GRU layers mean fewer multiplications per step. Larger batches mean fewer gradient updates per epoch. Both outweigh the cost of longer sequences.

**Q3.** Why are warmup runs necessary when measuring inference latency?
- A) To heat up the GPU hardware
- B) The first few runs are slow due to TensorFlow graph compilation and cache loading; warmup ensures timed runs reflect steady-state performance
- C) To fill the training data cache
- D) Warmup is optional and has no effect
> **Answer: B** -- TensorFlow JIT-compiles operations on first use. Without warmup, the first measurement would include compilation time, skewing results.

**Q4.** What does `_sync_if_needed()` do, and why is it critical?
- A) It synchronizes the random seed across threads
- B) It forces TensorFlow to complete all pending GPU operations before measuring time, preventing measurement of launch time instead of computation time
- C) It saves the model weights to disk
- D) It synchronizes the training and validation data
> **Answer: B** -- GPU operations are asynchronous. Without sync, `time.perf_counter()` captures when the operation was launched, not when it finished.

**Q5.** What is the difference between p50 and p95 latency?
- A) p50 is always exactly half of p95
- B) p50 is the median (typical latency); p95 is the value below which 95% of measurements fall (worst reasonable case)
- C) p50 measures GPU time; p95 measures CPU time
- D) p50 is for batch size 50; p95 is for batch size 95
> **Answer: B** -- p50 tells you "what latency will you usually see?" p95 tells you "what's the worst case you should plan for?"

**Q6.** How many fewer parameters does the pruned model have vs. the baseline, and where does the reduction come from?
- A) 24,192 fewer, from removing 5 GRU layers
- B) 24,192 fewer, mostly from smaller GRU layers (64 vs 96 units) and 5 fewer input features reducing first-layer weights
- C) 960 fewer, only from 5 fewer input features
- D) No difference in parameters
> **Answer: B** -- Baseline: 86,744. Pruned: 62,552. Difference: 24,192. Most comes from smaller GRU layers; 960 comes from 5 fewer input features at the first layer.

**Q7.** Were inference latencies significantly different across the three models?
- A) Yes, the baseline was 10x slower
- B) No, all three were similar (~12-13ms per batch of 32) because at these model sizes, inference is fast regardless
- C) The pruned model was 5x faster
- D) Latency was not measured
> **Answer: B** -- All three models are small enough that inference overhead (GPU launch, memory transfer) dominates. Parameter count differences don't significantly affect inference speed.

**Q8.** What does the accuracy vs. efficiency scatter plot (Figure 15) show?
- A) All three models have identical accuracy and efficiency
- B) The Pruned EA GRU is closest to the ideal bottom-left corner (low MAE, few parameters)
- C) The baseline is the most efficient model
- D) There is no relationship between accuracy and efficiency
> **Answer: B** -- The ideal position is low MAE + few parameters. The pruned model achieves the best accuracy with the fewest parameters.

**Q9.** What is the project's key efficiency finding?
- A) Larger models are always more efficient
- B) The evolutionary optimization and XAI pruning found a model that is simultaneously more accurate AND more efficient than the baseline -- no trade-off
- C) Efficiency doesn't matter for weather forecasting
- D) The baseline was the most efficient model
> **Answer: B** -- The pruned EA model beats the baseline on accuracy, parameters, and training time simultaneously because the baseline was over-parameterized.

**Q10.** Under what circumstances would the accuracy-efficiency trade-off reappear?
- A) Never; it can always be avoided
- B) If you pushed further -- reducing parameters below ~62,000 or simplifying the architecture beyond the current optimal point
- C) Only when using a different dataset
- D) Only when using GPUs instead of CPUs
> **Answer: B** -- The current result sits at a sweet spot. Further simplification (fewer units, fewer features, fewer layers) would eventually start hurting accuracy.

---

## 11 -- Discussion and Conclusion (Sections 13-15)

**Q1.** What is the MAE progression from persistence to the final model?
- A) 3.144 -> 1.650 -> 1.598 -> 1.575 degC
- B) 1.575 -> 1.598 -> 1.650 -> 3.144 degC
- C) 3.144 -> 1.575 -> 1.598 -> 1.650 degC
- D) 0.36 -> 0.19 -> 0.13 -> 0.12 (scaled)
> **Answer: A** -- Persistence (3.144) -> Baseline GRU (1.650) -> EA GRU (1.598) -> Pruned EA GRU (1.575).

**Q2.** Which of the following was NOT discovered by the EA?
- A) RobustScaler outperforms StandardScaler
- B) MAE loss outperforms Huber loss
- C) Transformer architecture outperforms GRU
- D) 144-hour lookback outperforms 120-hour lookback
> **Answer: C** -- The EA only searched over GRU configurations. Transformer architectures were not included in the search space and are listed as future work.

**Q3.** Why was the pruned model more stable across seeds than the full model?
- A) It was trained for more epochs
- B) Removing 5 noisy features reduced the variability introduced by random weight initialization on those unused dimensions
- C) It used a fixed random seed
- D) The pruned features were the most variable in the dataset
> **Answer: B** -- Fewer noisy inputs means less opportunity for random weight initialization to produce different learning trajectories.

**Q4.** How did XAI validate that the model's behavior is physically plausible?
- A) By comparing predictions to a physics simulation
- B) By showing that past temperature dominates, seasonal/diurnal cycles matter, and recent history is most important -- all consistent with meteorological knowledge
- C) By computing the Navier-Stokes equations
- D) By validating against satellite imagery
> **Answer: B** -- The XAI findings match what meteorologists would expect: temperature autocorrelation, cyclical patterns, and recency bias.

**Q5.** Which is NOT a limitation acknowledged in the discussion?
- A) Only GRU architectures were explored (no transformers)
- B) The forecast horizon (H=24) was fixed
- C) The model was too accurate for practical use
- D) TimeGAN synthetic realism was insufficient
> **Answer: C** -- All others are explicitly listed as limitations. "Too accurate" is not a limitation.

**Q6.** What future work direction would most directly address the TimeGAN limitation?
- A) More random seeds for robustness testing
- B) Longer adversarial training, larger latent dimensions, or diffusion-based generators
- C) Adding more input features
- D) Using a different dataset
> **Answer: B** -- The TimeGAN's main issues were short training (15 epochs) and limited latent capacity (dim=24). Diffusion models are a newer, more stable alternative to GANs.

**Q7.** What is the "optimize -> explain -> prune -> validate" loop?
- A) A data preprocessing pipeline
- B) EA finds a good model, XAI reveals feature importance, pruning removes weak features, multi-seed testing confirms the improvement is robust
- C) A method for generating synthetic data
- D) The TimeGAN training procedure
> **Answer: B** -- This closed loop is the project's key methodological contribution. XAI is used not just for interpretation but for active model refinement.

**Q8.** Why is the final model described as "more accurate, more compact, more interpretable, AND more efficient"?
- A) These properties always go together
- B) The baseline was over-parameterized, so the EA and pruning found a model that improves on all fronts simultaneously
- C) The final model uses a completely different architecture
- D) Efficiency metrics were not actually measured
> **Answer: B** -- When starting from an over-parameterized baseline, optimization can improve accuracy and efficiency at the same time. The extra baseline complexity was hurting, not helping.

**Q9.** What was the best single-seed MAE observed in the entire project?
- A) 1.650 degC (GRU Baseline)
- B) 1.598 degC (Best EA GRU)
- C) ~1.569 degC (Pruned EA GRU, seed 7)
- D) 3.144 degC (Persistence)
> **Answer: C** -- The Pruned EA GRU with seed 7 achieved approximately 1.569 degC MAE, the best individual result in the project.

**Q10.** If you extended this project with one experiment, which would likely have the most impact?
- A) Adding a 4th GRU layer to the pruned model
- B) Expanding the search to include transformer or attention-based architectures
- C) Removing the persistence baseline
- D) Using MinMaxScaler instead of RobustScaler
> **Answer: B** -- Transformers handle long-range dependencies differently than GRUs and might capture patterns the GRU cannot. This is the most impactful limitation to address.
