# xgboost model -----------------------------------------------------------
# setup -------------------------------------------------------------------
source("R/functions/xgboosting_functions.R")
source("R/functions/data_functions.R")
library(xgboost)
library(dplyr)
library(ggplot2)
set.seed(42)

# data --------------------------------------------------------------------
df <- read.csv("data_for_dominik.csv") %>%
  mutate(player_id = as.numeric(player_id)) %>%
  mutate(row_index = row_number()) %>%
  dplyr::mutate(
    w_h = 1 / (time_gap_1 + 1),
    w_sr = 1 / sqrt(time_gap_1 + 1),
    w_log = 1 / (time_gap_1 + exp(1))
  )

main_output_path <- "R/R_output_files/"

xgboost_grid_results_path <- "xgboost_grid_results"

data_check_params <- list(
  UNIQUE_IDENTIFIERS = c("player_id", "match_id")
)

# checkDuplicateRows(df, params = data_check_params)


xg_boost_setup_folder <- file.path("R/xgboost_setup_folder")


variable_files_vec <- list.files(xg_boost_setup_folder)


grid_list <- list(
  n_rounds = c(2000),
  max_depth = c(6),
  eta = c(0.1, 0.3),
  gamma = 0,
  colsample_bytree = c(0.5, 0.7, 0.8),
  lambda = c(0, 0.5, 1), # Adding L2 regularization
  alpha = c(1, 0.5, 0) # Adding L1 regularization
)

fixed_xgb_cv_options_list <- list(
  BOOSTER = "gbtree",
  OBJECTIVE = "reg:squarederror",
  EVAL_METRIC = "rmse",
  NFOLD = 5,
  EARLY_STOPPING_ROUNDS = 4,
  VERBOSE = TRUE
)


xgboost_grid_results_output_folder <- file.path(
  main_output_path,
  xgboost_grid_results_path
)

if (!dir.exists(xgboost_grid_results_output_folder)){
  dir.create(xgboost_grid_results_output_folder)
}


for (j in seq_along(variable_files_vec)) {
  variables_path <- file.path(xg_boost_setup_folder, variable_files_vec[j])

  variables <- yaml::read_yaml(file = variables_path)

  dependent_var <- variables$DEPENDENT_VARIABLE

  covariates_vec <- variables$COVARIATES

  train_test_params <- list(
    DEPENDENT_VARIABLE = "rating",
    COVARIATE_VECTOR = covariates_vec,
    TRAINING_SAMPLE_FRACTION = 0.7,
    INDEX_COLUMN = "row_index"
  )


  train_test_sample_list <- makeTrainTestSample(
    data = df,
    params = train_test_params
  )

  xgb_train <- xgboost::xgb.DMatrix(
    data = train_test_sample_list$TRAIN_X,
    label = train_test_sample_list$TRAIN_Y
  )

  xgb_test <- xgboost::xgb.DMatrix(
    data = train_test_sample_list$TEST_X,
    label = train_test_sample_list$TEST_Y
  )

  spec_name <- stringr::str_remove(variable_files_vec[j], ".yaml")

  search_grid <- expand.grid(
    grid_list
  )

  best_train_rmse <- Inf
  best_params <- list()
  best_nrounds <- 0
  best_test_rmse <- Inf

  for (i in 1:nrow(search_grid)) {
    params <- list(
      booster = fixed_xgb_cv_options_list$BOOSTER,
      objective = fixed_xgb_cv_options_list$OBJECTIVE,
      eval_metric = fixed_xgb_cv_options_list$EVAL_METRIC,
      max_depth = search_grid$max_depth[i],
      eta = search_grid$eta[i],
      colsample_bytree = search_grid$colsample_bytree[i],
      lambda = search_grid$lambda[i],
      alpha = search_grid$alpha[i]
    )

    cv_results <- xgb.cv(
      params = params,
      data = xgb_train,
      nrounds = search_grid$n_rounds[i],
      nfold = fixed_xgb_cv_options_list$NFOLD,
      early_stopping_rounds = fixed_xgb_cv_options_list$EARLY_STOPPING_ROUNDS,
      verbose = fixed_xgb_cv_options_list$VERBOSE
    )

    min_test_rmse <- min(cv_results$evaluation_log$test_rmse_mean)
    min_train_rmse <- min(cv_results$evaluation_log$train_rmse_mean)

    if (min_test_rmse < best_test_rmse) {
      best_test_rmse <- min_test_rmse
      best_train_rmse <- min_train_rmse
      best_params <- params
      best_nrounds <- cv_results$best_iteration
    }
  }

  xgboost_grid_results_output_path <- file.path(xgboost_grid_results_output_folder, spec_name)

  save_grid_params <- list(
    FOLDER_PATH = file.path(xgboost_grid_results_output_path),
    FILE_NAME = "grid_results.yaml",
    INITIAL_GRID = search_grid
  )

  # STORE SETUP AND RESULTS
  saveGridSearchResults(
    data = best_params,
    params = save_grid_params
  )

  yaml::write_yaml(grid_list,
    file = file.path(
      xgboost_output_path,
      "grid_setup_list.yaml"
    )
  )

  yaml::write_yaml(
    fixed_xgb_cv_options_list,
    file = file.path(
      xgboost_output_path,
      "fixed_xgb_cv_params.yaml"
    ),
  )

  rmse_list <- list(
    BEST_TRAIN_RMSE = best_train_rmse,
    BEST_TEST_RMSE = best_test_rmse,
    BEST_N_ROUNDS = best_nrounds
  )


  yaml::write_yaml(
    rmse_list,
    file = file.path(
      xgboost_output_path,
      "RMSE_info.yaml"
    )
  )

  final_model <- xgb.train(
    params = best_params, # best parameters from the grid search
    data = xgb_train, # training data
    nrounds = best_nrounds, # Best number of rounds from CV
    watchlist = list(train = xgb_train, test = xgb_test),
    early_stopping_rounds = 10,
    verbose = 1
  )

  model_output_folder <- file.path(
    main_output_path,
    "xgboost_model_objects"
  )

  if (!dir.exists(model_output_folder)){
    dir.create(model_output_folder)
  }

  save_model_params <- list(
    MODEL_OUTPUT_FOLDER = model_output_folder,
    MODEL_FILE_NAME = paste0("final_xgboost_model_", spec_name, "_", Sys.Date(), ".rds")
  )

  saveModel(final_model, params = save_model_params)
}




# Eval --------------------------------------------------------------------

# Train the final model on the entire training data with the best parameters


# Make predictions on the test set
preds <- predict(final_model, xgb_test)

# Calculate RMSE on the test set
test_rmse <- sqrt(mean((train_test_sample_list$TEST_Y - preds)^2))
print(paste("Test RMSE: ", test_rmse))

importance_matrix <- xgb.importance(model = final_model)
xgb.plot.importance(importance_matrix)




df <- train_test_sample_list$TEST_Y %>%
  as.data.frame("rating") %>%
  dplyr::rename(rating = V1) %>%
  mutate(
    predicted_rating = preds, # Add predicted ratings to the dataframe
    average_rating = mean(rating),
    row_index = dplyr::row_number(),
    residual = abs(rating - predicted_rating), # Calculate absolute error (residual)
    quartile = ntile(rating, 4)
  ) # Add the average rating as a constant

# Step 2: Create the plot
ggplot(df, aes(x = row_index)) + # Assuming row_index is an index column in the dataset
  geom_point(aes(y = rating, color = "Actual Rating"), size = 1, alpha = 0.5) + # Actual rating
  geom_point(aes(y = predicted_rating, color = "Fitted Rating"), linewidth = 1, linetype = "dashed", alpha = 0.5) + # Fitted rating
  geom_hline(aes(yintercept = mean(rating), color = "Average Rating"), linetype = "dotted", size = 1) + # Average rating
  labs(
    title = "Actual Rating, Fitted Rating, and Average Rating",
    x = "Row Index",
    y = "Rating"
  ) +
  scale_color_manual(
    name = "Legend",
    values = c(
      "Actual Rating" = "blue",
      "Fitted Rating" = "red",
      "Average Rating" = "green"
    )
  ) +
  theme_minimal()


hist(log(df$rating))


quartile_performance <- df %>%
  group_by(quartile) %>%
  summarise(
    actual_mean = mean(rating),
    predicted_mean = mean(predicted_rating),
    rmse = sqrt(mean((rating - predicted_rating)^2)),
    residual_mean = mean(residual)
  )

print(quartile_performance)

# RMSE for each quartile
ggplot(quartile_performance, aes(x = quartile, y = rmse)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(
    title = "RMSE for each Quartile of Rating",
    x = "Quartile",
    y = "RMSE"
  ) +
  theme_minimal()
