source("R_scripts/results.R")
source("R_scripts/traces.R")
library("latex2exp")
library("glue")
## Results

results <- load_results() %>% filter(
				     startsWith(experiment_name, "keepdata_True"),
				     w_update_frequency == "each_step"
)
real_experiment_names <- unique(results$experiment_name)

res_summary <- results %>%
    result_summary() %>%
    ungroup() %>%
    select(-c(experiment_name))


### Summary table

auc_table(res_summary)  %>% 
    gtsave(auc_table_output_path)

### Prediction table
edge_df  <- get_edge_acc_by_edge_type_df(results)

create_edge_table_full(edge_df)

### ROC plots

create_roc_plots(results)

### trace plots 
exp_names  <- get_exp_names()
exp_paths  <- get_exp_paths()
for (i in seq_along(exp_names)) {
    generate_plot_for_full_trace(exp_names[i], exp_paths[i])
}

results  %>% select(fit_duration)  %>% group_by(experiment_name)  %>% 
    summarise(duration_in_mins = mean(fit_duration) / 60, duration_per_iteration = mean(fit_duration) / 500)
