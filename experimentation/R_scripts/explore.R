source("R_scripts/common.R")
library("ggplot2")
data <- load_data("all")
data  %>% colnames()
unique(data$experiment_name)

data %>% group_by(experiment_name)  %>% summarise(duration = mean(fit_duration))

each_step  <- data  %>% filter(experiment_name == "rows_100_data_chain_frequency_each_step")
each_step_large  <- data  %>% filter(experiment_name == "rows_500_data_chain_frequency_each_step")

each_target  <- data  %>% filter(experiment_name == "rows_100_data_chain_frequency_each_target")
mod  <-   lm(fit_duration ~ data_rows + w_update_frequency, data=data)
mod

NROW(each_step)
NROW(each_step_large)
NROW(each_target)
colnames(data)


each_step_large  <- enhance_graph_data(each_step_large)
max(each_step_large$depth)

make_plot  <- function(data) {
    ggplot(data,
	   aes(x = iteration_index,
	       y = value,
	       group=edge,
	       color=conditionally_independent,
	       )) + geom_line()
}
make_plot(each_step)
make_plot(each_step_large)
make_plot(each_target)

### results

results <- load_results()
colnames(results)
results  %>% group_by(experiment_name) %>% summarize(duration = mean(fit_duration))
unique(results$edge)
results   <- results %>% group_by(experiment_name)  %>% mutate(normalized_value = value / max(value))
results

results  %>% group_by(experiment_name, conditionally_independent)  %>% summarise(mean_error = mean(abs(expected_value - normalized_value))) 


experiment_summary <- function(df) {
    result  <- df  %>%
	group_by(experiment_name)  %>%
	summarise(
		  mean_error = mean(abs(expected_value - normalized_value)),
		  auc = pROC::auc(expected_value, normalized_value)
	) 
    return(result)
}
results  %>% experiment_summary()

auc <- function(expected, predicted) {
    result <- pROC::auc(expected, predicted)
    attributes(result)  <- NULL
    return(result)
}

experiment_summary <- function(df) {
    result  <- df  %>%
	group_by(experiment_name)  %>%
	summarise(
		  mean_error = mean(abs(expected_value - normalized_value)),
		  auc = auc(expected_value, normalized_value)
	) 
    return(result)
}

dname  <- unique(results$experiment_name)[[1]]
table(results$experiment_name,results$expected_value)
results %>% experiment_summary()




