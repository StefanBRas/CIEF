library("dplyr")
output_path <- "data/outputs"
parquet_info_path <- "data/outputs/full_info.parquet"
parquet_result_path <- "data/outputs/results.parquet"
fig_output_path  <- "../thesis/generated_figs/"
auc_table_output_path <- paste0(fig_output_path, "auc_table.png")

format_g  <- function(x) {
    dplyr::case_when(
		     x == "chain" ~ "Chain",
		     x == "tree" ~ "Tree",
		     x == "tree_mirrored" ~ "Tree Mirrored",
		     TRUE ~ as.character(x)
		     )}


get_exp_names <- function() {
    files  <- list.files(output_path)
    files   <- files[startsWith(files, "keepdata_True") & endsWith(files, "fit_info.parquet") & grepl("each_step", files)]
    unlist(lapply(strsplit(files, "_fit_info.parquet"), \(x) x[[1]]))
}

get_exp_paths <- function() {
    names   <- get_exp_names()
    paste0(output_path,"/", names, "_fit_info.parquet")
}

clean_common  <- function(data) {
    edges_split  <- strsplit(data$edge, "-")
    edges_different <- unlist(lapply(edges_split, function(x) x[[1]] != x[[2]]))
    data_cleaned <- data[edges_different, ]
    data_cleaned["conditionally_independent"]  <- !data_cleaned["connected_precision"]
    data_cleaned   <- data_cleaned %>% mutate(expected_value = ifelse(conditionally_independent, 0, 1))
    return(data_cleaned)
}

clean_trace  <- function(data) {
    data_cleaned <- clean_common(data)
    data_cleaned   <- data_cleaned %>%
	group_by(experiment_name, iteration_index)  %>%
	mutate(normalized_value = value / max(value))
    return(data_cleaned)
}

clean_result  <- function(data) {
    data_cleaned <- clean_common(data)
    data_cleaned   <- data_cleaned %>%
	group_by(experiment_name, estimator)  %>%
	mutate(normalized_value = value / max(value))
    return(data_cleaned)
}


enhance_graph_data  <- function(data) {
    get_abs_diff  <- function(x) {
	tmp  <- as.integer(x)
	return(abs(tmp[1] - tmp[2]))
    }
    data$distance  <- unlist(lapply(strsplit(data$edge, "-"), get_abs_diff))
    data$edge_type  <- as.character(data$distance)
    return(data)
}


enhance_tree_data  <- function(data) {
    data  <- data  %>% mutate(
	      edge_1 = gsub("-\\d*", "", edge),
	      edge_2 = gsub("\\d*-", "", edge)
	      )  %>%
    mutate(edge_type = ifelse(as.integer(edge_1) == 0,"root-leaf", "leaf-leaf"))  %>% 
    select(-c(edge_1, edge_2))
    return(data)
}

enhance_mirrored_tree_data  <- function(data) {
    get_edge_type  <- function(edge_1, edge_2, dim) {
	k  <- floor(dim / 2)
	e_type <- function(x) ifelse(x < k, "g", ifelse(x == k, "m", "s"))
	edge_1_t  <- e_type(as.integer(edge_1))
	edge_2_t  <- e_type(as.integer(edge_2))
	paste(edge_1_t, edge_2_t, sep = "-")
    }
    data  <- data  %>% mutate(
	      edge_1 = gsub("-\\d*", "", edge),
	      edge_2 = gsub("\\d*-", "", edge)
	      )  %>%
    mutate(edge_type = get_edge_type(edge_1, edge_2, data_columns))
    # %>% select(-c(edge_1, edge_2))
    return(data)
}

filter_experiment  <- function(data, name) {
    exp_data  <- filter(data, experiment_name == name)
    graph_type  <- exp_data$graph_type[1]
    if (graph_type == "chain") exp_data  <- enhance_graph_data(exp_data)
    if (graph_type == "tree") exp_data  <- enhance_tree_data(exp_data)
    if (graph_type == "tree_mirrored") exp_data  <- enhance_mirrored_tree_data(exp_data)
    return(exp_data)
}


help_load_data  <- function(test) {
    data <- arrow::read_parquet(parquet_info_path, as_data_frame = TRUE)  %>% 
	filter(test == is_test(experiment_name))
    data  <- clean_trace(data)
    return(data)
}

load_data  <- function(path) {
    data  <- help_load_data(F)
    return(data)
}

load_exp  <- function(path) {
    data <- arrow::read_parquet(path, as_data_frame = TRUE)  %>% clean_trace()
    return(data)
}


load_test_data  <- function(path) {
    data  <- help_load_data(T)
    return(data)
}

is_test  <- function(name) startsWith(name, "test")


help_load_results  <- function(test) {
    data <- arrow::read_parquet(parquet_result_path, as_data_frame = TRUE)  %>% 
	filter(test == is_test(experiment_name))
    # print(filter(data, startsWith(experiment_name, "test")))
    # print(filter(data, is_test == startsWith(experiment_name, "test")))
    data  <- clean_result(data)
    return(data)
}

load_results  <- function() {
    return(help_load_results(FALSE))
}

load_test_results  <- function() {
    return(help_load_results(TRUE))
}


auc <- function(expected, predicted) {
    result <- pROC::auc(expected, predicted)
    attributes(result)  <- NULL
    return(result)
}

cief_summary  <- function(df) {
	summarise(df,
		  mean_error = mean(abs(expected_value - normalized_value)),
		  auc = auc(expected_value, normalized_value),
		  duration = mean(fit_duration),
		  rows = first(data_rows),
		  columns = first(data_columns),
		  w_update_frequency = first(w_update_frequency),
		  graph_type = first(graph_type),

		  #iterations = first(max_iterations)
	) 
}

data_summary <- function(df) {
    result  <- df  %>%
	group_by(experiment_name, iteration_index)  %>%
	cief_summary()  %>% 
	mutate(estimator = "CIEF")
	return(result)
}


result_summary <- function(df) {
    result  <- df  %>%
	group_by(experiment_name, estimator)  %>%
	cief_summary()
	return(result)
}


