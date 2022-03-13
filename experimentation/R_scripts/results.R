source("R_scripts/common.R")
library("ggplot2")
library("gt")
library("tidyr")
library(purrr)
library(patchwork)

### funcs
roc_plot <- function(experiment_results) {
    rocs <- experiment_results %>%
	group_by(estimator) %>%
	group_map(~ pROC::roc(.x$expected_value, .x$normalized_value), .keep = T) %>%
	setNames(unique(sort(results$estimator)))
    roc_plot <- pROC::ggroc(rocs) +
	geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "grey", linetype = "dashed")
    return(roc_plot)
}


create_roc_plot <- function(experiment_results) {
    r_plot <- roc_plot(experiment_results)
    graph_type <- unique(experiment_results$graph_type[1])
    output_path <- paste0(fig_output_path, graph_type, ".pdf")
    ggsave(plot = r_plot, filename = output_path)
}

get_edge_acc_by_edge_type <- function(data, experiment_name, threshold) {
    results <- data %>%
	filter_experiment(experiment_name) %>%
	filter(estimator == "CIEF") %>%
	group_by(edge_type, graph_type) %>%
	mutate(prediction = ifelse(normalized_value < threshold, 0, 1)) %>%
	mutate(correct = prediction == expected_value) %>%
	summarise(
		  n = n(),
		  acc = sum(correct) / n(),
		  expected = first(expected_value),
		  graph_type = first(graph_type),
		  threshold = threshold
	)
	return(results)
}

get_edge_acc <- function(data, experiment_name, threshold) {
    results <- data %>%
	filter_experiment(experiment_name) %>%
	mutate(prediction = ifelse(normalized_value < threshold, 0, 1)) %>%
	mutate(correct = prediction == expected_value) %>%
	summarise(
		  n = n(),
		  acc = sum(correct) / n(),
		  expected = first(expected_value),
		  graph_type = first(graph_type),
		  threshold = threshold
	)
	return(results)
}


get_edge_acc_by_edge_type_df <- function(results) {
    threshold_grid <- expand.grid(
				  experiment = unique(results$experiment_name),
				  threshold = c(0.1, 0.25, 0.5, 0.75, 0.90)
    )
    threshold_grid %>%
	split(1:nrow(.)) %>%
	purrr::map_dfr(function(x) get_edge_acc_by_edge_type(results, x$experiment, x$threshold))
}


### Summary table

graph_tab_spanner <- function(data, graph_type, label) {
    tab_spanner(data,
		label = label,
		columns = ends_with(graph_type)
    )
}

auc_md <- md("AUC")
r_auc_md <- md("Relative<br>to best")

bin_col <- scales::col_bin(c("red", "green"), domain = c(-1, 0), bins = 2)

bin_col <- function(x) {
    res <- ifelse(x < 0, "red", "green")
    return(res)
}

auc_table <- function(res_summary) {
    res_summary %>%
	group_by(graph_type) %>%
	mutate(
	       relative_to_best = auc - max(auc)
	       ) %>%
	ungroup() %>%
	select(auc, relative_to_best, graph_type, estimator) %>%
	pivot_wider(names_from = graph_type, values_from = c(auc, relative_to_best)) %>%
	gt(rowname_col = "estimator") %>%
	tab_header(
		   title = "AUC of predictions by experiment and estimator",
		   subtitle = "Area under the Receiver Operator Curve."
		   ) %>%
	cols_align(
		   align = "left",
		   ) %>%
	tab_style(
		  style = list(
			       cell_text(align = "left")
			       ),
		  locations = cells_stub(rows = TRUE)
		  ) %>%
	graph_tab_spanner("chain", "Chain") %>%
	graph_tab_spanner("tree", "Tree") %>%
	graph_tab_spanner("tree_mirrored", "Tree Mirrored") %>%
	fmt_number(
		   columns = everything(),
		   rows = everything(),
		   decimals = 3
		   ) %>%
	cols_label(
		   auc_chain = auc_md,
		   auc_tree = auc_md,
		   auc_tree_mirrored = auc_md,
		   relative_to_best_chain = "Difference",
		   relative_to_best_tree = "Difference",
		   relative_to_best_tree_mirrored = "Difference"
		   ) %>%
	data_color(
		   columns = starts_with("relative"),
		   colors = bin_col,
	)
}


### Prediction table

edge_rename <- function(x) {
    dplyr::case_when(
		     x == "g-g" ~ "Ancestor-Ancestor",
		     x == "g-m" ~ "Ancestor-Middle",
		     x == "g-s" ~ "Ancestor-Descendent",
		     x == "m-s" ~ "Middle-Descendent",
		     x == "s-s" ~ "Descendent-Descendent",
		     x == "leaf-leaf" ~ "Leaf-Leaf",
		     x == "root-leaf" ~ "Root-Leaf",
		     TRUE ~ as.character(x)
    )
}



get_edge_table <- function(edge_df, graph) {
    df <- edge_df %>%
	pivot_wider(names_from = threshold, values_from = acc) %>%
	ungroup()
    if (graph == "chain") {
	df <- arrange(df, as.integer(edge_type))
	row_label <- "Distance"
    } else {
	row_label <- "Edge Type"
    }
    gt(df, rowname_col = "edge_type") %>%
	tab_stubhead(label = md(row_label)) %>%
	cols_hide(graph_type) %>%
	tab_spanner(
		    label = "Threshold",
		    columns = c(
				`0.25`,
				`0.5`,
				`0.75`
		    )
		    ) %>%
	fmt_number(
		   columns = starts_with("0"),
		   decimals = 3
		   ) %>%
	tab_header(
		   title = "Predictions Accuaracy under threshold",
		   subtitle = "TODO"
		   ) %>%
	cols_label(
		   expected = md("Conditionally<br>Independent?"),
		   ) %>%
	data_color(
		   columns = starts_with("0"),
		   colors = scales::col_numeric(
						palette = c(
							    "red", "green"
							    ),
						domain = c(0, 1)
						),
		   ) %>%
	text_transform(
		       locations = cells_stub(rows = TRUE),
		       fn = edge_rename
		       ) %>%
	tab_style(
		  style = list(
			       cell_text(align = "left")
			       ),
		  locations = cells_stub(rows = TRUE)
	)
}

create_edge_table <- function(edge_df) {
    graph_type <- unique(edge_df$graph_type[1])
    table <- get_edge_table(edge_df, graph_type)
    output_path <- paste0(fig_output_path, graph_type, "_prediction_table.png")
    gtsave(table, output_path)
}

get_edge_table_full <- function(edge_df) {
    df <- edge_df %>%
	pivot_wider(names_from = threshold, values_from = acc) %>%
	arrange(as.integer(edge_type))  %>% 
	ungroup()  %>% 
    gt(rowname_col = "edge_type") %>%
	tab_stubhead(label = md("Edge Type")) %>%
	tab_spanner(
		    label = "Threshold",
		    columns = c(
				`0.1`,
				`0.25`,
				`0.5`,
				`0.75`,
				`0.9`
		    )
		    ) %>%
	fmt_number(
		   columns = starts_with("0"),
		   decimals = 3
		   ) %>%
	tab_header(
		   title = "Prediction Accuracy of CIEF under different thresholds",
		   subtitle = "Arranged by experiment and edge type"
		   ) %>%
	cols_label(
		   expected = md("Conditionally<br>Independent?"),
		   ) %>%
	data_color(
		   columns = starts_with("0"),
		   colors = scales::col_numeric(
						palette = c(
							    "red", "green"
							    ),
						domain = c(0, 1)
						),
		   ) %>%
	text_transform(
		       locations = cells_stub(rows = TRUE),
		       fn = edge_rename
		       ) %>%
	text_transform(
		       locations = cells_body(
					      columns = expected 
					      ),
		       fn = function(x) {ifelse(x == 0, "Yes", "No")}
		       ) %>%
	tab_style(
		  style = list(
			       cell_text(align = "left")
			       ),
		  locations = cells_stub(rows = TRUE)
	)  %>% tab_row_group(
		label = md("**Chain**"),
		rows = graph_type == "chain" 
    ) %>% tab_row_group(
		label = md("**Tree**"),
		rows = graph_type == "tree" 
    ) %>% tab_row_group(
		label = md("**Tree Mirrored**"),
		rows = graph_type == "tree_mirrored" 
    ) %>% 
	cols_hide(graph_type)
}


create_edge_table_full <- function(edge_df) {
    table <- get_edge_table_full(edge_df)
    output_path <- paste0(fig_output_path, "prediction_table.png")
    gtsave(table, output_path)
}


get_coords_for_ggplot <- function(roc) {
	df <- pROC::coords(roc, "all", transpose = FALSE)
	return(df[rev(seq(nrow(df))),])
}



create_roc_plots <- function(results) {
    roc_plots <- results %>%
	group_by(estimator, graph_type) %>%
	group_modify(~ get_coords_for_ggplot(pROC::roc(.x$expected_value, .x$normalized_value)), .keep = T)  %>% 
	ggplot(aes(x = specificity, y = sensitivity, color=estimator)) + 
	geom_line() + 
	ggplot2::scale_x_reverse(lim=c(1, 0)) +
	facet_grid(.~graph_type,
labeller = as_labeller(format_g)
		   ) + 
	geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "grey", linetype = "dashed") + 
	labs(
	     title =  "Receiver Operating Characteristic Curve",
	     caption = waiver(),
	     tag = waiver(),
	     alt = waiver(),
	     alt_insight = waiver()
	     ) + theme(legend.position = "bottom") + 
				     scale_colour_discrete(name = "Estimator")


	ggsave(roc_plots, filename = paste0(fig_output_path, "roc_curves.pdf"), width=8,
	       height = 6
	)
}

