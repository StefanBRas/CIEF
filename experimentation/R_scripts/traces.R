source("R_scripts/common.R")
library("ggplot2")
library("patchwork")

make_auc_plot  <- function(res_summary, exp_data_summary) {
    aucs <- res_summary  %>% 
	select(auc, estimator)  %>% 
	filter(estimator != "CIEF")
    plot   <- exp_data_summary  %>% ggplot(aes(x = iteration_index, y = auc, color = estimator)) +
	geom_line() + 
	geom_hline(data = aucs, aes(yintercept = auc, color =estimator))
    return(plot)
}

save_auc_plot  <- function(res_summary, exp_data_summary) {
    a_plot  <- make_auc_plot(res_summary, exp_data_summary)
    graph_type <- unique(res_summary$graph_type[1])
    output_path <- paste0(fig_output_path, graph_type, "_auc_trace.pdf")
    ggsave(plot = a_plot, filename = output_path)
}

make_trace_plot  <- function(data) {
    ggplot(data,
	   aes(x = iteration_index,
	       y = value,
	       group=edge,
	       color=conditionally_independent,
	       )) + geom_line()
}

save_trace_plot <- function(trace_data) {
    t_plot <- make_trace_plot(trace_data)
    graph_type <- unique(trace_data$graph_type[1])
    output_path <- paste0(fig_output_path, graph_type, "_w_delta_trace.pdf")
    ggsave(plot = t_plot, filename = output_path)
}

make_full_trace_plot  <- function(res_summary, exp_data_summary, trace_data) {
    trace_plot  <- make_trace_plot(trace_data)
    auc_plot  <-   make_auc_plot(res_summary, exp_data_summary)
    return(trace_plot / auc_plot)
}

save_full_trace_plot  <- function(res_summary, exp_data_summary, trace_data) {
    f_plot  <- make_full_trace_plot(res_summary, exp_data_summary, trace_data)
    graph_type <- unique(trace_data$graph_type[1])
    output_path <- paste0(fig_output_path, graph_type, "_full_trace.pdf")
    ggsave(plot = f_plot, filename = output_path)
}

generate_plot_for_trace <- function(exp_name, exp_path) {
    trace_data <- load_exp(exp_path)
    data_summarized  <-  trace_data  %>% data_summary()  %>% ungroup()
    results  <- load_results()  %>% ungroup()
    results_summarized  <- results  %>% result_summary()  %>% filter(experiment_name == exp_name)
    save_full_trace_plot(results_summarized, data_summarized, trace_data)
}


############## TRACE PLOTS NEW

make_auc_plot  <- function(res_summary, exp_data_summary) {
    g_type  <- format_g(unique(exp_data_summary$graph_type))
    aucs <- res_summary  %>% 
	select(auc, estimator)  %>% 
	filter(estimator != "CIEF")
    plot   <- exp_data_summary  %>% ggplot(aes(x = iteration_index, y = auc, color = estimator)) +
	geom_line() + 
	geom_hline(data = aucs, aes(yintercept = auc, color = estimator)) +
	labs(
	     title =  glue("AUC during training of CIEF on {g_type} data"),
	     caption = waiver(),
	     tag = waiver(),
	     alt = waiver(),
	     alt_insight = waiver()
	     ) +
				     xlab(TeX(r"($\textbf{W}_\Delta$-update index)")) +
				     ylab("AUC") +
				     theme(legend.position = c(0.88, .40)) +
				     scale_colour_discrete(name = "Estimator")
    return(plot)
}

make_trace_plot  <- function(data) {
    g_type  <- format_g(unique(data$graph_type))
    end_title  <-  glue("for training CIEF on {g_type} data")
    p_title  <- TeX(paste0(r"($\textbf{W}_{ij}$ estimates at each $\textbf{W}_\Delta$-update)",
			   end_title))
    ggplot(data,
	   aes(x = iteration_index,
	       y = value,
	       group=edge,
	       color=conditionally_independent,
	       )) + geom_line(aes(alpha=factor(conditionally_independent))) + 
				     labs(
					  title =  p_title,
					  subtitle = "Each line is the accumulated score of an edge",
					  tag = waiver(),
					  alt = waiver(),
					  alt_insight = waiver()
					  ) +
				     xlab(TeX(r"($\textbf{W}_\Delta$-update index)")) +
				     ylab(TeX("Accumulated score")) +
				     theme(legend.position = c(.14, .88)) +
				     scale_colour_discrete(name = "Conditionally Independent?"
							   , labels = c("No", "Yes")) +
				     scale_alpha_manual(values = c(0.99, 0.3), guide = "none")
}

make_full_trace_plot  <- function(res_summary, exp_data_summary, trace_data) {
    trace_plot  <- make_trace_plot(trace_data)
    auc_plot  <-   make_auc_plot(res_summary, exp_data_summary)
    full_plot  <- trace_plot / auc_plot + plot_layout(height = c(3, 1))
    return(full_plot)
}

generate_plot_for_full_trace <- function(exp_name, exp_path) {
    trace_data <- load_exp(exp_path)
    data_summarized  <-  trace_data  %>% data_summary()  %>% ungroup()
    results  <- load_results()  %>% ungroup()
    results_summarized  <- results  %>% result_summary()  %>% filter(experiment_name == exp_name)
    f_plot  <- make_full_trace_plot(results_summarized, data_summarized, trace_data)
    graph_type <- unique(trace_data$graph_type[1])
    output_path <- paste0(fig_output_path, graph_type, "_full_trace.pdf")
    ggsave(filename = output_path, dpi = 1200, device = cairo_pdf )
}

