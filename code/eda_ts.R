# Simple TS Analysis in R: facet plot, seasonal decomposition, and ACF/PACF

library(ggplot2)
library(forecast)
library(dplyr)
library(tidyr)
library(gridExtra)

data_path <- "data/processed/combined_daily.csv"
outdir <- "outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Load data
df <- read.csv(data_path)
df$date <- as.Date(df$date)

targets <- c("NO2_mean_ppb", "PM25_mean_ugm3", "CO_mean_ppm")

# 1. Facet plot
df_long <- df %>%
  pivot_longer(cols = all_of(targets), names_to = "variable", values_to = "value")

p_facet <- ggplot(df_long, aes(x = date, y = value)) +
  geom_line(color = "#0072B2", linewidth = 0.8) +
  facet_wrap(~ variable, ncol = 1, scales = "free_y") +
  theme_minimal(base_size = 12) +
  labs(title = "Daily Pollutant Levels (Los Angeles County)",
       x = "Date", y = "Value")

ggsave(file.path(outdir, "facet_aqi_pollutants.png"), p_facet, width = 9, height = 6, dpi = 300)
cat("[SAVE] outputs/facet_aqi_pollutants.png\n")

# 2. Decomposition + ACF/PACF
for (col in targets) {
  series <- ts(df[[col]], start = c(as.numeric(format(min(df$date), "%Y")),
                                    as.numeric(format(min(df$date), "%j"))),
               frequency = 365)

  decomp <- stl(na.interp(series), s.window = "periodic")
  p_decomp <- autoplot(decomp) +
    ggtitle(paste("Seasonal Decomposition of", col)) +
    theme_minimal(base_size = 12)

  ggsave(file.path(outdir, paste0("decompose_", col, ".png")),
         p_decomp, width = 9, height = 6, dpi = 300)
  cat("[SAVE]", paste0("outputs/decompose_", col, ".png\n"))


  p_acf <- ggAcf(series, lag.max = 60) + ggtitle(paste("ACF of", col))
  p_pacf <- ggPacf(series, lag.max = 60) + ggtitle(paste("PACF of", col))
  g_comb <- grid.arrange(p_acf, p_pacf, ncol = 2)

  ggsave(file.path(outdir, paste0("acf_pacf_", col, ".png")),
         g_comb, width = 10, height = 4, dpi = 300)
  cat("[SAVE]", paste0("outputs/acf_pacf_", col, ".png\n"))
}