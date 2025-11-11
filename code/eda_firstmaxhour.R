# First Max Hour Analysis in R: Histogram, Density plot, Month trend

library(ggplot2)
library(dplyr)

data_path <- "data/processed/combined_daily.csv"
outdir <- "outputs"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Load data
aqi <- read.csv(data_path)
aqi$date <- as.Date(aqi$date)

# Distribution histogram + density
p_hour <- ggplot(aqi, aes(x = NO2_first_max_hour_mode)) +
  geom_histogram(aes(y = ..density..), bins = 24,
                 fill = "#56B4E9", color = "white", alpha = 0.7) +
  geom_density(color = "#E69F00", linewidth = 1) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Distribution of NO₂ First Max Hour (Los Angeles County)",
    x = "Hour of Day", y = "Density"
  )

ggsave(file.path(outdir, "first_max_hour_distribution.png"),
       p_hour, width = 7, height = 4, dpi = 300)
cat("[SAVE] outputs/first_max_hour_distribution.png\n")


# Monthly pattern
month_mode <- aqi %>%
  mutate(month = format(date, "%b")) %>%
  group_by(month) %>%
  summarise(mode_hour = {
    x <- na.omit(NO2_first_max_hour_mode)
    if (length(x) == 0) NA else as.numeric(names(sort(table(x), decreasing = TRUE)[1]))
  }, .groups = "drop") %>%
  mutate(month = factor(month, levels = month.abb))

p_month_mode <- ggplot(month_mode, aes(x = month, y = mode_hour, group = 1)) +
  geom_line(color = "#0072B2", linewidth = 1) +
  geom_point(size = 2, color = "#E69F00") +
  geom_text(aes(label = round(mode_hour, 1)), vjust = -0.8, size = 3)
  theme_minimal(base_size = 12) +
  labs(
    title = "Monthly Mode of NO\u2082 First Max Hour",
    x = "Month", y = "Mode of First Max Hour (0–23)"
  )

ggsave(file.path(outdir, "first_max_hour_month_mode.png"),
       p_month_mode, width = 7, height = 4, dpi = 300)
cat("[SAVE] outputs/first_max_hour_month_mode.png\n")