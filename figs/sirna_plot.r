library(tidyverse)
library(ggplot2)
setwd("~/Dropbox/Documents/tri-i/FoDS/project/analysis")

df = read_csv("sirna_functions.csv") %>%
  replace_na(list(parent = "NA")) 

df = df %>%
  group_by(parent) %>%
  summarise(`siRNAs targeting` = n()) %>%
  arrange(-`siRNAs targeting`) %>%
  mutate(parent = factor(parent, levels = parent)) %>%
  mutate(n = if_else(parent == 'NA', 1, 0)) %>%
  rename(`Functional class` = parent)
  
ggplot(df) +
  geom_col(aes(x = `Functional class`, y = `siRNAs targeting`, fill = as.factor(n))) +
  theme_bw() +
  scale_fill_manual(values = c("1" = "grey80", "0" = "steelblue")) +
  theme(legend.position = 'none') +
  theme(axis.text.x = element_text(angle = 80, hjust = 1))
ggsave("sirna_class_dist.pdf", device = 'pdf', height = 4, width = 10)
