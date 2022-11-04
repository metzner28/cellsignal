library(tidyverse)
library(ggplot2)
setwd("~/Dropbox/Documents/tri-i/FoDS/project/analysis/from_gpu")

df_test = read_csv("md_test.csv") %>%
  select(-c(1,2))
df_pred = read_csv("resnet18_baseline_preds.csv")

df = cbind(df_test, df_pred) %>%
  mutate(exp_label_check = experiment == exp_label,
         sirna_check = sirna_id == true_label) %>%
  filter(exp_label_check & sirna_check)

df_clean = df %>%
  select(cell_type, well_type, sirna_id, pred_class, correct)

cell_type_proportion = df_clean %>%
  group_by(cell_type) %>%
  summarise(pct_correct = 100 * sum(correct) / sum(df_clean$correct))

test_set_dist = df_test %>%
  group_by(cell_type) %>%
  summarise(pct_test = 100 * n() / nrow(.))

train_dist = read_csv("md_downsampled.csv") %>%
  group_by(cell_type) %>%
  summarise(overall_train = 100 * n() / nrow(.))

df_plot = inner_join(cell_type_proportion, test_set_dist, by = "cell_type") %>%
  inner_join(train_dist, by = 'cell_type') %>%
  rename(overall_test = pct_test,
         correct = pct_correct) %>%
  pivot_longer(2:4, names_to = 'distribution', values_to = "pct")

ggplot(df_plot) + 
  geom_col(aes(x = cell_type, y = pct, fill = distribution), position = 'dodge') +
  theme_bw() +
  theme(legend.position = c(0.15, 0.8))
ggsave("../../milestone/cell_type_dist.pdf", device = 'pdf', width = 6, height = 4)

train_path = read_csv("resnet_18_training_path.csv") %>%
  mutate(epoch = 1:20,
         train_acc = 100 * train_acc,
         val_acc = 100 * val_acc) %>%
  rename(train = train_acc, 
         val =  val_acc) %>%
  select(epoch, train, val) %>%
  pivot_longer(2:3, names_to = "accuracy", values_to = "pct")
  
ggplot(train_path) +
  geom_line(aes(x = epoch, y = pct, color = accuracy)) +
  theme_bw() +
  theme(legend.position = c(0.1,0.8))
ggsave("../../milestone/training_path.pdf", device = 'pdf', width = 10, height = 3)

well_type_corrects = df_clean %>%
  filter(correct) %>%
  group_by(well_type) %>%
  summarise(correct = 100 * n() / nrow(.)) %>%
  arrange(-correct)

well_type_test = df_test %>%
  group_by(well_type) %>%
  summarise(overall_test = 100 * n() / nrow(.))

well_type_train = read_csv("md_downsampled.csv") %>%
  group_by(well_type) %>%
  summarise(overall_train = 100 * n() / nrow(.))

well_types = inner_join(well_type_test, well_type_train, by = 'well_type') %>%
  inner_join(well_type_corrects, by = 'well_type') %>%
  pivot_longer(2:4, names_to = 'distribution', values_to = 'pct')

ggplot(well_types) + 
  geom_col(aes(x = well_type, y = pct, fill = distribution), position = 'dodge') +
  theme_bw() +
  theme(legend.position = c(0.15, 0.8))
ggsave("../../milestone/well_type_dist.pdf", device = 'pdf', width = 6, height = 4)
  
  
             