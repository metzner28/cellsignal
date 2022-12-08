library(tidyverse)
library(ggplot2)
setwd("~/Dropbox/Documents/tri-i/FoDS/project/analysis/celltype_results/")

df_test = read_csv("../../cellsignal/md_test_final.csv") %>%
  select(-c(1,2))
df_pred = read_csv("cell_type_ce_preds_top5.csv")

df = cbind(df_test, df_pred) %>%
  mutate(exp_label_check = experiment == exp,
         sirna_check = sirna_id == label) %>%
  filter(exp_label_check & sirna_check)

df_clean = df %>%
  select(cell_type, well_type, sirna_id, top1, top5)

cell_type_proportion = df_clean %>%
  group_by(cell_type) %>%
  summarise(pct_correct = 100 * sum(top5) / sum(df_clean$top5))

test_set_dist = df_test %>%
  group_by(cell_type) %>%
  summarise(pct_test = 100 * n() / nrow(.))

train_dist = read_csv("../md_final.csv") %>%
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
  theme(legend.position = c(0.85, 0.85))
ggsave("celltype_model_cell_type_dist.pdf", device = 'pdf', width = 9, height = 6)

train_path = read_csv("cell_type_ce.csv") %>%
  mutate(epoch = 1:32,
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
ggsave("celltype_training_path.pdf", device = 'pdf', width = 10, height = 3)

well_type_corrects = df_clean %>%
  filter(top5 == 1) %>%
  group_by(well_type) %>%
  summarise(correct = 100 * n() / nrow(.)) %>%
  arrange(-correct)

well_type_test = df_test %>%
  group_by(well_type) %>%
  summarise(overall_test = 100 * n() / nrow(.))

well_type_train = read_csv("../md_final.csv") %>%
  group_by(well_type) %>%
  summarise(overall_train = 100 * n() / nrow(.))

well_types = inner_join(well_type_test, well_type_train, by = 'well_type') %>%
  inner_join(well_type_corrects, by = 'well_type') %>%
  pivot_longer(2:4, names_to = 'distribution', values_to = 'pct')

ggplot(well_types) + 
  geom_col(aes(x = well_type, y = pct, fill = distribution), position = 'dodge') +
  theme_bw() +
  theme(legend.position = c(0.15, 0.8))
ggsave("cell_type_model_well_type_dist.pdf", device = 'pdf', width = 9, height = 6)

# heatmap of functional class distribution
sirna_classes = read_csv("../sirna_functions.csv")  %>%
  select(-c(1,2))
df_clean = inner_join(df_clean, sirna_classes, by = 'sirna_id')

class_corrects = df_clean %>%
  group_by(parent) %>%
  summarise(correct = 100 * sum(top5) / sum(df_clean$top5))

df_test = inner_join(df_test, sirna_classes, by = 'sirna_id')
test_classes = df_test %>%
  group_by(parent) %>%
  summarise(overall_test = 100 * n() / nrow(.))

train_classes = read_csv("../md_final.csv") %>%
  inner_join(sirna_classes, by = 'sirna_id') %>%
  group_by(parent) %>%
  summarise(overall_train = 100 * n() / nrow(.))

classes = inner_join(class_corrects, test_classes, by = 'parent') %>%
  inner_join(train_classes, by = 'parent') %>%
  pivot_longer(2:4, names_to = 'distribution', values_to = 'pct') %>%
  rename(sirna_protein_class = parent) %>%
  mutate(pct_round = round(pct,2))

ggplot(classes) +
  geom_tile(aes(x = sirna_protein_class, y = distribution, fill = distribution, 
                alpha = pct)) +
  scale_alpha_continuous(range = c(0,1)) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) +
  theme(axis.text.x = element_text(angle = 75, hjust = 1)) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) +
  ylab("") +
  geom_text(aes(x = sirna_protein_class, y = distribution, label = pct_round), size = 2)
ggsave("cell_type_pclass_dist.pdf", device = 'pdf', width = 10, height = 5)
  
             