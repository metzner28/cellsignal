library(tidyverse)
setwd("/Volumes/DRIVE")

df_site1 = read_csv("md_train.csv") %>%
  filter(site == 1) %>%
  mutate(exp_n = str_sub(experiment, -3,-1)) %>%
  mutate(exp_n = as.numeric(exp_n)) %>%
  filter(exp_n < 4) %>%
  group_by(cell_type, sirna_id) %>%
  mutate(sirna_instance = row_number()) %>%
  filter(sirna_instance < 4)

df_site2 = read_csv("md_train.csv") %>%
  filter(site == 2) %>%
  mutate(exp_n = str_sub(experiment, -3,-1)) %>%
  mutate(exp_n = as.numeric(exp_n)) %>%
  filter(exp_n < 4) %>%
  group_by(cell_type, sirna_id) %>%
  mutate(sirna_instance = row_number()) %>%
  filter(sirna_instance < 4)

df = rbind(df_site1, df_site2) %>%
  arrange(site_id) %>%
  mutate(path = str_replace(path, "/Volumes/DRIVE/rxrx1", "/home/ubuntu/cellsignal")) %>%
  select(-c(exp_n, sirna_instance)) %>%
  ungroup() %>%
  arrange(sirna_id, site) %>%
  mutate(img_idx = row_number())

df_celltypes = df %>%
  group_by(cell_type) %>%
  summarise(n())

df_classes = df%>%
  group_by(sirna_id) %>%
  summarise(n())

# this is really dumb
val_indices = vector(mode = 'numeric', length = 0)
train_indices = vector(mode = 'numeric', length = 0)
for (i in seq(from = 0, to = length(unique(df$sirna_id))-1)) {
  possible_indices = seq(from = i * 24 + 1, to = i * 24 + 24)
  val_idx = sample(possible_indices, size = 4)
  train_idx = possible_indices[!possible_indices %in% val_idx]
  train_idx = sample(train_idx, size = length(train_idx))
  val_indices = c(val_indices, val_idx)
  train_indices = c(train_indices, train_idx)
}

df_val = df %>%
  filter(img_idx %in% val_indices)

df_train = df %>%
  filter(!img_idx %in% val_indices) %>%
  mutate(train_row = row_number()) %>%
  mutate(batch_idx = 1 + floor(train_row / 4 - 0.1)) %>%
  select(-c(img_idx, train_row))

lookup = enframe(unique(df_train$batch_idx))
colnames(lookup) = c("batch_idx", "batch_order")
lookup = lookup %>%
  slice_sample(prop = 1) %>%
  mutate(batch_order = row_number())
df_train = inner_join(df_train, lookup, by = 'batch_idx') %>%
  arrange(batch_order)

train_celltypes = df_train %>% group_by(cell_type) %>% summarise(n())
train_classes =  df_train %>% group_by(sirna_id) %>% summarise(n())
val_celltypes = df_val %>% group_by(cell_type) %>% summarise(n())
val_classes = df_val %>% group_by(sirna_id) %>% summarise(n())

# frequency histogram of how many classes have each number of images
# distribution of cell types in data

write.csv(df_train, "md_contrastive_train_4views.csv", row.names = F)
write.csv(df_val, "md_contrastive_val_4views.csv", row.names = F)


