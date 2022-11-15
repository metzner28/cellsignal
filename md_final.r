library(tidyverse)
setwd("/Volumes/DRIVE")

df = read_csv("md_equal.csv") %>% 
  select(-c(1,14)) %>%
  group_by(cell_type, sirna_id) %>%
  mutate(sirna_instance = row_number()) %>%
  filter(sirna_instance < 10)

df_celltypes = df %>%
  group_by(cell_type) %>%
  summarise(n())

df_classes = df %>%
  group_by(sirna_id) %>%
  summarise(n())

# frequency histogram of how many classes have each number of images
# distribution of cell types in data

write.csv(df, "md_final.csv", row.names = F)

# resample test dataset in the same way
df_test = read_csv("md_test.csv") %>%
  group_by(cell_type, sirna_id) %>%
  mutate(sirna_instance = row_number()) %>%
  filter(sirna_instance < 4) %>%
  mutate(path = str_replace(path, "/Volumes/DRIVE/rxrx1", "/home/ubuntu/cellsignal"))
  
test_celltypes = df_test %>% group_by(cell_type) %>% summarise(n = n())
test_classes = df_test %>% group_by(sirna_id) %>% summarise(n = n())

write.csv(df_test, "md_test_final.csv", row.names = F)
