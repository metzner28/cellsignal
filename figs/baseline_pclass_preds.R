library(tidyverse)

df_base = read_csv('baseline_results/baseline_ce_preds_top5.csv')
sirna_classes = read_csv('sirna_functions.csv')

get_sirna = function(sirna) {
  return(sirna_classes$parent[sirna_classes$sirna_id == sirna])
}

df_pred_base = df_base %>%
  mutate(pred_class = sapply(max_pred, get_sirna),
         true_class = sapply(label, get_sirna),
         correct_class = if_else(pred_class == true_class,1,0))
base_acc = sum(df_pred_base$correct_class, na.rm = T) / sum(!is.na(df_pred_base$true_class))

df_cell = read_csv('celltype_results/cell_type_ce_preds_top5.csv')
df_pred_cell = df_cell %>%
  mutate(pred_class = sapply(max_pred, get_sirna),
         true_class = sapply(label, get_sirna),
         correct_class = if_else(pred_class == true_class,1,0))
cell_acc = sum(df_pred_cell$correct_class, na.rm = T) / sum(!is.na(df_pred_cell$true_class))

df_focal = read_csv('baseline_results/baseline_focal_preds_top5.csv')
df_pred_focal = df_focal %>%
  mutate(pred_class = sapply(max_pred, get_sirna),
         true_class = sapply(label, get_sirna),
         correct_class = if_else(pred_class == true_class,1,0))
focal_acc = sum(df_pred_focal$correct_class, na.rm = T) / sum(!is.na(df_pred_focal$true_class))