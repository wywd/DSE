cub_data=torch.load('data/cubsphere.t7')
cub_data_val=cub_data['val']
cub_data_val_labels=cub_data_val['imageClass']
torch.save('data/cubsphere_labels.t7',cub_data_val_labels)
