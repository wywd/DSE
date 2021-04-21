local matfile = require('matio')

t=torch.load('output/cub_contribution_map.t7')

matfile.save('output/cub_contribution_map.mat',t)

t=torch.load('output/cub_activation_map.t7')

matfile.save('output/cub_activation_map.mat',t)
