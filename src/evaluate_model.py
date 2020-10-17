import torch
from models.mcnnpytorch.src.crowd_count import CrowdCounter
from models.mcnnpytorch.src import network
import numpy as np

def evaluate_model(trained_model, data_loader, is_cuda=False):
    dtype = torch.FloatTensor if not is_cuda else torch.cuda.FloatTensor
    net = CrowdCounter(is_cuda=is_cuda)
    network.load_net(trained_model, net, dtype=dtype)
    if is_cuda:
        net.cuda()
    net.eval()
    
    # values
    MAEcrowddensity = {'High': 0, 'Med': 0, 'Low': 0}
    MSEcrowddensity = {'High': 0, 'Med': 0, 'Low': 0}
    MAEweather = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    MSEweather = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}
    MAE = 0.0
    MSE = 0.0
    
    # counts for averaging
    crowddensity_count = {'High': 0, 'Med': 0, 'Low': 0}
    weather_count = {'None': 0, 'Fog': 0, 'Rain': 0, 'Snow': 0}

    with torch.no_grad():
        for blob in data_loader:                        
            im_data = blob['data']
            gt_data = blob['gt_density']
            metadata = blob['metadata']
            
            crowd_density = metadata['crowd_density']
            weather = metadata['weather']
            
            density_map = net(im_data, gt_data)
            density_map = density_map.data.cpu().numpy()
            gt_count = np.sum(gt_data)
            et_count = np.sum(density_map)
            
            # updating the values
            MAEcrowddensity[crowd_density] += abs(gt_count-et_count)
            MSEcrowddensity[crowd_density] += ((gt_count-et_count)*(gt_count-et_count))
            MAEweather[weather] += abs(gt_count-et_count)
            MSEweather[weather] += ((gt_count-et_count)*(gt_count-et_count))
            MAE += abs(gt_count-et_count)
            MSE += ((gt_count-et_count)*(gt_count-et_count))
            
            # updating the counts
            crowddensity_count[crowd_density] += 1
            weather_count[weather] += 1
        
    # averaging
    for key in crowddensity_count:
        MAEcrowddensity[key] = MAEcrowddensity[key] / crowddensity_count[key] if crowddensity_count[key] else 0
        MSEcrowddensity[key] = MSEcrowddensity[key] / crowddensity_count[key] if crowddensity_count[key] else 0
    
    # averaging
    for key in weather_count:
        if weather_count[key] != 0:
            MAEweather[key] /= weather_count[key]
            MSEweather[key] /= weather_count[key]
    
    # averaging
    MAE /= data_loader.get_num_samples()
    MSE /= data_loader.get_num_samples()
    RMSE = np.sqrt(MSE)
    
    return MAEcrowddensity, MSEcrowddensity, MAEweather, MSEweather, MAE, MSE, RMSE