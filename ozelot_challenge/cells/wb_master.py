import wandb

sweep_config = {
    'method': 'bayes', #'bayes' # 'random'
    'metric': {
        'name': 'val/dfl_loss',
        'goal': 'minimize'   
        },
    }

parameters_dict = {
    'model': {
        'values': ['s', 'm', 'l']
        },
    'epochs': {
        'values': [50]
        },
    'lr': {
        'values': [
            0.1, 
            0.01, 
            0.001,
        ]
    },
    'lrf': {
        'values': [
            1, 
            0.5, 
            0.01,
        ]
    },
    'warmup_epochs': {
        'values': [
            0, 
            2, 
            5, 
        ]
    },
    'cls': {
        'values': [
            1, 
            2, 
            4, 
        ]
    },    
    'dropout': {
        'values': [
            0, 
            0.1, 
            0.25, 
        ]
    },
    'scale': {
        'values': [
            0, 
            0.1, 
            0.25, 
        ]
    },
    'label_smoothing': {
        'values': [
            0, 
            0.1, 
            0.25, 
        ]
    },
    'mixup': {
        'values': [
            0, 
            0.1,  
        ]
    },
    'copy_paste': {
        'values': [
            0, 
            0.1,  
        ]
    }
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="YOLOV8_OCELOT_Cells")
print(sweep_id)