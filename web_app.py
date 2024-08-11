import torch
from dataset_webapp import MoleculeDataset
from model_webapp import ModelPredNumPeak
from torch_geometric.loader import DataLoader
from utils_webapp import resume, count_parameters, enable_dropout
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import streamlit as st


def predict_num_of_peaks(smile, mc_sam=10):
    model_name = "ModelPredNumPeak"
    params = {}
    arch_params = {
        'dim_h': 256,  # Example parameter, adjust as needed
        'additional_feature_size': 12  # Example parameter, adjust as needed
    }
    n_data_points = 1
    dtf_predictions = pd.DataFrame({'smile': [smile]})

    for str_dataset in ['down', 'up']:
        dataset = MoleculeDataset(dtf_predictions)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        lst_file = [model for model in os.listdir(rf'web_app/pred_num_peak_{str_dataset}') if model.endswith('.pth')]
        lst_file.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        best_model_ckpt = lst_file[-1]

        params["model_edge_dim"] = dataset[0].edge_attr.shape[1]

        # device = "cuda" if torch.cuda.is_available else "cpu"
        device = torch.device("cpu")
        # model = eval(model_name)(node_feature_size=dataset[0].x.shape[1],
        #                          edge_feature_size=dataset[0].edge_attr.shape[1],
        #                          n_data_points=n_data_points, **arch_params)
        model = eval(model_name)(
            node_feature_size=dataset[0].x.shape[1],
            edge_feature_size=params["model_edge_dim"],
            dim_h=arch_params['dim_h'],
            n_data_points=n_data_points,
            additional_feature_size=arch_params['additional_feature_size']
        )

        print("Number of params: ", count_parameters(model))
        model.to(device)

        resume(model, os.path.join(f'web_app/pred_num_peak_{str_dataset}', best_model_ckpt))
        model.eval()
        enable_dropout(model)

        for batch in tqdm(loader):
            lst_pred = []
            batch.to(device)
            # if len(batch.smiles) < 32: continue

            for i in range(mc_sam):
                pred = model(batch.x.float(),
                             None,
                             batch.edge_attr.float(),
                             batch.edge_index,
                             batch.batch)
                lst_pred.append(pred)

            pred = torch.mean(torch.stack(lst_pred, dim=2), dim=2)
            y_pred_batch = np.round(torch.squeeze(pred).cpu().detach().numpy())

        dtf_predictions[f'PRED_NUM_PEAK_{str_dataset}'] = y_pred_batch

    return dtf_predictions


# Main title and user input fields
st.title('Prediction of number of  raman peaks starting from a SMILE')
st.write('Enter the SMILE representation of a molecule')

# User input for pixel values (8x8 image)
smile = st.text_area('Insert SMILE')

# Prediction and display result
if st.button('Predict'):
    prediction = predict_num_of_peaks(smile)
    st.write(f'Predicted fingerprint region peaks: {prediction.PRED_NUM_PEAK_down.iloc[0]}')
    st.write(f'Predicted CH region peaks: {prediction.PRED_NUM_PEAK_up.iloc[0]}')


# smile = 'CCO'
# pred = predict_num_of_peaks(smile)
# print(pred)