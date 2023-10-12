# Human-Motion-Prediction

Recent advances have allowed for the largest Archive of Motion Capture as Surface Shapes (AMASS) to be created as well as the expansion of models for motion prediction to include transformers and temporal or spatial networks. 
In this project, a subset from AMASS to test a new model that has not been used for motion prediction yet is used: Temporal Convolutional Networks (TCN). This model is compared to baseline models using an optimized RNN and Transformer-Encoder using Mean Angle Error (MAE) and Mean Squared Error (MSE) loss. 
Finally,a new error metric for this problem is considered, Average Displacement Error (ADE) and experiments are run to iterate several hyperparameters to optimize the TCN. Ultimately, a novel model for human motion prediction that outperforms traditionally used baseline models is introduced.

This project was performed using Google Colab Pro.

Models can be trained using choice of following scripts:
tcn_training.py, training_rnn_base.py, or training_transformer_base.py.

