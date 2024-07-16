import columns
import training_feature_engineering
import training_data_preprocessing
import model_input
import labeling
import model
import model_training
import evaluation

if __name__ == '__main__':
    training_feature_engineering.feature_engineering()
    training_data_preprocessing.preprocess_data()
    model_input.load_model_input()
    labeling.macd_labeling()
    model.construct_model()
    model_training.training_model()
    evaluation.evaluate_prediction()
