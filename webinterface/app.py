import os
import shap
import lief
import ember
import numpy as np
import gradio as gr
import lightgbm as lgb
import matplotlib.pyplot as plt
from features import get_feature_names

def evaluate_file(binary_location, max_display, data_dir):
    lgbm_model = lgb.Booster(model_file=os.path.join(data_dir, "ember_model_2018.txt"))
    lgbm_model.params['objective'] = 'regression'
    extractor2 = ember.PEFeatureExtractor(2)
    file_data = open(binary_location, "rb").read()
    feature_vector = extractor2.feature_vector(file_data)
    # raw_features = extractor2.raw_features(file_data)
    feature_names = get_feature_names()
    explainer = shap.TreeExplainer(lgbm_model, feature_names=feature_names)
    shap_values = explainer(np.array([feature_vector], dtype=np.float32))
    shap.plots.waterfall(shap_values[0], max_display=max_display,show=False)
    plt.savefig('shap_plot.png', bbox_inches='tight')
    plt.close()
    return 'shap_plot.png'
    
def display_shap_plot():
    return 'shap_plot.png'

def classify_vectors(binary_location:str, data_dir) -> float:
    lgbm_model = lgb.Booster(model_file=os.path.join(data_dir, "ember_model_2018.txt"))
    extractor2 = ember.PEFeatureExtractor(2)

    file_data = open(binary_location, "rb").read()
    feature_vector = extractor2.feature_vector(file_data)
    return lgbm_model.predict([np.array(feature_vector, dtype=np.float32)])[0]

def main():
    data_dir = "/workspaces/torment-nexus/ember2018/"
    with gr.Blocks() as demo:
        binary_location = gr.File(label="Upload your file")
        submit_button = gr.Button("Evaluate")
        max_display = gr.Slider(value=5, minimum=0, maximum=50, step=1, label="Max display")
        output = gr.Image(label="Summary plot", width="100%", show_share_button=True)
        submit_button.click(fn=evaluate_file, inputs=[binary_location, max_display, gr.State(data_dir)], outputs=output)

    demo.launch()

if __name__ == "__main__":
    main()

