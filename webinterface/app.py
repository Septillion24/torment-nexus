import os
import shap
import lief
import math
import ember
import numpy as np
import pandas as pd
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
    plt.savefig('/tmp/shap_plot.png', bbox_inches='tight')
    plt.close()
    return '/tmp/shap_plot.png'



def optimize_binary(binary_location:str, data_dir:str, output_dir:str, num_particles:int, num_iterations:int, cognitive_component:float, social_component:float, weight:float,):
    def objective_function(df):
        if type(df) is pd.DataFrame:
            df = df.to_numpy()
         
        lgbm_model = lgb.Booster(model_file=os.path.join(data_dir, "ember_model_2018.txt"))
        return lgbm_model.predict([np.array(df, dtype=np.float32)])[0]

    def get_dataframe(feature_vector:np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(feature_vector).T
        df.columns = get_feature_names()
        return df

    def get_vectors(binary_location:str) -> np.ndarray:
        extractor2 = ember.PEFeatureExtractor(2)
        file_data = open(binary_location, "rb").read()
        return extractor2.feature_vector(file_data)

    df = get_dataframe(get_vectors(binary_location))

    dim = df.shape[1]
 
    boundsdict = {"header.coff.timestamp": (0, 0xFFFFFFFF),
                "directories.certificate_table_size": (0,0xFFFFFFFF),
                "directories.debug_vaddress":(0,0xFFFFFFFF),
                "directories.certificate_table_vaddress": (0,0xFFFFFFFF),
                "header.optional.major_subsystem_version":(7,10),
                "directories.export_table_vaddress":(0,0xFFFFFFFF),
                "directories.export_table_size":(0,0xFFFFFFFF),
                # "general.has_tls":(0,1),
                # "general.has_signature":(0,1),
                # "general.has_debug":(0,1),
                }



    
    
    changeable_str = ["header.coff.timestamp",
                    "directories.certificate_table_size",
                    "directories.debug_vaddress",
                    "directories.certificate_table_vaddress",
                    "directories.export_table_vaddress",
                    "directories.export_table_size",
                    "header.optional.major_subsystem_version",
                    # "general.has_tls",
                    # "general.has_signature",
                    # "general.has_debug"
                    ]
    bounds = []
    changeable = []
    for index,feature in enumerate(df):
        if feature not in changeable_str:
            bounds.append((df[feature].iloc[0], df[feature].iloc[0]))
        else:
            bounds.append(boundsdict[feature])
            changeable.append(index)

    particles = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([objective_function(p) for p in particles])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]


    for iteration in range(num_iterations):
        for i in range(num_particles):
            for j in changeable:
                velocities[i, j] = (weight * velocities[i, j] +
                                    cognitive_component * np.random.rand() * (personal_best_positions[i, j] - particles[i, j]) +
                                    social_component * np.random.rand() * (global_best_position[j] - particles[i, j]))
                particles[i, j] += velocities[i, j]
                particles[i, j] = np.clip(particles[i, j], bounds[j][0], bounds[j][1])
            score = objective_function(particles[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]
                
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        print(f"Best so far: {min(personal_best_scores)}, particle #{np.argmin(personal_best_scores)}")

    print("Best position:", global_best_position)
    
    bestpositiondf = get_dataframe(global_best_position)
    binary = lief.PE.parse(binary_location)
    data_directory = binary.data_directories 

    timestamp = bestpositiondf["header.coff.timestamp"]
    certificate_table_vaddress = bestpositiondf["directories.certificate_table_vaddress"]
    certificate_table_size = bestpositiondf["directories.certificate_table_size"]
    debug_vaddress = bestpositiondf["directories.debug_vaddress"]
    export_table_vaddress = bestpositiondf["directories.export_table_vaddress"]
    export_table_size = bestpositiondf["directories.export_table_size"]
    major_subsystem_version = round(bestpositiondf["header.optional.major_subsystem_version"])
    has_tls = round(bestpositiondf["general.has_tls"])
    has_signatures = round(bestpositiondf["general.has_signature"])
    has_debug = round(bestpositiondf["general.has_debug"])
    binary.header.time_date_stamps = timestamp
    data_directory[4].rva = certificate_table_vaddress
    data_directory[4].size = certificate_table_size
    data_directory[6].rva = debug_vaddress
    data_directory[0].rva = export_table_vaddress
    data_directory[0].size = export_table_size
    # binary.has_tls = has_tls
    # binary.has_signatures = has_signatures
    # binary.has_debug = has_debug
    binary.optional_header.major_subsystem_version = major_subsystem_version

    binary.write(f"{output_dir}/edited.exe")
    
    

def classify_binary(binary_location:str, data_dir) -> float:
    lgbm_model = lgb.Booster(model_file=os.path.join(data_dir, "ember_model_2018.txt"))
    extractor2 = ember.PEFeatureExtractor(2)

    file_data = open(binary_location, "rb").read()
    feature_vector = extractor2.feature_vector(file_data)
    return lgbm_model.predict([np.array(feature_vector, dtype=np.float32)])[0]

def main():
    data_dir = "/workspaces/torment-nexus/ember2018/"
    output_dir = "/workspaces/torment-nexus/binaries/outputs"
    with gr.Blocks() as demo:
        binary_location = gr.File(label="Upload your file")
        max_display = gr.Slider(value=10, minimum=0, maximum=50, step=1, label="Max display")
        submit_button = gr.Button("Evaluate")
        output = gr.Image(label="Summary plot", width="100%", show_share_button=True)
        num_particles = gr.Slider(value=20, minimum=0, maximum=50, step=1, label="Number of particles")
        num_iterations = gr.Slider(value=5, minimum=0, maximum=50, step=1, label="Number of iterations")
        cognitive_component = gr.Number(value=1.2, minimum=0, maximum=2, label="Cognitive component")
        social_component = gr.Number(value=1.2, minimum=0, maximum=2, label="Social component")
        weight = gr.Number(value=0.8, minimum=0, maximum=2, label="Weight")
        optimize_button = gr.Button("Optimize")
        submit_button.click(fn=evaluate_file, inputs=[binary_location, max_display, gr.State(data_dir)], outputs=output)
        optimize_button.click(fn=optimize_binary, inputs=[binary_location, gr.State(data_dir), gr.State(output_dir),num_particles, num_iterations, cognitive_component, social_component, weight])

    demo.launch()

if __name__ == "__main__":
    main()

