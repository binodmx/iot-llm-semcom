from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from Codec import Codec
from Validator import Validator
from ParsedDecisionTreeClassifier import ParsedDecisionTreeClassifier
from tqdm import tqdm
import gradio as gr
import pandas as pd
import seaborn as sns
import os
import pickle
import matplotlib.pyplot as plt
import time
import numpy as np


df = None
resampled_df = None
X_train = None
y_train = None
X_test = None
y_test = None
feature_names = {}
dataset_name = "cic-iot-2023"
dataset_type = "sample"
llm_name = "gpt-5-mini"
n_models = 10
dataset_names = ["cic-iot-2023", "wustl-iiot", "ton-iot", "bot-iot"]
dataset_types = ["sample", "population"]
llm_names = ["gpt-5-mini", "gemini-2.5-flash", "claude-haiku-4-5"]
f1_scores_4 = {}
f1_scores_3 = {}
f1_scores_3_improved = {}
c_reports_4 = {}
c_reports_3 = {}
c_reports_3_improved = {}
cms_4 = {}
cms_3 = {}
cms_3_improved = {}


encoder_system_msg = """
You are given a decision tree represented as text.
Each line shows a split based on a feature and threshold, with indentation indicating tree depth.
1. First, extract all the paths from the root to the leaves of the tree.
2. Then identify the most important decision rules from these paths composed of feature thresholds that lead to a class label.
3. Finally, ONLY output 3 most important decision rules.
"""
encoder_human_msg = """
Analyze the decision tree.
ONLY output 3 most important decision rules.
"""
decoder_system_msg ="""
You are a highly skilled AI model specialized in decision tree modification.
Given a decision tree text, your task is to modify the tree by replacing some decision rules.
1. First, extract all the paths from the root to the leaves of the tree.
2. Identify redundant features that can be replaced using the provided information.
3. Make sure to keep two nodes for same feature (for <= and >) otherwise ignore the feature.
4. ONLY output the modified decision tree text.
"""
decoder_human_msg = """
Analyze the give decision tree text and refine using the given information.
ONLY output the modified decision tree text between triple backticks.
"""
validator_system_msg = """
You are a highly skilled AI model specialized in decision tree error handling.
1. First, given an improved decision tree text, your task is to identify and fix the errors.
    1.1. Check the given improved tree satisfy the binary tree structure.
    1.2. Check each feature evaluation has <= and > thresholds.
    1.3. Check whether each condition node has its opposite condition node.
    1.4. Check whether feature names are in feature names.
    1.5. You are given old decision tree for reference to validate the improved decision tree text.
2. Then, use the 'validate' tool to validate the modified decision tree text until the tree text is valid.
3. Check whether given decision tree exceeds the expected f1-score.
4. Finally, output the correct decision tree text between triple backticks.
"""
validator_human_msg = """
Validate the following decision tree text.
"""


def generate_decision_tree_models():
    global X_train
    global y_train
    global feature_names
    global n_models

    models_3 = []
    tree_texts_3 = []
    models_4 = []
    tree_texts_4 = []

    px_size = len(X_train) // n_models
    feature_names[dataset_name] = X_train.columns.tolist()

    for i in tqdm(range(n_models), ncols=100, desc=f"Generating tree texts with max_depth=3"):
        model = DecisionTreeClassifier(
            max_depth=3,
            max_leaf_nodes=4,
            max_features=3,
            random_state=42
        )
        model.fit(X_train[i*px_size:i*px_size+px_size], y_train[i*px_size:min(i*px_size+px_size,len(X_train))])
        models_3.append(model)
        tree_text = export_text(model, feature_names=feature_names[dataset_name], max_depth=3)
        tree_texts_3.append(tree_text)

    for i in tqdm(range(n_models), ncols=100, desc=f"Generating tree texts with max_depth=4"):
        model = DecisionTreeClassifier(
            max_depth=4,
            max_leaf_nodes=8,
            max_features=7,
            random_state=42
        )
        model.fit(X_train[i*px_size:i*px_size+px_size], y_train[i*px_size:min(i*px_size+px_size,len(X_train))])
        models_4.append(model)
        tree_text = export_text(model, feature_names=feature_names[dataset_name], max_depth=4)
        tree_texts_4.append(tree_text)

    with open(os.getcwd() + f'/models/{dataset_name}_{dataset_type}_{n_models}_dts_4.pkl', 'wb') as f:
        pickle.dump(models_4, f)

    with open(os.getcwd() + f'/models/{dataset_name}_{dataset_type}_{n_models}_dts_3.pkl', 'wb') as f:
        pickle.dump(models_3, f)

    with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_4.pkl', 'wb') as f:
        pickle.dump(tree_texts_4, f)

    with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_3.pkl', 'wb') as f:
        pickle.dump(tree_texts_3, f)

    gr.Info("Decision Tree models generated successfully!")


def get_plot(df):
    fig = plt.figure(figsize=(4, 3))   
    s = sns.countplot(data=df.sort_values('label'), x='label', hue='label')
    for p in s.patches:
        s.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 9),
                textcoords = 'offset points')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def resample_dataset(df):
    # Separate majority and minority classes
    df_majority = df[df.label == df.label.value_counts().idxmax()]
    df_minority = df[df.label != df.label.value_counts().idxmax()]
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                    replace=True,                  # sample with replacement
                                    n_samples=len(df_majority),    # match majority class
                                    random_state=42)
    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_balanced


def update_dataset():
    global dataset_name
    global dataset_type
    global df
    global resampled_df
    global X_train
    global y_train
    global X_test
    global y_test
    global feature_names
    dataset_path = f"/data/{dataset_name}/{dataset_type}.csv"
    df = pd.read_csv(os.getcwd() + dataset_path, low_memory=False)
    if dataset_name == "cic-iot-2023":
        df['label'] = df['label'].apply(lambda x: "Attack" if x != "BenignTraffic" else "Benign")
    elif dataset_name == "wustl-iiot":
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column])
        df['label'] = df['Target'].apply(lambda x: "Attack" if x != 0 else "Benign")
        df = df.drop(columns=['Target', 'Traffic'])
    elif dataset_name == "ton-iot":
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column])
        df['label'] = df['label'].apply(lambda x: "Attack" if x != 0 else "Benign")
        df = df.drop(columns=['type'])
    elif dataset_name == "bot-iot":
        label_encoder = LabelEncoder()
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column].astype(str))
        df['label'] = df['attack'].apply(lambda x: "Attack" if x != 0 else "Benign")
        df = df.drop(columns=['attack', 'category', 'subcategory', 'pkSeqID'])
    resampled_df = resample_dataset(df)
    X = resampled_df.drop(columns='label')
    y = resampled_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.2,               # 80% train, 20% test (can adjust)
        stratify=y,                  # This preserves the class ratio
        random_state=42              # For reproducibility
    )
    feature_names[dataset_name] = X_train.columns.tolist()
    print(f"Dataset updated to {dataset_name}'s {dataset_type}")
    return None


def update_plots(dataset_name_val, dataset_type_val):
    global dataset_name
    global dataset_type
    global df
    global resampled_df
    dataset_name = dataset_name_val
    dataset_type = dataset_type_val
    update_dataset()
    return get_plot(df), get_plot(resampled_df)


def update_llm_model(llm_model_val):
    global llm_name
    llm_name = llm_model_val
    print(f"LLM model updated to {llm_name}")
    return None


def encode(progress=gr.Progress()):
    global n_models
    global llm_name
    global dataset_name
    global dataset_type
    print(f"Encoding with {dataset_name}'s decision tree texts using {llm_name}")
    with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_4.pkl', 'rb') as f:
        tree_texts_4 = pickle.load(f)
    semantic_encoder = Codec("semantic_encoder", model_name=llm_name)
    encoded_ai_messages = []
    for i in progress.tqdm(range(n_models), desc="Encoding decision trees"):
        system_message = SystemMessage(content=encoder_system_msg)
        human_message = HumanMessage(content=f"""{encoder_human_msg}
        Here is the decision tree text:
        ```
        {tree_texts_4[i]}
        ```
        """)
        final_state = semantic_encoder.invoke({"messages": [system_message, human_message]}, [dataset_name, dataset_type, llm_name, "encode"])
        encoded_ai_message = final_state['messages'][-1]
        encoded_ai_messages.append(encoded_ai_message.content)
    with open(os.getcwd() + f'/encoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_etts_4.pkl', 'wb') as f:
        pickle.dump(encoded_ai_messages, f)
    gr.Info("Decision trees encoded successfully!")
    return "Encoding Completed!"


def decode(progress=gr.Progress()):
    global n_models
    global llm_name
    global dataset_name
    global dataset_type
    print(f"Decoding with {dataset_name}'s decision tree texts using {llm_name}")
    with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_3.pkl', 'rb') as f:
        tree_texts_3 = pickle.load(f)
    with open(os.getcwd() + f'/encoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_etts_4.pkl', 'rb') as f:
        encoded_ai_messages = pickle.load(f)
    semantic_decoder = Codec("semantic_decoder", model_name=llm_name)
    decoded_ai_messages = []
    for i in progress.tqdm(range(n_models), desc="Decoding decision trees"):
        system_message = SystemMessage(content=decoder_system_msg)
        human_message = HumanMessage(content=f"""{decoder_human_msg}
        Decision tree text:
        ```
        {tree_texts_3[i]}
        ```

        Information about better decision tree:
        ```
        {encoded_ai_messages[i]}
        ```
        """)
        final_state = semantic_decoder.invoke({"messages": [system_message, human_message]}, [dataset_name, dataset_type, llm_name, "decode"])
        decoded_ai_message = final_state['messages'][-1]
        decoded_ai_messages.append(decoded_ai_message.content)
    with open(os.getcwd() + f'/decoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_dtts_3.pkl', 'wb') as f:
        pickle.dump(decoded_ai_messages, f)
    gr.Info("Decision trees decoded successfully!")
    return "Decoding Completed!"


def validate(progress=gr.Progress()):
    global n_models
    global llm_name
    global dataset_name
    global dataset_type
    global feature_names
    print(f"Validating with {dataset_name}'s decision tree texts using {llm_name}")
    with open(os.getcwd() + f'/models/{dataset_name}_{dataset_type}_{n_models}_dts_3.pkl', 'rb') as f:
        models_3 = pickle.load(f)
    with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_3.pkl', 'rb') as f:
        tree_texts_3 = pickle.load(f)
    with open(os.getcwd() + f'/decoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_dtts_3.pkl', 'rb') as f:
        decoded_ai_messages = pickle.load(f)

    @tool
    def validate(tree_text):
        """
        Validate the given decision tree text by attempting to execute after parsing it.
        If the execution is successful, the tree text is valid.
        """
        try:
            clf = ParsedDecisionTreeClassifier(tree_text, feature_names[dataset_name])
            clf.validate_tree()
            clf = ParsedDecisionTreeClassifier(tree_text, feature_names[dataset_name])
            y_true = y_test.astype(str)
            y_pred = clf.predict(X_test)
            f1 = classification_report(y_true, y_pred, digits=4, output_dict=True)['macro avg']['f1-score']
            print(f"F1-score: {f1}")
            return f"This decision tree text is valid with F1-score: {f1}."
        except Exception as e:
            print(e)
            return f"This decision tree text is invalid due to: {e}"
    
    dt_validator = Validator("dt_validator", model_name="gpt-5-mini-2025-08-07", validator_func=validate)
    validated_ai_messages = []
    for i in progress.tqdm(range(n_models), desc="Validating decision trees"):
        model = models_3[i]
        y_true = y_test
        y_pred = model.predict(X_test)
        f1 = classification_report(y_true, y_pred, digits=4, output_dict=True)['macro avg']['f1-score']
        print(f"Validating tree {i} expecting >{f1}")
        system_message = SystemMessage(content=f"""{validator_system_msg}
        
        Feature names:
        ```
        {feature_names[dataset_name]}
        ```

        Expected F1-Score: {f1}
        """)
        human_message = HumanMessage(content=f"""{validator_human_msg}
        Old decision tree text:
        ```
        {tree_texts_3[i]}
        ```

        Improved decision tree text:
        ```
        {decoded_ai_messages[i]}
        ```
        """)
        final_state = dt_validator.invoke({"messages": [system_message, human_message]}, [dataset_name, dataset_type, llm_name, "validate"])
        validated_ai_message = final_state['messages'][-1]
        validated_ai_messages.append(validated_ai_message.content)
    with open(os.getcwd() + f'/validated-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_vtts_3.pkl', 'wb') as f:
        pickle.dump(validated_ai_messages, f)
    gr.Info("Decision trees validated successfully!")
    return "Validation Completed!"


def evaluate(progress=gr.Progress()):
    global dataset_name
    global dataset_type
    global feature_names
    global llm_name

    total_steps = 3 * n_models
    completed_steps = 0
    dn = dataset_name
    ln = llm_name
    c_reports_4[dn] = {}
    f1_scores_4[dn] = {}
    cms_4[dn] = {}
    c_reports_3[dn] = {}
    f1_scores_3[dn] = {}
    cms_3[dn] = {}
    c_reports_3_improved[dn] = {}
    f1_scores_3_improved[dn] = {}
    cms_3_improved[dn] = {}

    try:
        with open(os.getcwd() + f'/models/{dn}_{dataset_type}_{n_models}_dts_4.pkl', 'rb') as f:
            models_4 = pickle.load(f)
    except FileNotFoundError:
        print(f"Global models not found for {dn}_{dataset_type}_{n_models}")
    c_reports_4[dn][ln] = []
    f1_scores_4[dn][ln] = []
    cms_4[dn][ln] = []
    for i in range(n_models):
        model = models_4[i]
        y_true = y_test
        y_pred = model.predict(X_test)
        c_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        c_reports_4[dn][ln].append(c_report)
        f1_scores_4[dn][ln].append(c_report['macro avg']['f1-score'])
        cms_4[dn][ln].append(cm)
        progress(completed_steps / total_steps, desc=f"Evaluating decision trees of depth 4")
        completed_steps += 1
    with open(os.getcwd() + f'/classification-reports/{dn}_{dataset_type}_{ln}_{n_models}_cr_4.pkl', 'wb') as f:
        pickle.dump(c_reports_4, f)

    with open(os.getcwd() + f'/confusion-matrices/{dn}_{dataset_type}_{ln}_{n_models}_cm_4.pkl', 'wb') as f:
        pickle.dump(cms_4, f)

    try:
        with open(os.getcwd() + f'/models/{dn}_{dataset_type}_{n_models}_dts_3.pkl', 'rb') as f:
            models_3 = pickle.load(f)
    except FileNotFoundError:
        print(f"Local models not found for {dn}_{dataset_type}_{n_models}")
    c_reports_3[dn][ln] = []
    f1_scores_3[dn][ln] = []
    cms_3[dn][ln] = []
    for i in range(n_models):
        model = models_3[i]
        y_true = y_test
        y_pred = model.predict(X_test)
        c_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        c_reports_3[dn][ln].append(c_report)
        f1_scores_3[dn][ln].append(c_report['macro avg']['f1-score'])
        cms_3[dn][ln].append(cm)
        progress(completed_steps / total_steps, desc=f"Evaluating decision trees of depth 3")
        completed_steps += 1
    with open(os.getcwd() + f'/classification-reports/{dn}_{dataset_type}_{ln}_{n_models}_cr_3.pkl', 'wb') as f:
        pickle.dump(c_reports_3, f)

    with open(os.getcwd() + f'/confusion-matrices/{dn}_{dataset_type}_{ln}_{n_models}_cm_3.pkl', 'wb') as f:
        pickle.dump(cms_3, f)

    try:
        with open(os.getcwd() + f'/validated-texts/{dn}_{dataset_type}_{ln}_{n_models}_vtts_3.pkl', 'rb') as f:
            validated_ai_messages = pickle.load(f)
    except FileNotFoundError:
        print(f"Validated texts not found for {dn}_{dataset_type}_{ln}_{n_models}")
    c_reports_3_improved[dn][ln] = []
    f1_scores_3_improved[dn][ln] = []
    cms_3_improved[dn][ln] = []
    for i in range(n_models):
        try:
            clf = ParsedDecisionTreeClassifier(validated_ai_messages[i].split('```')[1].split('```')[0], feature_names[dn])
            y_true = y_test.astype(str)
            y_pred = clf.predict(X_test)
            c_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
            labels = np.unique(y_true)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            f1_score = c_report['macro avg']['f1-score']
        except Exception as e:
            f1_score = 0.0
            c_report = None
            cm = None
            print(e)
        c_reports_3_improved[dn][ln].append(c_report)
        f1_scores_3_improved[dn][ln].append(f1_score)
        cms_3_improved[dn][ln].append(cm)
        progress(completed_steps / total_steps, desc=f"Evaluating improved decision trees of depth 3")
        completed_steps += 1
    with open(os.getcwd() + f'/classification-reports/{dn}_{dataset_type}_{ln}_{n_models}_cr_3_improved.pkl', 'wb') as f:
        pickle.dump(c_reports_3_improved, f)

    with open(os.getcwd() + f'/confusion-matrices/{dn}_{dataset_type}_{ln}_{n_models}_cm_3_improved.pkl', 'wb') as f:
        pickle.dump(cms_3_improved, f)
    gr.Info("Decision trees evaluated successfully!")
    return "Evaluation Completed!"


def get_text(dataset_name, dataset_type, llm_name, tt_type, i, n_models=10):
    if tt_type == "Tree Texts 4":
        with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_4.pkl', 'rb') as f:
            tree_texts = pickle.load(f)
    elif tt_type == "Tree Texts 3":
        with open(os.getcwd() + f'/tree-texts/{dataset_name}_{dataset_type}_{n_models}_tts_3.pkl', 'rb') as f:
            tree_texts = pickle.load(f)
    elif tt_type == "Encoded Tree Texts":
        with open(os.getcwd() + f'/encoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_etts_4.pkl', 'rb') as f:
            tree_texts = pickle.load(f)
    elif tt_type == "Decoded Tree Texts":
        with open(os.getcwd() + f'/decoded-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_dtts_3.pkl', 'rb') as f:
            tree_texts = pickle.load(f)
    elif tt_type == "Validated Tree Texts":
        with open(os.getcwd() + f'/validated-texts/{dataset_name}_{dataset_type}_{llm_name}_{n_models}_vtts_3.pkl', 'rb') as f:
            tree_texts = pickle.load(f)
    return list(tree_texts)[i]


with gr.Blocks() as app:
    gr.Markdown("# AGL-IDS: Agentic LLM-Driven Semantic Policy Interpretation and Enforcement for Guided Learning in IoT Intrusion Detection Systems")

    with gr.Tab("Main"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Generate Decision Tree Models (GM & LM)")
                dataset_dropdown = gr.Dropdown(choices=dataset_names, value=dataset_names[0], label="Select Dataset", interactive=True)
                dataset_type_dropdown = gr.Dropdown(choices=dataset_types, value=dataset_types[0], label="Select Dataset Type", interactive=True)
                button1 = gr.Button("Generate Decision Tree Models")
                button1.click(fn=generate_decision_tree_models)
            with gr.Column():
                gr.Markdown("### &nbsp;")
                dataset_plot = gr.Plot(label="Dataset Distribution")
            with gr.Column():
                gr.Markdown("### &nbsp;")
                resampled_dataset_plot = gr.Plot(label="Resampled Dataset Distribution")
            dataset_dropdown.change(fn=update_plots, inputs=[dataset_dropdown, dataset_type_dropdown], outputs=[dataset_plot, resampled_dataset_plot])
            dataset_type_dropdown.change(fn=update_plots, inputs=[dataset_dropdown, dataset_type_dropdown], outputs=[dataset_plot, resampled_dataset_plot])
            app.load(fn=update_plots, inputs=[dataset_dropdown, dataset_type_dropdown], outputs=[dataset_plot, resampled_dataset_plot])
        gr.HTML("<hr>")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 2: Encode ➜ Decode ➜ Validate")
                llm_model_dropdown = gr.Dropdown(choices=llm_names, value=llm_names[0], label="Select LLM Model", interactive=True)
                llm_model_dropdown.change(fn=update_llm_model, inputs=[llm_model_dropdown])
            with gr.Column():
                gr.Markdown("&nbsp;")
            with gr.Column():
                gr.Markdown("&nbsp;")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    system_msg_textbox = gr.Textbox(label="System Message", value=encoder_system_msg, max_lines=10)
                    user_msg_textbox = gr.Textbox(label="User Message", value=encoder_human_msg, max_lines=10)
                encoding_status = gr.Textbox(label="Status", lines=1)
                encode_button = gr.Button("Encode")
                encode_button.click(fn=encode, outputs=[encoding_status])
                
            with gr.Column():
                with gr.Row():
                    system_msg_textbox = gr.Textbox(label="System Message", value=decoder_system_msg, max_lines=10)
                    user_msg_textbox = gr.Textbox(label="User Message", value=decoder_human_msg, max_lines=10)
                decoding_status = gr.Textbox(label="Status", lines=1)
                decode_button = gr.Button("Decode")
                decode_button.click(fn=decode, outputs=[decoding_status])

            with gr.Column():
                with gr.Row():
                    system_msg_textbox = gr.Textbox(label="System Message", value=validator_system_msg, max_lines=10)
                    user_msg_textbox = gr.Textbox(label="User Message", value=validator_human_msg, max_lines=10)
                validation_status = gr.Textbox(label="Status", lines=1)
                validate_button = gr.Button("Validate")
                validate_button.click(fn=validate, outputs=[validation_status])
        gr.HTML("<hr>")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 3: Evaluate")
                evaluate_status = gr.Textbox(label="Status", lines=1)
                evaluate_button = gr.Button("Evaluate")
                evaluate_button.click(fn=evaluate, outputs=[evaluate_status])
            with gr.Column():
                gr.Markdown("&nbsp;")
            with gr.Column():
                gr.Markdown("&nbsp;")

    with gr.Tab("Observe"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Tree Text 1")
                tt_type = gr.Radio(label="Select Type", choices=["Tree Texts 4", "Tree Texts 3", "Encoded Tree Texts", "Decoded Tree Texts", "Validated Tree Texts"], value="Tree Texts 4", interactive=True)
                dataset_name = gr.Radio(label="Select Dataset", choices=dataset_names, value="cic-iot-2023", interactive=True)
                dataset_type = gr.Radio(label="Select Type", choices=dataset_types, value="population", interactive=False)
                llm_name = gr.Radio(label="Select LLM", choices=llm_names, value=llm_names[0], interactive=True)
                i = gr.Radio(label="Select Index", choices=list(range(10)), value=0, interactive=True)
                output_text = gr.Textbox(label="Output", interactive=False, lines=10)

                
            
            with gr.Column():
                gr.Markdown("### Tree Text 2")
                tt_type2 = gr.Radio(label="Select Type", choices=["Tree Texts 4", "Tree Texts 3", "Encoded Tree Texts", "Decoded Tree Texts", "Validated Tree Texts"], value="Tree Texts 4", interactive=True)
                dataset_name2 = gr.Radio(label="Select Dataset", choices=dataset_names, value="cic-iot-2023", interactive=True)
                dataset_type2 = gr.Radio(label="Select Type", choices=dataset_types, value="population", interactive=False)
                llm_name2 = gr.Radio(label="Select LLM", choices=llm_names, value=llm_names[0], interactive=True)
                i2 = gr.Radio(label="Select Index", choices=list(range(10)), value=0, interactive=True)
                output_text2 = gr.Textbox(label="Output", interactive=False, lines=10)
            
            tt_type.change(get_text, [dataset_name, dataset_type, llm_name, tt_type, i], output_text)
            dataset_name.change(get_text, [dataset_name, dataset_type, llm_name, tt_type, i], output_text)
            dataset_type.change(get_text, [dataset_name, dataset_type, llm_name, tt_type, i], output_text)
            llm_name.change(get_text, [dataset_name, dataset_type, llm_name, tt_type, i], output_text)
            i.change(get_text, [dataset_name, dataset_type, llm_name, tt_type, i], output_text)
            i.change(fn=lambda x: x, inputs=[i], outputs=[i2])
            app.load(fn=get_text, inputs=[dataset_name, dataset_type, llm_name, tt_type, i], outputs=output_text)

            tt_type2.change(get_text, [dataset_name2, dataset_type2, llm_name2, tt_type2, i2], output_text2)
            dataset_name2.change(get_text, [dataset_name2, dataset_type2, llm_name2, tt_type2, i2], output_text2)
            dataset_type2.change(get_text, [dataset_name2, dataset_type2, llm_name2, tt_type2, i2], output_text2)
            llm_name2.change(get_text, [dataset_name2, dataset_type2, llm_name2, tt_type2, i2], output_text2)
            i2.change(fn=lambda x: x, inputs=[i2], outputs=[i])
            i2.change(get_text, [dataset_name2, dataset_type2, llm_name2, tt_type2, i2], output_text2)
            app.load(fn=get_text, inputs=[dataset_name2, dataset_type2, llm_name2, tt_type2, i2], outputs=output_text2)

    with gr.Tab("Charts"):
        gr.Markdown("### Variation in F1-Score")
        with gr.Row():
            with gr.Column():
                gr.Image("charts/aaa-cic-iot-2023_bar_chart_f1.png", label="cic-iot-2023")
            with gr.Column():
                gr.Image("charts/aaa-wustl-iiot_bar_chart_f1.png", label="wustl-iiot")
            with gr.Column():
                gr.Image("charts/aaa-ton-iot_bar_chart_f1.png", label="ton-iot")
            with gr.Column():
                gr.Image("charts/aaa-bot-iot_bar_chart_f1.png", label="bot-iot")

        gr.Markdown("### Variation in False Positive Rate")
        with gr.Row():
            with gr.Column():
                gr.Image("charts/aaa-cic-iot-2023_dumbbell_chart_fpr.png", label="cic-iot-2023")
            with gr.Column():
                gr.Image("charts/aaa-wustl-iiot_dumbbell_chart_fpr.png", label="wustl-iiot")
            with gr.Column():
                gr.Image("charts/aaa-ton-iot_dumbbell_chart_fpr.png", label="ton-iot")
            with gr.Column():
                gr.Image("charts/aaa-bot-iot_dumbbell_chart_fpr.png", label="bot-iot")

        gr.Markdown("### Variation in Data Volume")
        with gr.Row():
            with gr.Column():
                gr.Image("charts/aaa-cic-iot-2023_bar_chart_compression.png", label="cic-iot-2023")
            with gr.Column():
                gr.Image("charts/aaa-wustl-iiot_bar_chart_compression.png", label="wustl-iiot")
            with gr.Column():
                gr.Image("charts/aaa-ton-iot_bar_chart_compression.png", label="ton-iot")
            with gr.Column():
                gr.Image("charts/aaa-bot-iot_bar_chart_compression.png", label="bot-iot")

        gr.Markdown("### Knowledge Distillation Loss")
        with gr.Row():
            with gr.Column():
                gr.Plot()
            with gr.Column():
                gr.Plot()
            with gr.Column():
                gr.Plot()
            with gr.Column():
                gr.Plot()


app.launch()
