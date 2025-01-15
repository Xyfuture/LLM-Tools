import streamlit as st
from llm_tools.config.memory_config import load_predefined_models
from llm_tools.utils.resource_calc import InferenceConfig,ModelConfig
from llm_tools.utils.resource_calc import ComputeCalc,MemoryCalc
from llm_tools.utils.resource_calc import concat_unit,convert_unit_from_bytes,convert_unit_from_ops

st.set_page_config(page_title= 'LLM Resource Requirements')
st.title("LLM Resource Requirements")




# ----------------- Sidebar UI ----------------- #

# available pre-defined models
MODELS = load_predefined_models()
DATA_TYPES = {"fp16","fp8","fp4","int16","int8","int4"}



inference_config = InferenceConfig()


# initial value 
def initial_value_from_json():
    pass 

def initial_session_state():
    # lm_config 
    for key,value in inference_config.lm_config.model_dump().items():
        st.session_state[f'model_config-{key}'] = value 
    # sequence_config 
    for key,value in inference_config.sequence_config.model_dump().items():
        st.session_state[f'sequence_config-{key}'] = value
    # data_type_config 
    for key,value in inference_config.data_type_config.model_dump().items():
        st.session_state[f'data_type_config-{key}'] = value 

initial_session_state()

print('info')
for key,value in st.session_state.items():
    print(key)
print('end')


# update value 
def update_model_config():
    print('update model config')
    for key,value in st.session_state.items():
        if '-' not in key:
            continue
        config_name, entry = key.split('-',1)
        print( config_name, entry)
        if config_name == 'model_config':
            inference_config.lm_config.__dict__.update({entry:value})
    

def update_data_type_config():
    print('update data type config')
    for key,value in st.session_state.items():
        if '-' not in key:
            continue
        config_name, entry = key.split('-',1)
        print( config_name, entry)
        if config_name == 'data_type_config':
            inference_config.data_type_config.__dict__.update({entry:value})

def update_sequence_config():
    print('update sequecen config')
    for key,value in st.session_state.items():
        if '-' not in key:
            continue
        config_name, entry = key.split('-',1)
        print( config_name, entry)
        if config_name == 'sequence_config':
            inference_config.sequence_config.__dict__.update({entry:value})



# Model Selection
model = st.sidebar.selectbox(
    "Model", list(MODELS.keys()), index=None, on_change=initial_value_from_json, key="model"
)

# Parameters
model_size = st.sidebar.number_input(
    "Number of parameters (in billions)",
    min_value=0,
    step=1,
    value=None,
    key="model_config-model_size",
    help="Number of parameters in the model in billions",
    on_change=update_model_config,
)


weight_data_type = st.sidebar.selectbox(
    "Weight Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-weight_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    on_change=update_data_type_config,
)
activation_data_type = st.sidebar.selectbox(
    "Activation Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-activation_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    on_change=update_data_type_config,
)
kv_cache_data_type = st.sidebar.selectbox(
    "KV Cache Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-kv_cache_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    on_change=update_data_type_config,
)


batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=1,
    key="sequence_config-batch_size",
    on_change=update_sequence_config,

)
prefill_length = st.sidebar.number_input(
    "Prefill Length",
    min_value=0,
    step=1,
    value=2048,
    key="sequence_config-prefill_length",
    help="Number of tokens in the input sequence.",
    on_change=update_sequence_config,
)
decoding_length = st.sidebar.number_input(
    "Decoding Length",
    min_value=0,
    step = 1,
    value=2048,
    key='sequence_config-decoding_length',
    help="Number of tokens in the output sequence.",
    on_change=update_sequence_config,
)


hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=None,
    key="model_config-hidden_size",
    help="Size of the hidden layer (given by the model card).",
    on_change=update_model_config,
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=None,
    key="model_config-hidden_layers",
    help="Number of layers in the model (given by the model card).",
    on_change=update_model_config,
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="model_config-num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
    on_change=update_model_config,
)
num_key_value_heads = st.sidebar.number_input(
    "Number of Key Value Heads",
    min_value=0,
    step=1,
    value=None,
    key="model_config-num_key_value_heads",
    help="Number of key value heads in the model (given by the model card).",
    on_change=update_model_config,
)
intermediate_size = st.sidebar.number_input(
    "Intermediate Size",
    min_value=0,
    step=1,
    value=None,
    key="model_config-intermediate_size",
    help="Intermediate Size of FFN (given by the model card).",
    on_change=update_model_config,
)



memory_calc = MemoryCalc(inference_config)
compute_calc = ComputeCalc(inference_config)

memory_map = memory_calc.get_total_memory()
compute_map = compute_calc.get_total_ops()


# Memory Usage
st.write(f"**Total Inference Memory**: {concat_unit(convert_unit_from_bytes(memory_map['total']))}")
st.write(f"- **Model Weights**: {concat_unit(convert_unit_from_bytes(memory_map['weight']))}")
st.write(f"- **KV Cache**: {concat_unit(convert_unit_from_bytes(memory_map['kv_cache']))}")

st.markdown("---")

# Compute Usage
st.write(f"**Total Prefill Ops**: {compute_map['prefill']}")
st.write(f"**Total Decoding Ops**: {compute_map['decoding']}")




# ----------------- Error Handling ----------------- #
if None in st.session_state.values():
    st.warning("Some information is missing.")



