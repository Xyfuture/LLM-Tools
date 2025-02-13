import streamlit as st
from llm_tools.utils.resource_calc import InferenceConfig,ModelConfig
from llm_tools.utils.resource_calc import ComputeCalc,MemoryCalc
from llm_tools.utils.resource_calc import concat_unit,convert_unit_from_bytes,convert_unit_from_ops
from llm_tools.utils.resource_calc import get_predefined_models,load_predefined_model_config

st.set_page_config(page_title= 'LLM Resource Requirements')
st.title("LLM Resource Requirements")




# ----------------- Sidebar UI ----------------- #

# available pre-defined models
MODELS = get_predefined_models()
DATA_TYPES = {"fp16","fp8","fp4","int16","int8","int4"}




inference_config = InferenceConfig()


# initial value 
def initial_value_from_json():
    lm_config = load_predefined_model_config(st.session_state['model'])
    inference_config.lm_config = lm_config

    initial_session_state()


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


# update value 
def update_model_config():
    for key,value in st.session_state.items():
        if '-' not in key or value is None:
            continue
        config_name, entry = key.split('-',1)
        if config_name == 'model_config':
            inference_config.lm_config.__dict__.update({entry:value})
    

def update_data_type_config():
    for key,value in st.session_state.items():
        if '-' not in key or value is None:
            continue
        config_name, entry = key.split('-',1)
        if config_name == 'data_type_config':
            inference_config.data_type_config.__dict__.update({entry:value})

def update_sequence_config():
    for key,value in st.session_state.items():
        if '-' not in key or value is None :
            continue
        config_name, entry = key.split('-',1)
        if config_name == 'sequence_config':
            inference_config.sequence_config.__dict__.update({entry:value})



# Model Selection
model = st.sidebar.selectbox(
    "Model", list(MODELS), index=None, on_change=initial_value_from_json, key="model"
)

# Parameters
weight_data_type = st.sidebar.selectbox(
    "Weight Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-weight_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    # on_change=update_data_type_config,
)
activation_data_type = st.sidebar.selectbox(
    "Activation Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-activation_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    # on_change=update_data_type_config,
)
kv_cache_data_type = st.sidebar.selectbox(
    "KV Cache Data Type",
    DATA_TYPES,
    index=None,
    key="data_type_config-kv_cache_dtype",
    help="Data type used (int 8 and int 4 are for quantization)",
    # on_change=update_data_type_config,
)


batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=0,
    step=1,
    value=1,
    key="sequence_config-batch_size",
    # on_change=update_sequence_config,
)
prefill_length = st.sidebar.number_input(
    "Prefill Length",
    min_value=0,
    step=1,
    value=2048,
    key="sequence_config-prefill_length",
    help="Number of tokens in the input sequence.",
    # on_change=update_sequence_config,
)
decoding_length = st.sidebar.number_input(
    "Decoding Length",
    min_value=0,
    step = 1,
    value=2048,
    key='sequence_config-decoding_length',
    help="Number of tokens in the output sequence.",
    # on_change=update_sequence_config,
)


hidden_size = st.sidebar.number_input(
    "Hidden Size",
    min_value=0,
    step=1,
    value=None,
    key="model_config-hidden_size",
    help="Size of the hidden layer (given by the model card).",
    # on_change=update_model_config,
)
num_hidden_layers = st.sidebar.number_input(
    "Number of Layers",
    min_value=0,
    step=1,
    value=None,
    key="model_config-num_hidden_layers",
    help="Number of layers in the model (given by the model card).",
    # on_change=update_model_config,
)
num_attention_heads = st.sidebar.number_input(
    "Number of Attention Heads",
    min_value=0,
    step=1,
    value=None,
    key="model_config-num_attention_heads",
    help="Number of attention heads in the model (given by the model card).",
    # on_change=update_model_config,
)
num_key_value_heads = st.sidebar.number_input(
    "Number of Key Value Heads",
    min_value=0,
    step=1,
    value=None,
    key="model_config-num_key_value_heads",
    help="Number of key value heads in the model (given by the model card).",
    # on_change=update_model_config,
)
intermediate_size = st.sidebar.number_input(
    "Intermediate Size",
    min_value=0,
    step=1,
    value=None,
    key="model_config-intermediate_size",
    help="Intermediate Size of FFN (given by the model card).",
    # on_change=update_model_config,
)


update_model_config()
update_data_type_config()
update_sequence_config()


memory_calc = MemoryCalc(inference_config)
compute_calc = ComputeCalc(inference_config)

memory_info = memory_calc.get_all_info()
compute_info = compute_calc.get_all_info()


# # Memory Usage
# st.write(f"**Total Inference Memory**: {concat_unit(convert_unit_from_bytes(memory_map['total']))}")
# st.write(f"- **Model Weights**: {concat_unit(convert_unit_from_bytes(memory_map['weight']))}") 
# st.write(f"- **KV Cache**: {concat_unit(convert_unit_from_bytes(memory_map['kv_cache']))}")

# st.markdown("---")

# # Compute Usage
# st.write(f"**Total Prefill Ops**: {concat_unit(convert_unit_from_ops(compute_map['prefill']))}")
# st.write(f"**Total Decoding Ops**: {concat_unit(convert_unit_from_ops(compute_map['decoding']))}")

# # Communicate Usage
# st.markdown("---")
# st.write(f"**--**: {1}")


# Memory 指绝对的大小, Size 指元素的个数 

convert_memory = lambda x:concat_unit(convert_unit_from_bytes(x))
convert_ops = lambda x:concat_unit(convert_unit_from_ops(x))

# Memory Usage
st.write(f"**Memory Usage**:")
st.write(f"- **Model Total Memory**:{convert_memory(memory_info['model_total_memory'])}")
st.write(f"- **Model Weights Memory**:{convert_memory(memory_info['model_weights_memory'])}")
st.write(f"- **Model KV Cache Memory**:{convert_memory(memory_info['model_kv_cache_memory'])}")
st.write(f"- **Model KV Cache Per Token Memory**:{convert_memory(memory_info['model_kv_cache_per_token_memory'])}")
st.write(f" ")
st.write(f"- **Layer Wegihts Memory**:{convert_memory(memory_info['layer_weights_memory'])}")
st.write(f"- **Layer KV Cache Memory**:{convert_memory(memory_info['layer_kv_cache_memory'])}")
st.write(f"- **Layer KV Cache Per Token Memory**:{convert_memory(memory_info['layer_kv_cache_per_token_memory'])}")


# Compute Usage 
st.write(f"**Compute Usage**:")
st.write(f"- **Model Prefill Ops**:{convert_ops(compute_info['model_prefill_ops'])}")
st.write(f"- **Model Decoding Ops**:{convert_ops(compute_info['model_decoding_ops'])}")
st.write(f"  ")
st.write(f"- **Model Decoding MLP Ops**:{convert_ops(compute_info['model_decoding_mlp_ops'])}")
st.write(f"- **Model Decoding Attention Ops**:{convert_ops(compute_info['model_decoding_attention_ops'])}")
st.write(f"  ")
st.write(f"- **Layer Decoding MLP Ops**:{convert_ops(compute_info['layer_decoding_mlp_ops'])}")
st.write(f"- **Layer Decoding Attention Ops**:{convert_ops(compute_info['layer_decoding_attention_ops'])}")


# # Transfer Usage
# st.write(f"**Transfer Usage**:")
# st.write(f"- **Layer Prefill Projection Linear Size**:")
# st.write(f"- **Layer Prefill FNN Linear Size**:")
# st.write(f"- **Layer Decoding Projection Linear Size**:")
# st.write(f"- **Layer Decoding FFN Linear Size**:")















"""
暂时的需求

Model Summary 
    Hidden Dim
    Number of Layers
    Intermediate Size 
    Attention Heads
    KV Heads 


Memory Usage
    Model Weights
    Model KV Cache
    Model KV Cache Per Token 

    Layer Weights
    Layer KV Cache 
    Layer KV Cache Per token 

Compute Usage
    Model Prefill Ops
    Model Decoding Ops 
    
    Model Decoding MLP Ops
    Model Decoding Attention Ops

Transfer Usage
    # 简单起见,先做 layer的 model level的之后看看需要什么补上 
    
    Layer Prefill Projection Linear Size / Memory  # 普通的情况
    Layer Prefill FFN Linear Size / Memory  # FFN 中间的情况

    Layer Decoding Projection Linear Size / Memory
    Layer Decoding FFN Linear Size / Memory 



"""


# ----------------- Error Handling ----------------- #
if None in st.session_state.values():
    st.warning("Some information is missing.")



