from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int = 0 
    vocab_size: int = 0
    model_type:Literal['llama','gpt'] # use different ffn design


    @property
    def use_gqa(self):
        if self.num_key_value_heads == 0 or self.num_key_value_heads == self.num_attention_heads:
            return False # use MHA not GQA
        elif self.num_attention_heads != self.num_key_value_heads:
            return True # use GQA
        


@dataclass
class DataTypeConfig:
    weight_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"]
    activation_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"]
    kv_cache_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"]

    @staticmethod
    def get_bytes(dtype:Literal["fp16","fp8","fp4","int16","int8","int4"]) -> int :
        dtype_map  = {
            'fp16':2, 'fp8':1, 'fp4':0.5,
            'int16':2, 'int8':1, 'int4':0.5
        }

        return dtype_map [ dtype ]
    
    @staticmethod
    def convert_unit(data_bytes):
        # convert a value in bytes to Giga Bytes or Mega Bytes
        pass 


@dataclass
class SequenceConfig:
    batch_size: int 
    prefill_length: int
    decoding_length: int 
    

    @property
    def total_length(self):
        return self.prefill_length + self.decoding_length

@dataclass
class InferenceConfig:
    model_config:ModelConfig
    data_type_config:DataTypeConfig
    sequence_config:SequenceConfig    


class MemoryCalc:
    def __init__(self):
        
        self.inference_config:InferenceConfig = InferenceConfig()

    def get_total_memory(self):
        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        weight_layer_size = self.get_layer_weight_size()
        kv_cache_layer_size = self.get_layer_kv_cache_size()

        weight_memory = weight_layer_size * model_config.num_hidden_layers * DataTypeConfig.get_bytes(data_type_config.weight_dtype)
        kv_cache_memory = kv_cache_layer_size * model_config.num_hidden_layers * DataTypeConfig.get_bytes(data_type_config.kv_cache_dtype)
        
        total_memory = weight_memory + kv_cache_memory
        
        return {'total':total_memory, 'weight':weight_memory, 'kv_cache':kv_cache_memory}


    
    def get_layer_weight_size(self):
        # calc one layer weight size -- element number not bytes
        # W_qkv + W_o + W_ffn 

        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        # W_qkv -- MHA and GQA use different W_kv
        if not model_config.use_gqa:
            # MHA mode 
            qkv_size = 3* (self.inference_config.model_config.hidden_size ** 2)
        else:
            q_size = model_config.hidden_size ** 2
            per_head_size = model_config.hidden_size // model_config.num_attention_heads
            kv_size = 2 * model_config.hidden_size * per_head_size * model_config.num_key_value_heads
            qkv_size = q_size + kv_size 


        # W_o
        o_size = model_config.hidden_size ** 2 

        # FFN 
        if model_config.model_type == 'llama' :
            # 3 MLP
            ffn_size = 3 * model_config.hidden_size * model_config.intermediate_size 
        elif model_config.model_type == 'gpt':
            ffn_size = 2 * model_config.hidden_size * model_config.intermediate_size
        else:
            assert False
        
        layer_total_size = qkv_size + o_size + ffn_size
        
        return layer_total_size


    def get_layer_kv_cache_size(self):
        # prefill + decoding -> total size -- element number not bytes

        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        total_length = sequence_config.total_length
        per_head_size = model_config.hidden_size / model_config.num_attention_heads
        if model_config.use_gqa:
            # GQA mode 
            kv_size = 2 * per_head_size * model_config.num_key_value_heads * total_length
        else:
            # MHA mode 
            kv_size = 2 * per_head_size * model_config.num_attention_heads * total_length

        return kv_size 

        
 

class ComputeCalc:
    def __init__(self):
        self.inference_config:InferenceConfig = InferenceConfig()

    def get_total_ops(self):
        pass 

    def get_layer_qkv_proj_ops(self):
        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config
        
        if model_config.use_gqa:
            # GQA mode
            pass 
        else:
            # MHA mode 
            pass 


    def get_layer_attention_ops(self):
        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config


    def get_layer_ffn_ops(self):
        model_config,data_type_config,sequence_config  = self.inference_config.model_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        if model_config.model_type == 'llama':
            pass 
        elif model_config.model_type =='gpt':
            pass 
        else:
            pass 



class ResourceCalc:
    def __init__(self):
        pass

    








    