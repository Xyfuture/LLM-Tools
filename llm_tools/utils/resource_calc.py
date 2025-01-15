from dataclasses import dataclass,fields
from typing import Literal
import json 
from pydantic import BaseModel


def get_bytes(dtype:Literal["fp16","fp8","fp4","int16","int8","int4"]) -> float :
    dtype_map  = {
        'fp16':2, 'fp8':1, 'fp4':0.5,
        'int16':2, 'int8':1, 'int4':0.5
    }

    return dtype_map [ dtype ]

def convert_unit_from_bytes(data_bytes:int)->tuple[float,str]:
    # convert a value in bytes to Giga Bytes or Mega Bytes
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    index = 0
    while data_bytes >= 1024 and index < len(units) - 1:
        data_bytes /= 1024
        index += 1
    return data_bytes , units[index]


def convert_unit_from_ops(mac_ops:int)-> tuple[float,str]:
    if mac_ops < 2e30:
        return mac_ops / 2e30 , 'GOPs'
    else:
        return mac_ops/ 2e40 , 'TOPs'
    

def concat_unit(data:tuple[float,str])->str:
    return f"{data[0]:.2f}{data[1]}"


class ModelConfig(BaseModel):
    class Config:
        allow_mutation = True
    
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32 
    num_attention_heads: int = 32
    num_key_value_heads: int = 0
    vocab_size: int = 0
    model_type:Literal['llama','gpt'] = 'llama' # use different ffn design

        

    @property
    def use_gqa(self):
        if self.num_key_value_heads == 0 or self.num_key_value_heads == self.num_attention_heads:
            return False # use MHA not GQA
        elif self.num_attention_heads != self.num_key_value_heads:
            return True # use GQA
    
    @property
    def per_head_size(self):
        return self.hidden_size // self.num_attention_heads




class DataTypeConfig(BaseModel):
    class Config:
        allow_mutation = True
        
    weight_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'
    activation_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'
    kv_cache_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'


class SequenceConfig(BaseModel):
    class Config:
        allow_mutation = True

    batch_size: int = 32
    prefill_length: int = 1024
    decoding_length: int = 1024
    

    @property
    def total_length(self):
        return self.prefill_length + self.decoding_length


class InferenceConfig(BaseModel):
    class Config:
        allow_mutation = True
    data_type_config:DataTypeConfig = DataTypeConfig()
    lm_config:ModelConfig = ModelConfig()
    sequence_config:SequenceConfig = SequenceConfig()


class MemoryCalc:
    def __init__(self,inference_config:InferenceConfig):
        
        self.inference_config:InferenceConfig = inference_config
        # print(self.inference_config)

    def get_total_memory(self)->dict[str,float]:
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        weight_layer_size = self.get_layer_weight_size()
        kv_cache_layer_size = self.get_layer_kv_cache_size()

        weight_memory = weight_layer_size * lm_config.num_hidden_layers *  get_bytes(data_type_config.weight_dtype)
        kv_cache_memory = kv_cache_layer_size * lm_config.num_hidden_layers * get_bytes(data_type_config.kv_cache_dtype)
        
        total_memory = weight_memory + kv_cache_memory
        
        return {'total':total_memory, 'weight':weight_memory, 'kv_cache':kv_cache_memory}


    
    def get_layer_weight_size(self):
        # calc one layer weight size -- element number not bytes
        # W_qkv + W_o + W_ffn 

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config
        # print('inner', type(lm_config))

        # W_qkv -- MHA and GQA use different W_kv
        if not lm_config.use_gqa:
            # MHA mode 
            qkv_proj_size = 3* (self.inference_config.lm_config.hidden_size ** 2)
        else:
            q_proj_size = lm_config.hidden_size ** 2
            per_head_size = lm_config.hidden_size // lm_config.num_attention_heads
            kv_proj_size = 2 * lm_config.hidden_size * per_head_size * lm_config.num_key_value_heads
            qkv_proj_size = q_proj_size + kv_proj_size 


        # W_o
        o_proj_size = lm_config.hidden_size ** 2 

        # FFN 
        if lm_config.model_type == 'llama' :
            # 3 MLP
            ffn_size = 3 * lm_config.hidden_size * lm_config.intermediate_size 
        elif lm_config.model_type == 'gpt':
            ffn_size = 2 * lm_config.hidden_size * lm_config.intermediate_size
        else:
            assert False
        
        layer_total_size = qkv_proj_size + o_proj_size + ffn_size
        
        return layer_total_size


    def get_layer_kv_cache_size(self):
        # prefill + decoding -> total size -- element number not bytes

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        total_length = sequence_config.total_length
        per_head_size = lm_config.hidden_size / lm_config.num_attention_heads
        if lm_config.use_gqa:
            # GQA mode 
            kv_size = 2 * sequence_config.batch_size * per_head_size * lm_config.num_key_value_heads * total_length
        else:
            # MHA mode 
            kv_size = 2 * sequence_config.batch_size * per_head_size * lm_config.num_attention_heads * total_length

        return kv_size 

        
 

class ComputeCalc:
    def __init__(self,inference_config:InferenceConfig):
        self.inference_config:InferenceConfig = inference_config
        # print(self.inference_config)


    def get_total_ops(self)->dict[str,int]:
        # MAC ops 
        laeyr_qkv_proj_ops_map = self.get_layer_qkv_proj_ops()
        layer_attention_ops_map = self.get_layer_attention_ops()
        layer_ffn_ops_map = self.get_layer_ffn_ops()

        total_ops_map = {}
        for key in ['decoding','prefill']:
            total_ops_map[key] = self.inference_config.lm_config.num_hidden_layers *(
                                    laeyr_qkv_proj_ops_map[key]+layer_attention_ops_map[key] + layer_ffn_ops_map[key]
                                )

        return total_ops_map


    def get_layer_qkv_proj_ops(self)->dict[str,int]:
        # prefill and decoding -- different stages 

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config
        

        # decoding 
        if lm_config.use_gqa:
            # GQA mode
            q_proj_decoding_ops = sequence_config.batch_size * lm_config.hidden_size * lm_config.hidden_size
            kv_proj_decoding_ops = 2 * sequence_config.batch_size * lm_config.hidden_size * (lm_config.per_head_size * lm_config.num_key_value_heads)

            qkv_proj_decoding_ops = q_proj_decoding_ops + kv_proj_decoding_ops
             
        else:
            # MHA mode 
            qkv_proj_decoding_ops = 3 * sequence_config.batch_size * lm_config.hidden_size * lm_config.hidden_size
        
        qkv_proj_prefill_ops =  qkv_proj_decoding_ops * sequence_config.prefill_length

        return  {'decoding':qkv_proj_decoding_ops,'prefill':qkv_proj_prefill_ops}


    def get_layer_attention_ops(self)->dict[str,int]:
        # prefill and decoding -- different stages 

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        # prefill
        # MHA and GQA actually the same ops, GQA only brings memory usage gain 
        q_mul_k_prefill_ops = sequence_config.batch_size * (sequence_config.prefill_length * lm_config.per_head_size * sequence_config.prefill_length * lm_config.num_attention_heads)
        score_mul_v_prefill_ops = sequence_config.batch_size * (sequence_config.prefill_length * sequence_config.prefill_length * lm_config.per_head_size * lm_config.num_attention_heads)
        o_proj_prefill_ops = sequence_config.batch_size * sequence_config.prefill_length * lm_config.hidden_size * lm_config.hidden_size

        attention_prefill_ops = q_mul_k_prefill_ops + score_mul_v_prefill_ops + o_proj_prefill_ops 

        # decoding 
        q_mul_k_decoding_ops = sequence_config.batch_size * (1 * lm_config.per_head_size * sequence_config.total_length * lm_config.num_attention_heads)
        score_mul_v_decoding_ops = sequence_config.batch_size * (1 * sequence_config.total_length * lm_config.per_head_size * lm_config.num_attention_heads)
        o_proj_decoding_ops = sequence_config.batch_size * 1 * lm_config.hidden_size * lm_config.hidden_size

        attention_decoding_ops = q_mul_k_decoding_ops + score_mul_v_decoding_ops + o_proj_decoding_ops

        return {'decoding':attention_decoding_ops , 'prefill':attention_prefill_ops}




    def get_layer_ffn_ops(self)->dict[str,int]:
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        if lm_config.model_type == 'llama':
            # three MLP matrix
            ffn_prefill_ops = sequence_config.batch_size * 3 * sequence_config.prefill_length * lm_config.hidden_size * lm_config.hidden_size
            ffn_decoding_ops = sequence_config.batch_size * 3 * 1 * lm_config.hidden_size * lm_config.hidden_size 
        elif lm_config.model_type =='gpt':
            ffn_prefill_ops = sequence_config.batch_size * 3 * sequence_config.prefill_length * lm_config.hidden_size * lm_config.hidden_size
            ffn_decoding_ops = sequence_config.batch_size * 3 * 1 * lm_config.hidden_size * lm_config.hidden_size
        else:
            assert False

        return {'decoding':ffn_decoding_ops, 'prefill':ffn_prefill_ops}



class ResourceCalc:
    def __init__(self):
        pass

    








    