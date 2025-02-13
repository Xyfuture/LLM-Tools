from dataclasses import dataclass,fields
from typing import Literal
import json 
from pydantic import BaseModel
import os 


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
    if mac_ops < 2**40:
        return mac_ops / (2**30) , 'GOPs'
    else:
        return mac_ops/ (2**40) , 'TOPs'
    

def concat_unit(data:tuple[float,str])->str:
    return f"{data[0]:.2f}{data[1]}"


def get_predefined_models()->list:
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_directory = os.path.dirname(current_file_path)
    parent_directory = os.path.dirname(current_directory)

    models = []
    for model_file in os.listdir(os.path.join(parent_directory, "model_cards")):
        if model_file.endswith(".json"):
            models.append(model_file[:-5])
    return models


    




class ModelConfig(BaseModel):
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32 
    num_attention_heads: int = 32
    num_key_value_heads: int = 0
    vocab_size: int = 0
    model_type:Literal['llama','gpt','mixtral','phi','mistral','phi3','qwen2'] = 'llama' # use different ffn design

        

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
    weight_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'
    activation_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'
    kv_cache_dtype: Literal["fp16","fp8","fp4","int16","int8","int4"] = 'int8'


class SequenceConfig(BaseModel):
    batch_size: int = 1
    prefill_length: int = 1024
    decoding_length: int = 1024
    

    @property
    def total_length(self):
        return self.prefill_length + self.decoding_length


class InferenceConfig(BaseModel):
    data_type_config:DataTypeConfig = DataTypeConfig()
    lm_config:ModelConfig = ModelConfig()
    sequence_config:SequenceConfig = SequenceConfig()



def load_predefined_model_config(model_name)->InferenceConfig:
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_directory = os.path.dirname(current_file_path)
    parent_directory = os.path.dirname(current_directory)
    with open(os.path.join(parent_directory, "model_cards", f"{model_name}.json"), "r") as f:
        json_dict = json.load(f)
        lm_config = ModelConfig(**json_dict)

    return lm_config 




class MemoryCalc:
    def __init__(self,inference_config:InferenceConfig):
        
        self.inference_config:InferenceConfig = inference_config
        # print(self.inference_config)

    def get_all_info(self)->dict[str,float]:
        layer_weights_info = self.get_layer_weight_info()
        layer_kv_cache_info = self.get_layer_kv_cache_info()

        model_info = self.get_model_info()
        
        info_dict = {**layer_weights_info,**layer_kv_cache_info,**model_info}

        return info_dict


    def get_model_info(self)->dict[str,float]:
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        weight_layer_size = self.get_layer_weight_info()['layer_weights_size']
        kv_cache_layer_size = self.get_layer_kv_cache_info()['layer_kv_cache_size']
        kv_cache_per_token_layer_size = self.get_layer_kv_cache_info()['layer_kv_cache_per_token_size']
        vocab_misc_size = self.get_vocab_misc_info()['vocab_misc_size']

        weight_size = weight_layer_size * lm_config.num_hidden_layers
        kv_cache_size = kv_cache_layer_size * lm_config.num_hidden_layers
        kv_cache_per_token_size = kv_cache_per_token_layer_size * lm_config.num_hidden_layers

        weight_memory = weight_size *  get_bytes(data_type_config.weight_dtype)
        kv_cache_memory = kv_cache_size * get_bytes(data_type_config.kv_cache_dtype)
        kv_cache_per_token_memory = kv_cache_per_token_size * get_bytes(data_type_config.kv_cache_dtype)
        vocab_misc_memory = vocab_misc_size * get_bytes(data_type_config.weight_dtype)
        
        weight_memory += vocab_misc_memory

        total_memory = weight_memory + kv_cache_memory  
        
        return {'model_total_memory':total_memory,
                'model_weights_memory':weight_memory,
                'model_kv_cache_memory':kv_cache_memory,
                'model_kv_cache_per_token_memory': kv_cache_per_token_memory,
                }


    
    def get_layer_weight_info(self):
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
        elif lm_config.model_type == 'qwen2':
            ffn_size = 3 * lm_config.hidden_size * lm_config.intermediate_size
        else:
            assert False
        
        layer_weights_size = qkv_proj_size + o_proj_size + ffn_size
        layer_weights_memory = layer_weights_size * get_bytes(data_type_config.weight_dtype)

        return {'layer_weights_size':layer_weights_size,'layer_weights_memory':layer_weights_memory}


    def get_layer_kv_cache_info(self):
        # prefill + decoding -> total size -- element number not bytes

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        total_length = sequence_config.total_length
        per_head_size = lm_config.hidden_size / lm_config.num_attention_heads
        if lm_config.use_gqa:
            # GQA mode 
            layer_kv_cache_per_token_size = 2 * sequence_config.batch_size * per_head_size * lm_config.num_key_value_heads
            layer_kv_cache_size = layer_kv_cache_per_token_size * total_length 
        else:
            # MHA mode 
            layer_kv_cache_per_token_size = 2 * sequence_config.batch_size * per_head_size * lm_config.num_attention_heads
            layer_kv_cache_size = layer_kv_cache_per_token_size* total_length
        
        layer_kv_cache_memory = layer_kv_cache_size * get_bytes(data_type_config.kv_cache_dtype) 
        layer_kv_cache_per_token_memory = layer_kv_cache_per_token_size * get_bytes(data_type_config.kv_cache_dtype)


        return {'layer_kv_cache_size':layer_kv_cache_size,
                'layer_kv_cache_memory':layer_kv_cache_memory,
                'layer_kv_cache_per_token_size':layer_kv_cache_per_token_size,
                'layer_kv_cache_per_token_memory':layer_kv_cache_per_token_memory}
    

    def get_vocab_misc_info(self):
        # lm_head and tokenize
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config
        lm_head_size = lm_config.hidden_size * lm_config.vocab_size
        tokenizer_size = lm_config.vocab_size * lm_config.hidden_size


        vocab_misc_size = lm_head_size + tokenizer_size
        vocab_misc_memory = vocab_misc_size * get_bytes(data_type_config.weight_dtype)

        return {'vocab_misc_size':vocab_misc_size,'vocab_misc_memory':vocab_misc_memory}

 

class ComputeCalc:
    def __init__(self,inference_config:InferenceConfig):
        self.inference_config:InferenceConfig = inference_config
        # print(self.inference_config)

    def get_all_info(self)->dict[str,int]:
        # info = dict().update(self.get_model_ops()).update(self.get_layer_attention_ops()).update(self.get_layer_mlp_ops())
        info = {**(self.get_model_ops()),**(self.get_layer_attention_ops()),**(self.get_layer_mlp_ops())}
        return info 
        

    def get_model_ops(self)->dict[str,int]:
        # TODO 词表相关的运算要加进来
        # MAC ops 
        laeyr_qkv_proj_ops_map = self.get_layer_qkv_proj_ops()
        layer_attention_ops_map = self.get_layer_attention_ops()
        layer_ffn_ops_map = self.get_layer_ffn_ops()
        layer_output_proj_ops_map = self.get_layer_output_proj_ops()

        layer_mlp_ops_map = self.get_layer_mlp_ops()

        model_ops={'model_prefill_ops':0,'model_decoding_ops':0,
                   'model_decoding_mlp_ops':0,'model_decoding_attention_ops':0}
        for info in [laeyr_qkv_proj_ops_map,layer_attention_ops_map,layer_ffn_ops_map,layer_output_proj_ops_map]:
            for k,v in info.items():
                if 'prefill' in k:
                    model_ops['model_prefill_ops'] += v*self.inference_config.lm_config.num_hidden_layers
                if 'decoding' in k :
                    model_ops['model_decoding_ops'] += v*self.inference_config.lm_config.num_hidden_layers
        
        model_ops['model_decoding_mlp_ops'] = layer_mlp_ops_map['layer_decoding_mlp_ops'] * self.inference_config.lm_config.num_hidden_layers
        model_ops['model_decoding_attention_ops'] = layer_attention_ops_map['layer_decoding_attention_ops'] * self.inference_config.lm_config.num_hidden_layers

        return model_ops


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

        return  {'layer_decoding_qkv_projection_ops':qkv_proj_decoding_ops,'layer_prefill_qkv_projection_ops':qkv_proj_prefill_ops}


    def get_layer_attention_ops(self)->dict[str,int]:
        # prefill and decoding -- different stages 

        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        # prefill
        # MHA and GQA actually the same ops, GQA only brings memory usage gain 
        q_mul_k_prefill_ops = sequence_config.batch_size * (sequence_config.prefill_length * lm_config.per_head_size * sequence_config.prefill_length * lm_config.num_attention_heads)
        score_mul_v_prefill_ops = sequence_config.batch_size * (sequence_config.prefill_length * sequence_config.prefill_length * lm_config.per_head_size * lm_config.num_attention_heads)
        # o_proj_prefill_ops = sequence_config.batch_size * sequence_config.prefill_length * lm_config.hidden_size * lm_config.hidden_size

        attention_prefill_ops = q_mul_k_prefill_ops + score_mul_v_prefill_ops 

        # decoding 
        q_mul_k_decoding_ops = sequence_config.batch_size * (1 * lm_config.per_head_size * sequence_config.total_length * lm_config.num_attention_heads)
        score_mul_v_decoding_ops = sequence_config.batch_size * (1 * sequence_config.total_length * lm_config.per_head_size * lm_config.num_attention_heads)
        # o_proj_decoding_ops = sequence_config.batch_size * 1 * lm_config.hidden_size * lm_config.hidden_size

        attention_decoding_ops = q_mul_k_decoding_ops + score_mul_v_decoding_ops

        return {'layer_decoding_attention_ops':attention_decoding_ops , 'layer_prefill_attention_ops':attention_prefill_ops}

    def get_layer_output_proj_ops(self)->dict[str,int]:
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config
        o_proj_prefill_ops = sequence_config.batch_size * sequence_config.prefill_length * lm_config.hidden_size * lm_config.hidden_size
        o_proj_decoding_ops = sequence_config.batch_size * 1 * lm_config.hidden_size * lm_config.hidden_size

        return {'layer_decoding_output_projection_ops':o_proj_decoding_ops,'layer_prefill_output_projection_ops':o_proj_prefill_ops}

    def get_layer_ffn_ops(self)->dict[str,int]:
        lm_config,data_type_config,sequence_config  = self.inference_config.lm_config,self.inference_config.data_type_config,self.inference_config.sequence_config

        if lm_config.model_type == 'llama':
            # three MLP matrix
            ffn_prefill_ops = sequence_config.batch_size * 3 * sequence_config.prefill_length * lm_config.hidden_size * lm_config.intermediate_size
            ffn_decoding_ops = sequence_config.batch_size * 3 * 1 * lm_config.hidden_size * lm_config.intermediate_size 
        elif lm_config.model_type =='gpt':
            ffn_prefill_ops = sequence_config.batch_size * 3 * sequence_config.prefill_length * lm_config.hidden_size * lm_config.intermediate_size
            ffn_decoding_ops = sequence_config.batch_size * 3 * 1 * lm_config.hidden_size * lm_config.intermediate_size
        elif lm_config.model_type == 'qwen2':
            ffn_prefill_ops = sequence_config.batch_size * 3 * sequence_config.prefill_length * lm_config.hidden_size * lm_config.intermediate_size
            ffn_decoding_ops = sequence_config.batch_size * 3 * 1 * lm_config.hidden_size * lm_config.intermediate_size 
        else:
            assert False

        return {'layer_decoding_ffn_ops':ffn_decoding_ops, 'layer_prefill_ffn_ops':ffn_prefill_ops}



    def get_layer_mlp_ops(self)->dict[str,int]:
        layer_qkv_projection_info = self.get_layer_qkv_proj_ops()
        layer_ffn_info = self.get_layer_ffn_ops()
        layer_output_projection_info  = self.get_layer_output_proj_ops()

        layer_mlp_decoding_ops = layer_qkv_projection_info['layer_decoding_qkv_projection_ops'] + \
                                    layer_ffn_info['layer_decoding_ffn_ops'] + \
                                    layer_output_projection_info['layer_decoding_output_projection_ops']
        
        layer_mlp_prefill_ops = layer_qkv_projection_info['layer_prefill_qkv_projection_ops'] + \
                                    layer_ffn_info['layer_prefill_ffn_ops'] + \
                                    layer_output_projection_info['layer_prefill_output_projection_ops']
        
        return {
            'layer_decoding_mlp_ops':layer_mlp_decoding_ops,
            'layer_prefill_mlp_ops':layer_mlp_prefill_ops
        }




class TransferCalc:
    def __init__(self,inference_config:InferenceConfig):
        self.inference_config = inference_config

    def get_layer_transfer_size_breakdown(self):
        # output linear + ffn + RMSNorm 
        pass 
        
        


class ResourceCalc:
    def __init__(self):
        pass

    


