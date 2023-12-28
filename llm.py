from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, Accelerator
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
import torch
from abc import ABC, abstractmethod

torch.set_default_device("cuda")

bot_model = None

class BotModel(ABC):
    @abstractmethod
    def preprocess(self, msg):
        pass
        
    @abstractmethod
    def postprocess(self, output):
        pass

    @abstractmethod
    def generate(self, prompt):
        pass

    @abstractmethod
    def get_help_msg(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass        




class Zephyr7bModel(BotModel):
    def __init__(self):
        model_path = 'HuggingFaceH4/zephyr-7b-beta'
        weights_path = '/home/logtenk/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/dc24cabd13eacd3ae3a5fe574bd645483a335a4a'

        config = AutoConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        self.model = load_and_quantize_model(model, weights_location=weights_path, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def preprocess(self, hist):
        system_prefix = '<|system|>\n'
        user_prefix = '</s>\n<|user|>\n'
        assistant_prefix = '</s>\n<|assistant|>\n'

        hist_parts = hist.split('\n----\n')
        if len(hist_parts) == 3:
            sys, usr, ass = hist_parts
        elif len(hist_parts) == 2:
            sys, usrass = hist_parts
            usrass_lines = usrass.split('\n')
            usr = '\n'.join(usrass_lines[:-1])
            ass = usrass_lines[-1]
        else: 
            print(f"Should have one or two separator(s) in the text! Found {len(hist_parts)-1}.")
            raise SyntaxError()

        prompt = f'{system_prefix}{sys}{user_prefix}{usr}{assistant_prefix}{ass}'
        return prompt, ass

    def postprocess(self, text, ass):
        assistant_prefix = '<|assistant|>\n'
        post_assitant = text.split(assistant_prefix)[-1]
        out_list = post_assitant.split(ass)[1:]
        out = ass.join(out_list)
        if out.endswith(self.tokenizer.eos_token):
            out = out.rstrip(self.tokenizer.eos_token)
        return out

    def generate(self, msg):
        print('parsing input file...')
        prompt, ass = self.preprocess(msg)
        print(f'prompt generated: \n {prompt}\n\n')
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False, add_special_tokens=False).to('cuda') 
        print('generating outputs...')
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=516,
            temperature=0.8,
            do_sample=True
        )
        text = self.tokenizer.batch_decode(outputs)[0]
        print('postprocessing output...')
        out = self.postprocess(text, ass)
        return out

    def get_help_msg(self):
        return "Trial and error"

    def __repr__(self):
        return 'Zephyr-7b'     

model_dict = {'zephyr-7b': Zephyr7bModel}
