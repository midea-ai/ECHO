import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Force vLLM V0 engine (avoid V1 threading issues)
os.environ['VLLM_USE_V1'] = '0'


class ChatBot:
    def __init__(self, model_name,tensor_parallel_size=1):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            dtype="bfloat16",
            enforce_eager=True
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
        )

    def chat(self, text, prompt=None):
        if prompt is not None:
            prompt = prompt.format(content=text)
        else:
            prompt = text
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking_mode='off',
        )
        outputs = self.model.generate([text], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def batch_chat(self, text_list, prompt=None):
        """
        Batch generation (avoids multi-thread CUDA issues).

        Args:
            text_list: input strings
            prompt: optional template with {content}

        Returns:
            list of response strings
        """
        prompts = []
        for text in text_list:
            if prompt is not None:
                formatted_prompt = prompt.format(content=text)
            else:
                formatted_prompt = text
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt}
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode='off',
            )
            prompts.append(formatted_text)
        
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=True)

        responses = [output.outputs[0].text for output in outputs]
        return responses

    @staticmethod
    def extract_info(response):
        finding, impression = "", ""

        response = response.replace("```output", "```").replace("```Output", "```").replace("```OUTPUT", "```")

        response_list = response.split("```")
        for line in response_list:
            if "FINDINGS" and "IMPRESSION" in line:
                try:
                    finding, impression = line.split("IMPRESSION:")
                    finding = finding.split("FINDINGS:")[-1].strip(" \n:")
                    impression = impression.strip(" \n:")
                except Exception as e:
                    print(e)
                    break
                finally:
                    break
        return finding, impression
