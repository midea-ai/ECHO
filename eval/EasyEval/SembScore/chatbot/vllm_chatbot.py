from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class ChatBot:
    def __init__(self, model_name):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = LLM(
            model=model_name,
            tensor_parallel_size=1,
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
