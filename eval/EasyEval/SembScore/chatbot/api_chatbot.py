from openai import OpenAI


class ChatBot:
    def __init__(self, base_url="https://aimpapi.midea.com/t-aigc/baichuan-m2-32b-1/v1", api_key=""):
        self.model = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, text, prompt=None):
        if prompt is not None:
            prompt = prompt.format(content=text)
        else:
            prompt = text
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        resp = self.model.chat.completions.create(
            model="Baichuan-M2",
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            extra_body={
                "chat_template_kwargs": {"thinking_mode": "off"},
                "add_generation_prompt": True}
        )
        return resp.choices[0].message.content

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
