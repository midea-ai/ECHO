import json
from tqdm import tqdm
from chatbot.prompts import PROMPT_EN
from multiprocessing.pool import ThreadPool

src_path = "/data/hanxiao36/projects/RaTEScore/wandong_test/7b_zh.json"
# src_path = "/data/hanxiao36/projects/RaTEScore/wandong_test/32b_zh.json"
dst_path = "/data/hanxiao36/projects/RaTEScore/wandong_test/7b_zh_to_en.json"

engine = "api"  # or "api"

if engine == "api":
    from chatbot.api_chatbot import ChatBot

    THREAD = 16
    api_key = "your-api-key"  # for api_chatbot
    baichuan_llm = ChatBot(api_key=api_key)
else:
    from chatbot.vllm_chatbot import ChatBot

    THREAD = 4
    baichuan_path = "/data/share/250911/models/Baichuan-M2-32B"  # for vllm_chatbot
    baichuan_llm = ChatBot(model_name=baichuan_path)


# ZH -> EN translation pipeline
def _func(item):
    model_response = item["response"].replace("Finding:", "所见:").replace("Impression:", "结论:")
    gt_findings = item["findings"]
    gt_impression = item["impression"]
    golden = f"所见: {gt_findings} 结论: {gt_impression}"

    try:
        response = baichuan_llm.chat(model_response, PROMPT_EN)
        pred_finding, pred_impression = baichuan_llm.extract_info(response)
        response = f"Findings: {pred_finding} Impression: {pred_impression}"
    except Exception as e:
        print(e)
        return False, None

    try:
        gt = baichuan_llm.chat(golden, PROMPT_EN)
        gt_finding, gt_impression = baichuan_llm.extract_info(gt)
        # golden = f"Findings: {gt_finding} Impression: {gt_impression}"
    except Exception as e:
        print(e)
        return False, None

    result = {
        "response": response,
        "impression": gt_impression,
        "findings": gt_finding,
    }

    return True, result


if __name__ == '__main__':

    with open(src_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    results = []
    fail_nums = 0

    with ThreadPool(THREAD) as pool:
        with tqdm(total=len(data), desc="Processing Rows") as pbar:
            for result in pool.imap_unordered(_func, data):
                flag, item = result
                if flag:
                    results.append(item)
                else:
                    fail_nums += 1
                pbar.update(1)

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
