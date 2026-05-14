# PROMPT_EN: translate ZH reports to English. PROMPT_ZH: normalize native Chinese reports (strings are model-facing).
PROMPT_EN = """
Your task is to convert the following Chinese medical report into English, summarizing and standardizing the information in a specific format.

To achieve this, please follow these guidelines:

1. Identify and summarize the observations noted by the doctor for the FINDINGS section.
2. Summarize the doctor’s diagnoses and conclusions for the IMPRESSION section. Only reference specific FINDINGS if there are abnormalities. If there are no abnormalities, do not mention it in the IMPRESSION.
3. Translate the summarized information accurately into English, maintaining the original medical terminology and clinical details.
4. Standardize the format to include sections titled FINDINGS and IMPRESSION.
5. When summarizing, identify and exclude statements in the report that compare with previous CT or imaging studies, and do not include them in the FINDINGS or IMPRESSION sections.
6. When summarizing, identify and exclude statements in the report that contain specific measurements such as length, area, etc., and do not include them in the FINDINGS or IMPRESSION sections.
7. Wrap the standardized content in ```output
your response
``` delimiters.
Here is the Chinese medical report to be summarized, translated, and standardized:
```input
{content}
```

Below is the example format that your output should follow:

Example Output 1:
```output
 FINDINGS:

 Diffuse bilateral patchy airspace opacities, most pronounced in the left lower
 lung zone may reflect pulmonary edema although superimposed infection cannot
 be excluded.  The size of the cardiomediastinal silhouette is significantly
 enlarged.  No discrete pneumothorax identified.

 IMPRESSION:

 Diffuse bilateral airspace opacities with a markedly enlarged
 cardiomediastinal silhouette may reflect pulmonary edema.

 No discrete pneumothorax identified.
 ```

 Example Output 2:
 ```output
 FINDINGS:  

 No focal consolidation, pleural effusion, or pneumothorax is seen. 
 Mild peribronchial cuffing and interstitial prominence suggests small airways
 disease.  Heart and mediastinal contours are within normal limits.

 IMPRESSION:  

 Mild small airways disease.
 ```

  Example Output 3:
 ```output
 FINDINGS: 

 Diffuse bilateral patchy airspace opacities, most pronounced in the left lower
 lung zone may reflect pulmonary edema although superimposed infection cannot
 be excluded.  The size of the cardiomediastinal silhouette is significantly
 enlarged.  No discrete pneumothorax identified.

 IMPRESSION: 

 Diffuse bilateral airspace opacities with a markedly enlarged
 cardiomediastinal silhouette may reflect pulmonary edema.

 No discrete pneumothorax identified.
 ```

Please generate the translated and standardized medical report according to the guidelines above.
"""

PROMPT_ZH = """
你的任务是将以下中文医学报告提取和总结相关信息，并标准化为FINDINGS和IMPRESSION部分。请按照以下指南进行：

1. 提取医生观察到的现象并总结为FINDINGS部分。
2. 总结医生针对现象做出的诊断结论为IMPRESSION部分。如果有异常情况，可以引用FINDINGS部分的相关现象，如果正常情况则不用特别提及。
3. 保持医学术语和临床细节的准确性，将总结后的内容保持为中文。
4. 标准化格式应包括FINDINGS和IMPRESSION部分。
5. 在总结时，请识别并排除报告中**关于之前CT或影像的比较语句**，不将其包含在FINDINGS或IMPRESSION中。
6. 在总结时，识别并排除报告中带有**具体长度、面积等单位**的语句，不将其包含在FINDINGS或IMPRESSION中。
7. 将标准化内容用 ```output
your response
``` 分隔符包围。

以下是需要提取和总结的中文医学报告：
```input
{content}
```

下面是输出格式示例：

示例输出1：
```output
FINDINGS:

双侧弥漫性斑片状空气间隙混浊，最突出于左下肺区，可能反映肺水肿，尽管无法排除重叠感染。心中隔影像大小显著增大。未见明确气胸。

IMPRESSION:

双侧弥漫性空气间隙混浊与显著增大的心中隔影像可能反映肺水肿。未见明确气胸。
 ```

示例输出2:
 ```output
FINDINGS:

未见明显局灶性实变、胸腔积液或气胸。轻微支气管周围袖套和间质隆起提示小气道疾病。心脏和纵隔轮廓在正常范围内。

IMPRESSION:

轻度小气道疾病。
 ```

示例输出3:
 ```output
FINDINGS:

双侧弥漫性斑片状空气间隙混淆，最明显于左下肺区，可能反映肺水肿，尽管无法排除重叠感染。心中隔影像大小显著增大。未见明确气胸。

IMPRESSION:

双侧弥漫性空气间隙混淆与显著增大的心中隔影像可能反映肺水肿。未见明确气胸。
 ```

请根据以上指南生成标准化的中文医学报告。
"""
