PROMPT_EN = """
Your task is to convert the following Chinese chest X-ray medical report into a standardized English radiology report.

Please strictly follow these guidelines:

1. Extract and summarize the objective imaging observations into the FINDINGS section.
   - FINDINGS must contain only descriptive radiological observations.
   - Do NOT include any diagnostic conclusions in FINDINGS.

2. Extract and summarize the diagnostic interpretation into the IMPRESSION section.
   - IMPRESSION should reflect the clinician’s overall diagnostic assessment.
   - Do NOT introduce any new diagnoses that are not explicitly stated in the original report.

3. Translate the content accurately into English using standard radiology terminology.
   - Avoid literal word-by-word translation.
   - Use clinically accepted expressions (e.g., “increased bronchovascular markings” instead of “lung texture thickened”).

4. Preserve all expressions of uncertainty (e.g., “suggestive of”, “cannot exclude”, “likely”, “consider”).
   - Do NOT convert uncertain statements into definitive conclusions.

5. If the original Chinese report contains only FINDINGS or only IMPRESSION, do NOT fabricate the missing section.
   - Leave the missing section empty if necessary.

6. Standardized Output Format (strict):
FINDINGS:
<content>

IMPRESSION:
<content>

7. Wrap your final output strictly within:
```output
<your standardized report>
 ```
8.	Output ONLY the standardized English report.
    - Do NOT include any explanation, notes, or additional commentary.
Here is the Chinese medical report to be processed:
```input
{content}
```
    - If the original content is ambiguous, incomplete, or poorly structured, you must translate it faithfully without attempting to correct or improve it.
here is the output format example:

output sample 1:
```output
FINDINGS:

Increased and thickened bronchovascular markings in both lungs without significant consolidation. No nodular opacities at the hilum. Normal cardiac size. Eggshell-like hyperdense calcification at the aortic knob. Sharp costophrenic angles and smooth diaphragmatic surfaces

IMPRESSION:

Atherosclerosis of the aorta.
 ```
"""


