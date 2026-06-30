# Dataset Generation

<p align="center">
  <img width="800" alt="Image" src="https://github.com/user-attachments/assets/79fb2eb9-cafe-41de-a90d-6d01b72d39a8" />
</p>

## Content
- [Mislabeling](#mislabeling)
- [Label Ambiguity](#label-ambiguity)
- [Remain Categories](#remain-category)
- [Generate Explanation](#generate-explanation)
- [Examples](#examples)

## Mislabeling

This step identifies whether a misclassified sample is caused by an incorrect ground-truth label (Mislabeling).<br>
Before running the scripts, set your OpenAI API key in the source code or as an environment variable.

Run the following scripts in order.

```
python find_mis.py

python find_mis2.py

python find_mis3.py
```

1. Generate an image caption, detected objects, and a confidence score indicating whether the ground-truth label exists in the image using GPT-4o.
2. To improve reliability, the prompt is executed multiple times using **Self-Consistency**.
3. The confidence score that appears **most frequently** across all responses is selected as the final result (majority vote).
4. If the final confidence score is **greater than 0.6**, the sample is assigned to the **(g) Mislabeling** category. Otherwise, the sample proceeds to the **Label Ambiguity** step.

## Label Ambiguity

This step determines whether the ground-truth label and the predicted label are semantically or taxonomically ambiguous.

```
python label_ambiguity.py
```

## Remain Categories

This step evaluates the remaining image-related causes (a to e) of misclassification after **(g) Mislabeling** and **(f) Label Ambiguity** have been excluded.

```
python a2e.py
```

The script outputs the selected categories for each sample. If no category satisfies the threshold, the output is automatically assigned to **Category (h) Inherent Model Failure**.

## Generate Explanation

This step generates a natural language explanation for why the image was misclassified based on the categories selected in the previous steps.

```
ptyhon generate_explanation1.py
python generate_explanation2.py
```

## Examples

### Category

<p align="center">
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/ae665890-484f-425a-b167-d72ed8a6a71f" />
</p>

### Explanation

<img width="1442" height="2155" alt="Image" src="https://github.com/user-attachments/assets/bc5a4c2b-74d2-432d-bb0d-382f91ffe2d0" />
