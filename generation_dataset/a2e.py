# Visual Factors

import pandas as pd
import base64
from openai import OpenAI


def rebuild_path(row):
    base_name = row["data_path"].split("/")[-1].replace(".JPEG", "")
    
    new_path = f"../misclassify/val50/{row['label_index']}_{row['label']}/{base_name}_{row['predicted_index']}_{row['predicted']}.JPEG"
    return new_path


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def category(gt, mis, caption, objects):
    gt = ' '.join(gt.split('_'))
    mis = ' '.join(mis.split('_'))

    system = """You are an AI assistant that analyzes why a classification model misclassified an image.

You are given:
An image that was misclassified by a model.
The ground-truth label (GT) for the image.
The predicted label (MIS) that the model incorrectly assigned.
A caption describing the image.
A list of objects detected in the image.

Your task is to evaluate how much each of the following five factors may have contributed to the misclassification.

Return a confidence score between 0.00 and 1.00 for each factor:
A score close to 1.00 means you are highly confident that the factor significantly contributed to the misclassification.
A score close to 0.00 means the factor likely did not contribute.
Use intermediate scores if there is some uncertainty.

Answer all five questions in the given order, separated by commas, and do not include any extra text or explanation.
Return your five answers as a comma-separated list of floating-point numbers between 0.0 and 1.0.
If you are unsure, you must still return exactly six numbers, based on your best judgment.
The five factors are:

1. Visual resemblance: A score close to 1.00 means that an object in the image, although not actually the predicted label (MIS), closely resembles the MIS
2. Occlusion or low visibility: A score close to 1.00 means that the ground-truth object (GT) is hard to recognize
3. Contextual confusion: A score close to 1.00 means that the background is confuse or the overall scene context is associated with the predicted label (MIS)
4. Inclusion of predicted label: A score close to 1.00 means that the predicted label (MIS) is actually present in the image
5. Image corruption or quality issues: A score close to 1.00 means that the image quality is poor"""

    text_input = f"""You are given the following information about a misclassified image:
Ground-truth label (GT): {gt}
Predicted label (MIS): {mis}
Caption describing the image: {caption}
Detected objects in the image: {objects}

Factors to evaluate:

1. Visual resemblance: Does any object in the image closely resemble the {mis} in shape, color, texture, or other visual features?
2. Occlusion or low visibility: Was the {gt} partially occluded, too small, far away, or otherwise difficult to clearly perceive?
3. Contextual confusion: Could background confusion or scene context have misled the model into predicting {mis} instead of {gt}?
4. Inclusion of predicted lable (MIS): Does the image include the {mis}?
5. Image corruption or quality issues: Was the image difficult to interpret due to low quality issues like blurriness, dark, pixelation, poor lighting, or visual artifacts?

If uncertain, make your best estimate but still return exactly five scores.

Following are examples for your information:
Example 1:

Ground-truth label (GT): pickelhaube
Predicted label (MIS): tricycle
Caption describing the image: an old photo of a man in uniform holding a bike
Detected objects in the image: Bicycle wheel, Bicycle, Wheel, Man, Tire, Tree, Clothing
Answer: 0.78, 0.02, 0.28, 0.18, 0.83

Example 2:

Ground-truth label (GT): chain saw
Predicted label (MIS): lawn mower
Caption describing the image: a room with a lot of tools and a painting easel
Detected objects in the image: 
Answer: 0.64, 0.69, 0.91, 0.92, 0.20

Example 3:

Ground-truth label (GT): great white shark
Predicted label (MIS): falgpole
Caption describing: A woman stands inside the open mouth of a giant shark sculpture.
Detected objects in the image: Waman, Sculpture, Plants
Answer: 0.5, 0.41, 0.43, 0.01, 0.98"""

    return system, text_input


if __name__=='__main__':
    API_KEY = ""

    client = OpenAI(api_key=API_KEY)


    df_train = pd.read_csv('train_final_updated.csv', encoding='latin1')
    # df_test = pd.read_csv('test_final.csv')

    train_path = df_train.apply(rebuild_path, axis=1).tolist()
    # train_question = [q.split(".")[0] + "." for q in df_train["question"].to_list()]
    train_gt = df_train["label"].to_list()
    train_mis = df_train["predicted"].to_list()
    train_score = df_train["confidence_score"].to_list()
    train_caption = df_train["caption"].to_list()
    train_category = df_train["selected_categories"].to_list()
    train_object = df_train["detected_objects"].to_list()
    train_answer = df_train["answer"].to_list()

    # test_path = df_test.apply(rebuild_path, axis=1).tolist()
    # # test_question = [q.split(".")[0] + "." for q in df_test["question"].to_list()]
    # test_gt = df_test["label"].to_list()
    # test_mis = df_test["predicted"].to_list()
    # test_caption = df_test["caption"].to_list()
    # test_object = df_test["detected_objects"].to_list()
    # test_score = df_test["confidence_score"].to_list()
    # test_category = df_test["selected_categories"].to_list()
    # test_answer = df_test["answer"].to_list()


    def generate_response(mode_path, mode):
        answer_list = []

        try:
            for i, image_path in enumerate(mode_path):
                if not pd.isna(train_score[i]):
                    continue

                base64_image = encode_image(image_path)

                if mode == "train":
                    # if len(mode_path) != len(train_question):
                    #     raise ValueError(f"mode_path and {mode} do not match.")
                    system, text_input = category(train_gt[i], train_mis[i], train_caption[i], train_object[i])
                # if mode == "test":
                #     system, text_input = category(test_gt[i], test_mis[i], test_caption[i], test_object[i])

                response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                "role": "system",
                                "content": system
                                },
                                {
                                "role": "user",
                                "content": [
                                    {
                                    "type": "text",
                                    "text": text_input,
                                    },
                                    {
                                    "type": "image_url",
                                    "image_url": {
                                        "url":  f"data:image/jpeg;base64,{base64_image}"
                                    },
                                    },
                                ],
                                }
                            ],
                            )

                answer = response.choices[0].message.content


                if answer is None or "sorry" in answer or "Sorry" in answer or "unable" in answer or "I can't" in answer or "I'm" in answer or "apologize" in answer or "I don't" in answer or "cannot" in answer or '�' in answer or 'Ã' in answer or 'Â' in answer or '¤' in answer or '¿' in answer:
                    x = -1
                    while True:
                        x += 1
                        if x == 10:
                            answer = ''
                            break

                        response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                        "role": "system",
                                        "content": system
                                        },
                                        {
                                        "role": "user",
                                        "content": [
                                            {
                                            "type": "text",
                                            "text": text_input,
                                            },
                                            {
                                            "type": "image_url",
                                            "image_url": {
                                                "url":  f"data:image/jpeg;base64,{base64_image}"
                                            },
                                            },
                                        ],
                                        }
                                    ],
                                    )

                        answer = response.choices[0].message.content
                        if answer is not None:
                            answer = answer.replace("’", "'")

                        if answer is not None and "sorry" not in answer and "sorry" not in answer and "Sorry" not in answer and "unable" not in answer and "I can't" not in answer and "I'm" not in answer and "apologize" not in answer and "I don't" not in answer and "cannot" not in answer and '�' not in answer and 'Ã' not in answer and 'Â' not in answer and '¤' not in answer and '¿' not in answer:
                            break

                if len(answer.split(',')) != 5:
                    answer = '0.0, 0.0, 0.0, 0.0, 0.0'


                answer_list.append(answer)


                print(f"{str(int(i)+2)}\n{answer}")
                print()

            
            return answer_list
        
        except Exception as e:
            import traceback


            print(traceback.format_exc())
            
            with open("captions.txt", "w", encoding="utf-8") as f:
                for item in answer_list:
                    f.write(item + "\n")


    train_answer_test = generate_response(train_path, mode="train")
    
    try:
        df_train['confidence_score'] = train_answer_test
        df_train.to_csv('train_mis.csv', index=False)
    except:
        with open("train_category.txt", "w") as file:
            file.write(str(train_answer_test))


    # test_scores = generate_response(test_path, mode="test")

    # try:
    #     df_test['confidence_score'] = test_scores
    #     df_test.to_csv('test_step1.csv', index=False)
    # except:
    #     with open("test_category.txt", "w") as file:
    #         file.write(str(test_scores))
