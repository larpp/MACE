# Generate Explanantions

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
  

def get_answer(gt, mis, caption, object, confidence, category):
    gt = ' '.join(gt.split('_'))
    mis = ' '.join(mis.split('_'))

    system = """You are an AI assistant that explains why a classification model misclassified an image.

Your task is to generate a natural language explanation of why the misclassification likely occurred based on the following information:
Misclassified image
The ground-truth label (GT)
The predicted label (MIS)
A list of categories
A caption describing the image
A list of detected objects in the image
Confidence scores for nine diagnostic categories (a to i)
A list of selected categories, which are the categories identified as the main contributors to the misclassification based on the confidence scores

Instructions:
Focus your explanation on the selected categories.
Explain specifically (but not too long) what in the image may have caused confusion, using the given information.
Avoid simply rephrasing the category definitions. Be as specific and descriptive as possible, identifying particular objects, background elements, or image characteristics that may have contributed to the misclassification.
Do not mention the category letters (a, b, etc.) or numeric scores directly in your output.
Always review all available information to check whether the selected categories properly reflect the true cause of the misclassification. If necessary, provide a corrected or additional explanation based on your judgment.
Don't repeat the same thing, but say it in a specific but concise sentence."""

    text_input = f"""Ground-truth label (GT): {gt}
Predicted label (MIS): {mis}
Caption: {caption}
Detected objects: {object}
Confidence score: {confidence}
Selected categories: {category}

Categories:
a. Visual resemblance: Does any object in the image closely resemble the {mis} in shape, color, texture, or other visual features?
b. Occlusion or low visibility: Was the {gt} partially occluded, too small, far away, or otherwise difficult to clearly perceive?
c. Contextual confusion: Could background confusion or scene context have misled the model into predicting {mis} instead of {gt}?
d. Inclusion of predicted lable (MIS): Does the image include the {mis}?
e. Image corruption or quality issues: Was the image difficult to interpret due to low quality issues like blurriness, dark, pixelation, poor lighting, or visual artifacts?
f. Label ambiguity: The {gt} and {mis} refer to the same object or concept with slight variations in terminology, plurality, or specificity
g. Mislabeling: The {gt} is actually missing from the image
h. Inherent model failure: The {gt} is clearly present and distinguishable in the image or there are no visual or contextual clues in the image that could justify predicting {mis}.

Please generate a detailed (but not too long) natural language explanation of why this misclassification likely occurred, using the given information and describing specific clues from the image.

Following are examples for your information:
Example 1:

Ground-truth label (GT): ear
Predicted label (MIS): corn
Caption: A person holding an ear of corn.
Detected objects: hands, corn
Confidence score: [0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]
Selected categories: f

Answer: Because an 'ear (of corn)' and 'corn' look identical and are often used interchangeably.

Example 2:

Ground-truth label (GT): electric ray
Predicted label (MIS): sturgeon
Caption: A small shark swimming over a rocky bottom in an aquarium.
Detected objects: shark, rocks
Confidence score: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]
Selected categories: g

Answer: The image actually depicts a Denison's barb rather than an electric ray, indicating the ground-truth label was incorrect.

Example 3:

Ground-truth label (GT): velvet
Predicted label (MIS): clog
Caption: A pair of hands wearing decorative fingerless gloves.
Detected objects: Hands, gloves.
Confidence score: [0.3, 0.1, 0.5, 0.0, 0.1, 0.0, 0.05, 1.0]
Selected categories: h

Answer: The gloves are unmistakably made of velvet and there are no clog-like shapes or textures in the image, so the model's prediction of 'clog' reflects a model's performance issue rather than any real visual cue.

Example 4:

Ground-truth label (GT): mouse
Predicted label (MIS): desktop computer
Caption: A computer workstation with two monitors and a keyboard.
Detected objects: Monitors, keyboard, mouse, desk, computer.
Confidence score: [0.7, 0.3, 0.85, 0.95, 0.1, 0.0, 0.05, 0.0]
Selected categories: a, c, d

Answer: The model likely focused on the large monitors, keyboard, and overall computer setup while the actual mouse is small and partially hidden causing it to mistake the scene's dominant desktop hardware for a 'desktop computer' rather than recognizing the mouse.

Example 5:
Ground-truth label (GT): monastery
Predicted label (MIS): castle
Caption: A stone building surrounded by lush greenery.
Detected objects: Building, stone wall, greenery, pathway
Confidence score: [0.85, 0.1, 0.65, 0.05, 0.15, 0.1, 0.05, 0.0]
Selected categories: a, c

Answer: The misclassification likely occurred due to the thick stone walls, arched openings, and hilltop setting—hallmarks of medieval defensive architecture—for a castle, overlooking the subtle monastic features."""


    return system, text_input


if __name__=='__main__':
    API_KEY = ""

    client = OpenAI(api_key=API_KEY)


    df_train = pd.read_csv('train_final.csv')
    # df_test = pd.read_csv('test_final.csv', encoding='latin1')

    train_path = df_train.apply(rebuild_path, axis=1).tolist()
    train_gt = df_train["label"].to_list()
    train_mis = df_train["predicted"].to_list()
    train_caption = df_train["caption"].to_list()
    train_object = df_train["detected_objects"].to_list()
    train_confidence = df_train["confidence_score"].to_list()
    train_category = df_train["selected_categories"].to_list()
    train_answer = df_train["answer"].to_list()

    # test_path = df_test.apply(rebuild_path, axis=1).tolist()
    # test_gt = df_test["label"].to_list()
    # test_mis = df_test["predicted"].to_list()
    # test_caption = df_test["caption"].to_list()
    # test_object = df_test["detected_objects"].to_list()
    # test_confidence = df_test["confidence_score"].to_list()
    # test_category = df_test["selected_categories"].to_list()
    # test_answer = df_test["answer"].to_list()


    def generate_response(mode_path, mode):
        answer_list = []

        try:
            for i, image_path in enumerate(mode_path):
                if mode == "train":
                    if not pd.isna(train_answer[i]):
                        answer_list.append(train_answer[i])
                        continue
                    system, text_input = get_answer(train_gt[i], train_mis[i], train_caption[i], train_object[i], train_confidence[i], train_category[i])
                # if mode == "test":
                #     if not pd.isna(test_answer[i]):
                #         answer_list.append(test_answer[i])
                #         continue
                #     system, text_input = get_answer(test_gt[i], test_mis[i], test_caption[i], test_object[i], test_confidence[i], test_category[i])

                base64_image = encode_image(image_path)


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

                if answer is None or "sorry" in answer or "Sorry" in answer or "unable" in answer or "I can't" in answer or "I'm" in answer or "apologize" in answer or "I don't" in answer or "cannot" in answer or '�' in answer or 'Ã' in answer or 'Â' in answer or '¤' in answer or '¿' in answer or '' in answer:
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

                        if answer is not None and "sorry" not in answer and "sorry" not in answer and "Sorry" not in answer and "unable" not in answer and "I can't" not in answer and "I'm" not in answer and "apologize" not in answer and "I don't" not in answer and "cannot" not in answer and '�' not in answer and 'Ã' not in answer and 'Â' not in answer and '¤' not in answer and '¿' not in answer and '' not in answer:
                            break


                answer_list.append(answer)

                print(f"{str(int(i)+2)}\n{answer}")
                print()

            # with open(f"captions_{mode}.txt", "w", encoding="utf-8") as f:
            #     for item in answer_list:
            #         f.write(item + "\n")

            return answer_list
        
        except Exception as e:
            import traceback


            print(traceback.format_exc())
            
            with open(f"answer_{mode}.txt", "w", encoding="utf-8") as f:
                for item in answer_list:
                    f.write(item + "\n")

            # 오류 발생 후 None 반환
            # return answer_list


    train_answer_test = generate_response(train_path, mode="train")
    
    try:
        df_train['answer'] = train_answer_test
        df_train.to_csv('train_final_real.csv', index=False)
    except:
        with open("train_answer.txt", "w") as file:
            file.write(str(train_answer))


    # test_answers = generate_response(test_path, mode="test")

    # try:
    #     df_test['answer'] = test_answers
    #     df_test.to_csv('test_final_real.csv', index=False)
    # except:
    #     with open("test_answer.txt", "w") as file:
    #         file.write(str(test_answers))
