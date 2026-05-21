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
  

def find_mis(gt, caption, objects):
    gt = ' '.join(gt.split('_'))

    system = """You are a vision-language expert tasked with determining whether the assigned label for an image is appropriate. You will be given an image, the assigned label, a caption, and a list of detected objects. 
The caption and detected objects may contain errors, so your primary reference should be the image itself. Use the caption and detected objects only as supplementary clues. 
If the assigned label clearly appears in the image—either visually or is clearly supported by the caption or objects—respond with "Yes".
If the assigned label is not visibly present in the image and cannot be reasonably inferred from the caption or detected objects, respond with "No".
Output only "Yes" or "No". Do not include any explanation or justification."""

    text_input = f"""Image label: {gt}
Image caption: {caption}
Detected objects: {objects}

Question: Is the assigned label appropriate for this image?
Answer (Yes or No): """

    return system, text_input


if __name__=='__main__':
    API_KEY = ""

    client = OpenAI(api_key=API_KEY)


    df_train = pd.read_csv('train_final.csv')
    # df_test = pd.read_csv('test_final.csv')

    train_path = df_train.apply(rebuild_path, axis=1).tolist()
    train_gt = df_train["label"].to_list()
    train_mis = df_train["predicted"].to_list()
    train_caption = df_train["caption"].to_list()
    train_objects = df_train["detected_objects"].to_list()
    train_score = df_train["confidence_score"].to_list()
    train_category = df_train["selected_categories"].to_list()
    train_answer = df_train["answer"].to_list()


    # test_path = df_test.apply(rebuild_path, axis=1).tolist()
    # test_gt = df_test["label"].to_list()
    # test_mis = df_test["predicted"].to_list()
    # test_caption = df_test["caption"].to_list()
    # test_objects = df_test["detected_objects"].to_list()
    # test_score = df_test["confidence_score"].to_list()
    # test_category = df_test["selected_categories"].to_list()
    # test_answer = df_test["answer"].to_list()
    

    def generate_response(mode_path, mode):
        answer_list = []

        try:
            for i, image_path in enumerate(mode_path):
                if train_category[i] != "g":
                    answer_list.append(train_category[i])
                    continue
                # if not pd.isna(test_score[i]):
                #     answer_list.append('')
                #     continue


                base64_image = encode_image(image_path)

                if mode == "train":
                    system, text_input = find_mis(train_gt[i], train_caption[i], train_objects[i])
                # elif mode == "test":
                #     system, text_input = find_mis(test_gt[i])

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


                print(f"{str(int(i)+2)}\n{answer}")
                print()
                answer_list.append(answer)

            
            return answer_list
        
        except Exception as e:
            import traceback


            print(traceback.format_exc())
            
            with open("captions.txt", "w", encoding="utf-8") as f:
                for item in answer_list:
                    f.write(item + "\n")


    is_mis = generate_response(train_path, mode="train")

    try:
        df_train['final_mis'] = is_mis
        df_train.to_csv('train_final_mis.csv', index=False)
    except:
        with open("train_mis.txt", "w") as file:
            file.write(is_mis)

    # is_mis = generate_response(test_path, mode="test")

    # try:
    #     df_test['final_mis'] = is_mis
    #     df_test.to_csv('test_mis_final.csv', index=False)
    # except:
    #     with open("train_mis.txt", "w") as file:
    #         file.write(is_mis)