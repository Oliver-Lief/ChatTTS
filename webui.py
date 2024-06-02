import random
import argparse

import torch
import gradio as gr
import numpy as np

import ChatTTS


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {"__type__": "update", "value": new_seed}


def generate_audio(
    text,
    temperature,
    top_P,
    top_K,
    audio_seed_input,
    text_seed_input,
    refine_text_flag,
    par,
):
    # on_text_change(total_text_number_box, text)
    cut_word_num = 120

    torch.manual_seed(audio_seed_input)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        "spk_emb": rand_spk,
        "temperature": temperature,
        "top_P": top_P,
        "top_K": top_K,
    }

    # # 根据文本长度生成三种风格的数量
    # num_weight = int(len(text) / cut_word_num) + 1
    # num_oral = 5 * num_weight
    # num_laugh = 2 * num_weight
    # num_break = 5 * num_weight
    # print("数量:" + str(num_oral) + "," + str(num_laugh) + "," + str(num_break))
    # # 生成相应的参数
    # params_refine_text = {
    #     "prompt": f"[oral_{num_oral}][laugh_{num_laugh}][break_{num_break}]"
    # }

    params_refine_text = {"prompt": par}

    torch.manual_seed(text_seed_input)

    # 替换特定符号为空
    text = (
        text.replace("\n\n", ",")
        .replace(":", ",")
        .replace("：", ",")
        .replace("[", ",")
        .replace("]", ",")
    )

    print("文本长度:" + str(len(text)) + "\n")

    merged_audio = np.array([])

    if refine_text_flag:
        if len(text) > cut_word_num:
            # 分段处理文本
            segments = [
                text[i : i + cut_word_num] for i in range(0, len(text), cut_word_num)
            ]
            total_segments = len(segments)
            # generated_wav = []
            for i, segment in enumerate(segments):
                print(
                    "正在处理第{}个片段（共{}个）：{}\n".format(
                        i + 1, total_segments, segment
                    )
                )
                inferred_text = chat.infer(
                    segment,
                    skip_refine_text=False,
                    refine_text_only=True,
                    params_refine_text=params_refine_text,
                    params_infer_code=params_infer_code,
                    do_text_normalization=False,
                )
                wav = chat.infer(
                    inferred_text,
                    skip_refine_text=True,
                    params_refine_text=params_refine_text,
                    params_infer_code=params_infer_code,
                    do_text_normalization=False,
                )

                audio_data = np.array(wav[0]).flatten()
                merged_audio = np.append(merged_audio, audio_data)

                print("文本:" + str(inferred_text) + "\n")

        else:
            text = chat.infer(
                text,
                skip_refine_text=False,
                refine_text_only=True,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
                do_text_normalization=False,
            )
            wav = chat.infer(
                text,
                skip_refine_text=True,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
                do_text_normalization=False,
            )
            merged_audio = np.array(wav[0]).flatten()
    # print(type(text))
    # text = ["".join(sublist) for sublist in text]
    # text = ["".join(text)]
    # print(type(text))
    # print("\n\n文本:" + str(text) + "\n")

    # print(type(audio_data))
    # audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, merged_audio), text_data]


def on_text_change(text_input):
    return str(len(text_input))


def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS Webui")
        gr.Markdown(
            "ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)"
        )

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"
        text_input = gr.Textbox(
            label="Input Text",
            lines=4,
            placeholder="Please Input Text...",
            value=default_text,
        )

        with gr.Row():
            refine_text_checkbox = gr.Checkbox(label="Refine text", value=True)
            temperature_slider = gr.Slider(
                minimum=0.00001,
                maximum=1.0,
                step=0.00001,
                value=0.3,
                label="Audio temperature",
            )
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P"
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=20, step=1, value=20, label="top_K"
            )

        with gr.Row():
            audio_seed_input = gr.Number(value=2424, label="Audio Seed")
            generate_audio_seed = gr.Button("\U0001F3B2")
            text_seed_input = gr.Number(value=42, label="Text Seed")
            generate_text_seed = gr.Button("\U0001F3B2")
            total_text_number_box = gr.TextArea(
                lines=1,
                label="字符数",
                value=str(len(text_input.value)),
            )
            default_par = "[oral_2][laugh_0][break_6]"
            para_box = gr.Textbox(value=default_par)

        text_input.change(
            on_text_change, inputs=text_input, outputs=total_text_number_box
        )

        generate_button = gr.Button("Generate")

        text_output = gr.Textbox(label="Output Text", interactive=False)
        audio_output = gr.Audio(label="Output Audio")
        generate_audio_seed.click(generate_seed, inputs=[], outputs=audio_seed_input)

        generate_text_seed.click(generate_seed, inputs=[], outputs=text_seed_input)

        generate_button.click(
            generate_audio,
            inputs=[
                text_input,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                audio_seed_input,
                text_seed_input,
                refine_text_checkbox,
                para_box,
            ],
            outputs=[audio_output, text_output],
        )

    parser = argparse.ArgumentParser(description="ChatTTS demo Launch")
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="Server name"
    )
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    print("loading ChatTTS model...")
    global chat
    chat = ChatTTS.Chat()

    chat.load_models(source="local", local_path="./pzc163/chatTTS")

    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
