from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import shutil#用于存储文件到data
import re
import os

from prompt import ESG_prompts1
# import numpy as np
# import pandas as pd




from utils import load_model_on_gpus
# 1
model_path = "/home/kings/PycharmProjects/ChatGLM3/THUDM/chatglm3-6b"
data_base_path = 'data'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)

model = model.eval()

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

#增加判断是用户输入还是文件调用
def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values,decsion):
    if int(decsion):
        chatbot.append((parse_text(input), ""))
        for response, history, past_key_values in model.stream_chat(tokenizer, input, history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True,
                                                                    max_length=max_length, top_p=top_p,
                                                                    temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))
            yield chatbot, history, past_key_values
    else:
        chatbot.append((input,""))
        for response, history, past_key_values in model.stream_chat(tokenizer, input, history,past_key_values=past_key_values,return_past_key_values=True,max_length=max_length, top_p=top_p,temperature=temperature):
            chatbot[-1] = ('', parse_text(response))

            # chatbot[-1] = (parse_text(input), parse_text(response))

            yield chatbot, history, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None

#文件名过滤器，主要用于提取文件名称
def file_name_fileter(file_path):
    pattern = r".*/(.*)"  # 贪婪模式匹配任意字符直到"/"
    file_name = re.search(pattern, file_path).group(1)
    return file_name


#文件上传模块
def file_uploaded(upload_files:str):
    file_paths = [file.name for file in upload_files]
    for source_file_path in file_paths:# 指定要复制的源文件路径


        destination_file_path = "data/"+file_name_fileter(source_file_path)

        shutil.copy(source_file_path, destination_file_path)  # 复制文件

    return file_paths#传出所有上传文件路径，在gr.File模块展示

#得到data文件夹下面的所有文件的相对路径
def get_all_files(folder_path):
    all_files = []
    stack = [folder_path]

    while stack:
        current_path = stack.pop()
        for file_name in os.listdir(current_path):
            file_path = os.path.join(current_path, file_name)
            if os.path.isfile(file_path):
                all_files.append(file_path)
            else:
                stack.append(file_path)

    return all_files



#根据用户选择将数据模板文件，结合模板文件传入Chatglm，未完成
# def choose_datafile(choosed_file_name:str):
#     file_path = 'data/'+choosed_file_name
#     return 'Sucess'
#在这里调用文件
#更新data文件名称
def data_names_reload():
    global data_names
    for file_path in get_all_files(data_base_path):
        data_names.append(file_name_fileter(file_path=file_path))
    data_names = list(set(data_names))

#更新文件列表表单元素
def change_file_box():
    global file_box
    global data_names
    data_names_reload()
    file_box.choices = data_names
    return gr.Radio.update(choices=data_names, label="数据库ESG数据", info="选择ESG数据文件",scale=3)


file_inputs =['','','']
"""
输入：前端选择文件名
输出：列表[单独大模型E/S/G的内容 ]
"""
def file_input_to_LLM(file_path:str):
    global file_inputs

    file_path = 'data/' + file_path
    ESG_prompts = ESG_prompts1(file_path)
    file_inputs.clear()
    for file_inpute in ESG_prompts:
        file_inputs.append(file_inpute[0])
    a = gr.Textbox.update(value=file_inputs[0],visible=False)
    print(a)
    print(gr.Textbox.update(value=file_inputs[0],visible=False))
    return gr.Textbox.update(value=file_inputs[0],visible=False),gr.Textbox.update(value=file_inputs[1],visible=False),gr.Textbox.update(value=file_inputs[2],visible=False)







title_name = "基于langchain+FATE-ChatGLM3-6B的ESG报告生成系统"
with open('elemnt/main.html','r') as file:
    html_content = file.read()

html_content = html_content.replace('title_name', title_name)

data_names = []

# file_inputs = [gr.Textbox(value='', visible=False), gr.Textbox(value='', visible=False),
#                gr.Textbox(value='', visible=False)]

with gr.Blocks(theme=gr.themes.Soft(),title=title_name) as demo:
    with gr.Row():
        gr.HTML(html_content)
        # gr.Image('elemnt/logo.svg', show_label=False, show_download_button=False,shape=(0.5,0.5))
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Column(scale=12):
                chatbot = gr.Chatbot(show_share_button=True,show_label=True,label='ESG报告生成系统及LLM对话',container=True)
                user_input = gr.Textbox(show_label=False, placeholder="输入", lines=5,scale=2,)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("提交", variant="primary")


        with gr.Column(scale=1):
            emptyBtn = gr.Button("清空历史")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.6, step=0.01, label="Temperature", interactive=True)


            with gr.Column():
                # csv_file = gr.File(label='ESG数据文件',show_label=True)

                #文件箱子：展示data里面的所有文件
                #gr.Radio单选
                data_names_reload()
                file_box = gr.Radio(data_names, label="数据库ESG数据", info="选择ESG数据文件",scale=3)
                file_choose_Btn = gr.Button('选择文件')

                #将输出的元组信息传出至file_inputes中
                # file_choose_Btn.click(
                #     file_input_to_LLM,
                #     inputs= [file_box],
                #     outputs=None
                # )
                # file_inputs = [file_inpute[0] for file_inpute in file_inputs]
                # gr.Radio([file_name_fileter(file_path=file_path) for file_path in  get_all_files(data_base_path)], label="数据库ESG数据", info="选择ESG数据文件",scale=3),
                file_submitBtn = gr.UploadButton('点击上传文件', file_types=['.csv', '.xlsx'], file_count="multiple")
                # file_submitBtn.upload(file_uploaded,file_submitBtn,csv_file)
                file_submitBtn.upload(file_uploaded, file_submitBtn).then(fn=change_file_box,outputs=file_box)

    history = gr.State([])
    past_key_values = gr.State(None)

    #调用大模型模块
    E = gr.Textbox(visible=False)
    S = gr.Textbox(visible=False)
    G = gr.Textbox(visible=False)

    file_choose_Btn.click(
        file_input_to_LLM,
        inputs=[file_box],
        outputs=[E, S, G]
    )

    file_choose_Btn.click(predict,inputs =[E, chatbot, max_length, top_p, temperature, history, past_key_values,gr.Textbox(value=0,visible=False)],
                    outputs =[chatbot, history, past_key_values], show_progress=True).then(predict,inputs =[S, chatbot, max_length, top_p, temperature, history, past_key_values,gr.Textbox(value=0,visible=False)],
                    outputs =[chatbot, history, past_key_values], show_progress=True).then(predict,inputs =[G, chatbot, max_length, top_p, temperature, history, past_key_values,gr.Textbox(value=0,visible=False)],
                    outputs =[chatbot, history, past_key_values], show_progress=True)

    submitBtn.click(predict, inputs = [user_input, chatbot, max_length, top_p, temperature, history, past_key_values,gr.Textbox(value=1,visible=False)],
                    outputs = [chatbot, history, past_key_values], show_progress=True)


    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)


if __name__ == '__main__':
    demo.queue().launch(share=False, server_name="127.0.0.1", server_port=8501, inbrowser=True)