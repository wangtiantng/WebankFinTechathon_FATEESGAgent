# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

max_new_tokens = 8192
temperature = 0.2
top_p = 0.9
device = "cuda"
model_path_chat = "/home/kings/PycharmProjects/ChatGLM3/THUDM/chatglm3-6b"

tokenizer_path_chat = model_path_chat
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_chat, trust_remote_code=True, encode_special_tokens=True)
model = AutoModel.from_pretrained(model_path_chat, load_in_8bit=False, trust_remote_code=True).to(device)


def answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    response = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens, history=None,
                              temperature=temperature,
                              top_p=top_p, do_sample=True)
    response = response[0, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    return answer


company = "宁安如梦"
# 读取 Excel 文件
excel_file = r"/home/kings/ESG/ESG指标.xlsx"
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("\n\n\n------------以下是%sESG报告的运行结果-------------------\n" % company)

# E模块处理# ——————————————————————————————————————————————————————————————————————————————
df = pd.read_excel(excel_file, sheet_name="Sheet1")

# 提取第一列、第二列和第三列的内容到列表 l1, l2, l3（去除重复项）
l1 = list(set(df['Column1'].tolist()))
l2 = list(set(df['Column2'].tolist()))
print(f"l1:\n{l1}")
print(f"l2:\n{l2}")

# 循环遍历 l1
x = l1[0]
# 遍历 l2
str_l2 = []
answer_Es = ""
for l2_i in l2:
    str1 = ""
    # 根据条件筛选 DataFrame
    selected_rows = df[(df['Column1'] == x) & (df['Column2'] == l2_i)]
    len_l2 = len(selected_rows)
    # print(selected_rows)
    # 如果有符合条件的行
    for i in range(len_l2):
        if not np.isnan(selected_rows['Column5'].iloc[i]):
            # 提取第三列和第四列的值，合并成字符串，中间加一个空格
            merged_string = f"{selected_rows['Column3'].iloc[i]} {selected_rows['Column4'].iloc[i]}"

            # 输出合并后的字符串加上第五列的值
            str1 = str1 + f"{merged_string} {selected_rows['Column5'].iloc[i]}\n"
    # print(l2_i)
    # print(f"{str1}")
    prompt1 = f"""
                你是一位多年从事ESG报告编写的专业人士，你首先会解释公司ESG相关数据指标含义，然后进行数据披露和数据分析，最后提出ESG实施建议并用100字阐述公司相关方面的具体成就。\n
                以下是公司\'{company}\'的ESG报告中的{x}模块的{l2_i}部分的相关数据，\n
                {str1}\n
                要求：根据ESG报告撰写标准和以上数据为该企业生成一个全面的{x}模块中{l2_i}部分的报告，输出结果为txt格式。
                要求严格的数据支持，句子具有ESG报告风格，分层观点和结构，引人入胜，突出重点，分点列出时不要加编号，加上适当的标题。
                """
    # print(prompt1)
    answer_E = answer(prompt1)
    # print(answer_E)
    answer_Es = answer_Es + f"{answer_E}"
print(f"{x}部分分散：\n{answer_Es}")
# 打开文件并将列表内容附加到文件末尾
with open("运行记录.txt", "a", encoding="utf-8") as file:
        file.write("-------------------E模块个部分--------------------\n%s\n" % answer_Es)
answer_Xs = answer_Es


prompt2 = f"""
        你是一位多年从事ESG报告编写的专业人士，你会用300字对ESG报告{x}模块中所有内容进行综述性的摘要和总结。\n
        以下是{company}ESG报告中的{x}模块的内容：\n
        {answer_Xs}\n
        结果请突出综合性、专业性和严谨性，逻辑结构清晰，具有说服力。
        要求：按照专业撰写ESG报告的要求进行语言的组织与构思，引人入胜。\
        """
answer_Es_merge = answer(prompt2)
print(f"{x}部分合并：\n{answer_Es_merge}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("-----------------------E模块总结-------------------\n%s\n" % answer_Es_merge)

# S模块处理# ——————————————————————————————————————————————————————————————————————————————
df = pd.read_excel(excel_file, sheet_name="Sheet2")

# 提取第一列、第二列和第三列的内容到列表 l1, l2, l3（去除重复项）
l1 = list(set(df['Column1'].tolist()))
l2 = list(set(df['Column2'].tolist()))
print(f"l1:\n{l1}")
print(f"l2:\n{l2}")

# 循环遍历 l1
x = l1[0]
# 遍历 l2
str_l2 = []
answer_Ss = ""
for l2_i in l2:
    str1 = ""
    # 根据条件筛选 DataFrame
    selected_rows = df[(df['Column1'] == x) & (df['Column2'] == l2_i)]
    len_l2 = len(selected_rows)
    # print(selected_rows)
    # 如果有符合条件的行
    for i in range(len_l2):
        if not np.isnan(selected_rows['Column5'].iloc[i]):
            # 提取第三列和第四列的值，合并成字符串，中间加一个空格
            merged_string = f"{selected_rows['Column3'].iloc[i]} {selected_rows['Column4'].iloc[i]}"

            # 输出合并后的字符串加上第五列的值
            str1 = str1 + f"{merged_string} {selected_rows['Column5'].iloc[i]}\n"
    # print(l2_i)
    # print(f"{str1}")
    prompt1 = f"""
                你是一位多年从事ESG报告编写的专业人士，你首先会解释公司ESG相关数据指标含义，然后进行数据披露和数据分析，最后提出ESG实施建议并用100字阐述公司相关方面的具体成就。\n
                以下是公司\'{company}\'的ESG报告中的{x}模块的{l2_i}部分的相关数据，\n
                {str1}\n
                要求：根据ESG报告撰写标准和以上数据为该企业生成一个全面的{x}模块中{l2_i}部分的报告，输出结果为txt格式。
                要求严格的数据支持，句子具有ESG报告风格，分层观点和结构，引人入胜，突出重点，分点列出时不要加编号，加上适当的标题。
                """
    # print(prompt1)
    answer_S = answer(prompt1)
    # print(answer_E)
    answer_Ss = answer_Ss + f"{answer_S}"
print(f"{x}部分分散：\n{answer_Ss}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("----------------------S模块各部分--------------------------\n%s\n" % answer_Ss)

answer_Xs = answer_Ss
prompt2 = f"""
        你是一位多年从事ESG报告编写的专业人士，你会用300字对ESG报告{x}模块中所有内容进行综述性的摘要和总结。\n
        以下是{company}ESG报告中的{x}模块的内容：\n
        {answer_Xs}\n
        结果请突出综合性、专业性和严谨性，逻辑结构清晰，具有说服力。
        要求：按照专业撰写ESG报告的要求进行语言的组织与构思，引人入胜。\
        """
answer_Ss_merge = answer(prompt2)
print(f"{x}部分合并：\n{answer_Ss_merge}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("----------------------------S模块总结-----------------------\n%s\n" % answer_Ss_merge)

# G模块处理# ——————————————————————————————————————————————————————————————————————————————
df = pd.read_excel(excel_file, sheet_name="Sheet3")

# 提取第一列、第二列和第三列的内容到列表 l1, l2, l3（去除重复项）
l1 = list(set(df['Column1'].tolist()))
l2 = list(set(df['Column2'].tolist()))
print(f"l1:\n{l1}")
print(f"l2:\n{l2}")

# 循环遍历 l1
x = l1[0]
# 遍历 l2
str_l2 = []
answer_Gs = ""
for l2_i in l2:
    str1 = ""
    # 根据条件筛选 DataFrame
    selected_rows = df[(df['Column1'] == x) & (df['Column2'] == l2_i)]
    len_l2 = len(selected_rows)
    # print(selected_rows)
    # 如果有符合条件的行
    for i in range(len_l2):
        if not np.isnan(selected_rows['Column5'].iloc[i]):
            # 提取第三列和第四列的值，合并成字符串，中间加一个空格
            merged_string = f"{selected_rows['Column3'].iloc[i]} {selected_rows['Column4'].iloc[i]}"

            # 输出合并后的字符串加上第五列的值
            str1 = str1 + f"{merged_string} {selected_rows['Column5'].iloc[i]}\n"
    # print(l2_i)
    # print(f"{str1}")
    prompt1 = f"""
                你是一位多年从事ESG报告编写的专业人士，你首先会解释公司ESG相关数据指标含义，然后进行数据披露和数据分析，最后提出ESG实施建议并用100字阐述公司相关方面的具体成就。\n
                以下是公司\'{company}\'的ESG报告中的{x}模块的{l2_i}部分的相关数据，\n
                {str1}\n
                要求：根据ESG报告撰写标准和以上数据为该企业生成一个全面的{x}模块中{l2_i}部分的报告，输出结果为txt格式。
                要求严格的数据支持，句子具有ESG报告风格，分层观点和结构，引人入胜，突出重点，分点列出时不要加编号，加上适当的标题。
                """

    # print(prompt1)
    answer_G = answer(prompt1)
    # print(answer_E)
    answer_Gs = answer_Gs + f"{answer_G}"
print(f"{x}部分分散：\n{answer_Gs}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("----------------------------G模块各部分------------------------\n%s\n" % answer_Gs)

answer_Xs = answer_Gs
prompt2 = f"""
        你是一位多年从事ESG报告编写的专业人士，你会用300字对ESG报告{x}模块中所有内容进行综述性的摘要和总结。\n
        以下是{company}ESG报告中的{x}模块的内容：\n
        {answer_Xs}\n
        结果请突出综合性、专业性和严谨性，逻辑结构清晰，具有说服力。
        要求：按照专业撰写ESG报告的要求进行语言的组织与构思，引人入胜。\
        """
answer_Gs_merge = answer(prompt2)
print(f"{x}部分合并：\n{answer_Gs_merge}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("--------------------------G模块总结--------------------------\n%s\n" % answer_Gs_merge)

prompt3 = f"""
            你是一位多年从事ESG报告编写的专业人士，并擅长对报告内容进行综述性汇报，你会用400字对ESG报告中的所有内容进行综述性的摘要和总结。\n
            以下是{company}ESG报告中E、S、G三个模块的内容：\n
            （{answer_Es_merge}\n
            {answer_Ss_merge}\n
            {answer_Gs_merge}）\n
            请整合并丰富这三个模块的内容，润色并完成撰写该公司完整的ESG报告。\n
            要求：按照专业的撰写ESG报告的要求进行组织与构思，且要求严格的数据支持，分层观点和结构，引人入胜，
            结果请突出专业性和严谨性，逻辑结构清晰，形式美观，具有说服力，内容字数详实。
            """

answer_ESG = answer(prompt3)
print(f"E、S、G三部分合并：\n{answer_ESG}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("--------------------------ESG报告总结合并----------------------\n%s\n" % answer_ESG)



