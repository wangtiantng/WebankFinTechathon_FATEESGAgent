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
        请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
        社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
         **子模块分析与撰写**（对应E、S、G模块的子部分）:\n
       - 对{x}模块下的{l2_i}子部分的数据:\n{str1}进行详细分析。\n
       - 为每个子部分撰写报告，包括准确的标题、数据分析、影响评估、结论和建议。\n
       - 每个子报告应有明确的标题，如“{l2_i}部分分析”和“{l2_i}部分建议”。
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
    请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
    社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
    **模块内容整合与优化**（对应ESG报告的E、S、G模块中的{x}模块）:\n
   - 整合{x}模块下所有子部分的内容（如{answer_Xs}）。\n
   - 每个模块应有一个总标题，如“{x}模块综合分析”。\n
   - 对整合内容进行编辑和优化，确保内容连贯、逻辑清晰、专业性强。
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
        请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
        社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
         **子模块分析与撰写**（对应E、S、G模块的子部分）:\n
       - 对{x}模块下的{l2_i}子部分的数据:\n{str1}进行详细分析。\n
       - 为每个子部分撰写报告，包括准确的标题、数据分析、影响评估、结论和建议。\n
       - 每个子报告应有明确的标题，如“{l2_i}部分分析”和“{l2_i}部分建议”。
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
    请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
    社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
    **模块内容整合与优化**（对应ESG报告的E、S、G模块中的{x}模块）:\n
   - 整合{x}模块下所有子部分的内容（如{answer_Xs}）。\n
   - 每个模块应有一个总标题，如“{x}模块综合分析”。\n
   - 对整合内容进行编辑和优化，确保内容连贯、逻辑清晰、专业性强。
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
        请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
        社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
         **子模块分析与撰写**（对应E、S、G模块的子部分）:\n
       - 对{x}模块下的{l2_i}子部分的数据:\n{str1}进行详细分析。\n
       - 为每个子部分撰写报告，包括准确的标题、数据分析、影响评估、结论和建议。\n
       - 每个子报告应有明确的标题，如“{l2_i}部分分析”和“{l2_i}部分建议”。
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
    请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
    社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
    **模块内容整合与优化**（对应ESG报告的E、S、G模块中的{x}模块）:\n
   - 整合{x}模块下所有子部分的内容（如{answer_Xs}）。\n
   - 每个模块应有一个总标题，如“{x}模块综合分析”。\n
   - 对整合内容进行编辑和优化，确保内容连贯、逻辑清晰、专业性强。
"""
answer_Gs_merge = answer(prompt2)
print(f"{x}部分合并：\n{answer_Gs_merge}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("--------------------------G模块总结--------------------------\n%s\n" % answer_Gs_merge)

prompt3 = f"""
    请扮演数据分析师、ESG报告撰写专家和视觉设计师。您的任务是为公司{company}协助撰写和设计一个全面且专业的ESG报告，涵盖环境治理（E）、
    社会责任（S）和公司治理（G）三个主要部分。请根据以下步骤/方面来完成这个任务：\n
    **完整报告的撰写与设计**:\n
   - 将E、S、G三个模块的内容\n（{answer_Es_merge}\n、\n{answer_Ss_merge}\n、\n{answer_Gs_merge}）\n综合为一个完整报告。\n
   - 确保报告有一个引人注目的总标题，如“{company} ESG报告分析”。\n
   - 报告应包含导言、各模块内容、总结和建议，以及附录或参考文献部分。
"""
answer_ESG = answer(prompt3)
print(f"E、S、G三部分合并：\n{answer_ESG}")
with open("运行记录.txt", "a", encoding="utf-8") as file:
    file.write("--------------------------ESG报告总结合并----------------------\n%s\n" % answer_ESG)



