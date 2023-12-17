import os.path

import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# max_new_tokens = 8192
# temperature = 0.2
# top_p = 0.9
# device = "cuda"
# model_path_chat = "/home/kings/PycharmProjects/ChatGLM3/THUDM/chatglm3-6b"
#
# tokenizer_path_chat = model_path_chat
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_chat, trust_remote_code=True, encode_special_tokens=True)
# model = AutoModel.from_pretrained(model_path_chat, load_in_8bit=False, trust_remote_code=True).to(device)
# def answer(prompt):#大模型调用
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     response = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens, history=None,
#                               temperature=temperature,
#                               top_p=top_p, do_sample=True)
#     response = response[0, inputs["input_ids"].shape[-1]:]
#     answer = tokenizer.decode(response, skip_special_tokens=True)
#     return answer

def Read_excel(excel_file ,sheet_name):
    '''
    输入：excel文件名  sheet
    输出：第一列  第二列  模块名
    '''
    df = pd.read_excel(excel_file, sheet_name)  # 变表格

    # 提取第一列、第二列和第三列的内容到列表 l1, l2, l3（去除重复项）
    l1 = list(set(df['Column1'].tolist()))
    l2 = list(set(df['Column2'].tolist()))
    # print(f"l1:\n{l1}")
    # print(f"l2:\n{l2}")

    # 循环遍历 l1
    x = l1[0]
    return df,l1,l2,x

##in:文件名   输出是：prompt列表
def Prompt1(excel_file,sheet_name,company):
    '''
    返回最小一层prompt列表  eg：['能源', '污染物', '废弃物', '水资源', '环保投入', '温室气体']各自的prompt
    '''

    list_of_prompts1 = []
    df,l1,l2,x = Read_excel(excel_file ,sheet_name)
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
            if not pd.isnull(selected_rows['Column5'].iloc[i]):
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

        #print(prompt1)
        list_of_prompts1.append(prompt1)
    return  list_of_prompts1


#print(Prompt1(excel_file = r"/home/kings/ESG/ESG指标.xlsx",sheet_name="Sheet3",company = "宁安如梦"))
# answer = answer(ESG_prompts1()[0])
# print(answer)


#####ESG_prompts1部分，import这个函数就行
def ESG_prompts1(excel_file = r"/home/kings/ESG/ESG指标.xlsx",company = "宁安如梦"):#company = os.path.basename(excel_file)[:-5]
    '''
    输入：excel文件路径，sheet名称，公司名称
    输出：对应prompt元组（E部分的prompt列表，S部分的prompt列表，G部分的prompt列表）
         E部分的prompt列表  eg：['环保投入', '温室气体', '能源', '水资源', '污染物', '废弃物']对应的prompt
    '''
    E_prompt1 = Prompt1(excel_file ,sheet_name="Sheet1",company = os.path.basename(excel_file)[:-5])
    S_prompt1 = Prompt1(excel_file ,sheet_name="Sheet2",company = os.path.basename(excel_file)[:-5])
    G_prompt1 = Prompt1(excel_file ,sheet_name="Sheet3",company = os.path.basename(excel_file)[:-5])
    return E_prompt1,S_prompt1,G_prompt1
# print(len(ESG_prompts1()[0]))
# for i in ESG_prompts1():
#
#     print(len(i))
#     print('-'*50)
# lists = list(ESG_prompts1())
# for i in lists:
#     print(i)
# print(ESG_prompts1()[1])
# print(ESG_prompts1()[2])


def X_prompts2(answer_Xs = "小绵羊",excel_file = r"/home/kings/ESG/ESG指标.xlsx",sheet_name = "Sheet1",company = "宁安如梦"):   #answer_Xs李厅代码
    '''
    你需要调用3次，分别生成E/S/G对应的prompt2
    输入：由大模型输出的E部分 == E1+E2+...+E6
         文件路径
         Sheet1/2/3
         公司名称
    输出：对应的prompt2
    '''
    df,l1,l2,x = Read_excel(excel_file ,sheet_name)
    company = os.path.basename(excel_file)[:-5]
    prompt2 = f"""
            你是一位多年从事ESG报告编写的专业人士，你会用300字对ESG报告{x}模块中所有内容进行综述性的摘要和总结。\n
            以下是\'{company}\'ESG报告中的{x}模块的内容：\n
            {answer_Xs}\n
            结果请突出综合性、专业性和严谨性，逻辑结构清晰，具有说服力。
            要求：按照专业撰写ESG报告的要求进行语言的组织与构思，引人入胜。\
            """

    return prompt2
# print(X_prompts2("小绵阳",excel_file = r"/home/kings/ESG/中国石化.xlsx",sheet_name="Sheet1",company = "宁安如梦"))
# print(X_prompts2("中绵阳",excel_file = r"/home/kings/ESG/中国石化.xlsx",sheet_name="Sheet2",company = "宁安如梦"))
# print(X_prompts2("大绵阳",excel_file = r"/home/kings/ESG/中国石化.xlsx",sheet_name="Sheet3",company = "宁安如梦"))


# #####ESG_prompts2部分，import这个函数就行
# def ESG_prompts2(answer_Xs = ["e绵阳","s绵阳","g绵阳"],excel_file = r"/home/kings/ESG/ESG指标.xlsx",company = "宁安如梦"):
#     '''
#     输入：大模型的输出汇总['E1+E2+...+E6','S1+...+S8','G1+...+G11']
#          文件名
#          公司名称
#     输出：一个包含3个元素的元组，每个元素为一个prompt字符串（E部分总结 对应的prompt2,S部分总结 对应的prompt2，G部分总结 对应的prompt2）
#     '''
#     E_prompt2 = prompts2(answer_Xs[0],excel_file ,sheet_name="Sheet1",company = "宁安如梦")
#     S_prompt2 = prompts2(answer_Xs[1],excel_file ,sheet_name="Sheet2",company = "宁安如梦")
#     G_prompt2 = prompts2(answer_Xs[2],excel_file ,sheet_name="Sheet3",company = "宁安如梦")
#     return E_prompt2,S_prompt2,G_prompt2
#
# print(ESG_prompts2()[0])
# print(ESG_prompts2()[1])
# print(ESG_prompts2()[2])

def ESG_prompt3(answer_Es_merge = "dawei",answer_Ss_merge = "qingmei",answer_Gs_merge = "xuxiaoman",excel_file = r"/home/kings/ESG/ESG指标.xlsx",company = "宁安如梦"):
    '''
    调用1次
    输入：E1+...+E6,S1+...+S8,G1+...+G11,公司名称
    输出：一个元素 = ESG的汇总模板
    '''
    company = os.path.basename(excel_file)[:-5]
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
    return prompt3

# print(ESG_prompt3())