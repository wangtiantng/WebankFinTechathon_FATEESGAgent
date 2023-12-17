import gradio as gr


E1 = gr.Textbox(visible=False)




def get_variable_name(variable):
    # 使用反射机制获取变量名
    variable_name = [name for name, value in globals().items() if value is variable][0]
    return variable_name

# 测试代码
test = 42

variable_name = get_variable_name(E1)

print(type(variable_name))  # 输出：my_variable
