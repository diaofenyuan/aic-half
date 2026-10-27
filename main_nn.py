"""神经网络入口已停用；新版仅使用 LightGBM。"""


def __getattr__(name):
    raise AttributeError(f"神经网络已停用，不能调用 {name}；请使用独立短期/长期树模型")
