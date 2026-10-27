"""96步长期模型独立训练、保存及推理，复用无状态树训练原语。"""
from main_s import train_load, predict_load


def train_long(prepared, recipe=None, quick=False, progress=None):
    return train_load(prepared, "long", recipe, quick, progress)


def predict_long(bundle, prepared, origins):
    if bundle.span != "long":
        raise ValueError("长期入口拒绝短期模型")
    return predict_load(bundle, prepared, origins)
