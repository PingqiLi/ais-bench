"""
# StringConfig中的随机生成方法参数说明:
# -------------------------------------------------
# [Uniform均匀分布] -- "Method" : "uniform"
#   - MinValue: 最小值 (包含，整数/浮点数)
#   - MaxValue: 最大值 (不包含，需>MinValue)
#
# [Gaussian高斯分布] -- "Method" : "gaussian"
#   - Mean    : 平均值 (分布中心位置)
#   - Var     : 方差 (控制数据分散程度)
#   - MinValue: 截断下限 (可低于Mean)
#   - MaxValue: 截断上限 (可高于Mean)
#
# [Zipf齐夫分布] -- "Method" : "zipf"
#   - Alpha   : 形状参数 (>1，值越大分布越均匀)
#   - MinValue: 最小排名 (整数，通常从1开始)
#   - MaxValue: 最大排名 (整数，需>MinValue)
"""
synthetic_config = {
    "Type":"tokenid",   # [tokenid/string]，生成的随机数据集类型，支持固定长度的随机tokenid，和随机长度的string，两种类型的数据集
    "RequestCount": 10, # 生成的请求条数，应与模型侧配置文件中的 input_seq_len 一致
    "StringConfig" : {  # string类型的随机数据集的配置相关项，请参考以上注释处："StringConfig中的随机生成方法参数说明"
        "Input" : {     # 输入的token个数，即每条请求生成的token id的个数
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 200}
        },
        "Output" : {    # 期望生成的token个数，即期望的输出token id的长度
            "Method": "gaussian",
            "Params": {"Mean": 100, "Var": 200, "MinValue": 1, "MaxValue": 100}
        }
    },
    "TokenIdConfig" : { # tokenid类型的随机数据集的配置相关项
        "ModelPath": "ModelPath", # 模型权重路径，必须与模型侧配置文件中的weight_dir/model/path等表示模型或词汇表的路径相同
        "RequestSize": 10 # 每条请求的长度，即每条请求中token id的个数
    }
}