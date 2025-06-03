CUSTOM_PACKAGE_DIR=[

]
WORKERS_NUM = 0 # 进程数，可配置范围[0, cpu核数]。 默认为0， 根据用户配置的请求最大并发数自动分配

# 压测相关
PRESSURE_TIME = 0.2 * 60 # 压测时长，单位sec
CONNECTION_ADD_RATE = 1 # 每个进程新增连接个数的频率 单位 个/s
