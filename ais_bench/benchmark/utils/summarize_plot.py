import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ais_bench.benchmark.utils import get_logger
from typing import List
import numpy as np
import time


def plot_sorted_request_timelines(start_time_list: List[float], prefill_latency_list: List[float],
                                  end_time_list: List[float], decode_token_latencies_list: List[List[float]],
                                  output_file: str = "timeline.html", unit : str = "s"):
    """
    绘制请求时间线和并发图表
    
    参数:
    start_time_list -- 请求开始时间列表
    prefill_latency_list -- 首token时延列表
    end_time_list -- 请求结束时间列表
    decode_token_latencies_list -- 非首token时延列表
    output_file -- 输出HTML文件名
    unit -- 时间单位 (仅用于渲染时的内容显示，不涉及计算逻辑)
    """
    logger = get_logger()

    # ========= 渲染因子配置 ==========
    MAX_POINTS_PER_TRACE = 10000  # 根据硬件性能调整
    # ======= WebGL 配置 =======
    WEBGL_CONFIG = {
        'scrollZoom': True,
        'plotGlPixelRatio': 1,  # 渲染分辨率
        'showLink': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['toImage'],
        'queueLength': 10,      # 渲染队列长度
    }

    prev_time = None
    start_time = time.perf_counter()
    prev_time = start_time
    logger.info("Rendering request timeline diagram...")
    
    # ================== 参数校验 ==================
    n_requests = len(start_time_list)
    if (n_requests == 0 or 
        n_requests != len(prefill_latency_list) or 
        n_requests != len(end_time_list) or
        n_requests != len(decode_token_latencies_list)):
        logger.warning("No valid data to plot!")
        logger.warning("Input list lengths mismatch! Details: ")
        logger.warning(f"start_list:{n_requests}, prefill_latency_list:{len(prefill_latency_list)}, ")
        logger.warning(f"end_list:{len(end_time_list)}, decode_token_latencies_list:{len(decode_token_latencies_list)}")
        return

    # ================== 数据准备 ==================
    logger.info("Processing the received data...")
    
    start = np.asarray(start_time_list, dtype=np.float64)
    prefill = np.asarray(prefill_latency_list, dtype=np.float64) / 1000 #prefill数据单位为ms，而其他数据均为s
    end = np.asarray(end_time_list, dtype=np.float64)

    n_valid = n_requests # 有效数据数目，可根据场景调整
    
    # 计算每个请求的关键时间点
    first_token_time_list = start + prefill

    # 对每条请求是否含有非首token时延判断请求索引对应的end_time是否需要更新，
    # 因为end_time_list因为打点位置会有误差，需用first_token_time_list的值修正
    no_decode_indices = [i for i, lst in enumerate(decode_token_latencies_list) if not lst]
    if no_decode_indices:
        end[no_decode_indices] = first_token_time_list[no_decode_indices]
        del no_decode_indices

    # 找到全局最小时间，用于相对时间计算
    global_x_min = np.min(start)
    
    # 计算所有请求的相对时间
    adjusted_starts = (start - global_x_min)
    adjusted_first_tokens = (first_token_time_list - global_x_min)
    adjusted_ends = (end - global_x_min)

    curr_time = time.perf_counter()
    logger.info(f"Data preprocessing completed! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Processing data for request timeline diagram...")

    # ================== 请求图：准备渲染数据 ==================
    # ================== 预分配内存 ==================
    # 红线段数据（TTFT）：每个请求3个点（起点、终点、断开点None）
    red_x = np.full(3 * n_valid, np.nan, dtype=np.float32) # 标准化时间后不需要float64位存储
    red_y = np.full(3 * n_valid, np.nan, dtype=np.float32)
    
    # 蓝线段数据（Decode）
    blue_x = np.full(3 * n_valid, np.nan, dtype=np.float32)
    blue_y = np.full(3 * n_valid, np.nan, dtype=np.float32)
    
    # 悬停文本，仅在起点存储
    hover_text = np.full(3 * n_valid, None, dtype=object)
    
    # 使用开始时间排序
    sorted_indices = np.argsort(adjusted_starts)
    
    # ================== 填充线段数据 ==================
    for sorted_pos, orig_idx in enumerate(sorted_indices):
        # 获取当前请求的关键时间点
        start_t = adjusted_starts[orig_idx]
        first_token_t = adjusted_first_tokens[orig_idx]
        end_t = adjusted_ends[orig_idx]
        
        # 计算数组中的位置
        arr_idx = sorted_pos * 3
        
        # 红线段（TTFT）：从开始到第一个token
        red_x[arr_idx] = start_t
        red_x[arr_idx + 1] = first_token_t
        red_y[arr_idx:arr_idx + 2] = sorted_pos + 1
        
        blue_content_data = "NaN"

        # 蓝线段（解码时间）：从第一个token到结束
        if end_t > first_token_t:
            blue_x[arr_idx] = first_token_t
            blue_x[arr_idx + 1] = end_t
            blue_y[arr_idx:arr_idx + 2] = sorted_pos + 1
            decode_time = end_t - first_token_t
            blue_content_data = f"{first_token_t:.2f}→{end_t:.2f}={decode_time:.2f}"
        
        # 悬停文本，触发点在红线段起点
        ttft = first_token_t - start_t
        e2e = end_t - start_t
        
        red_content = f"<span style='color:red'>TTFT({unit}): {start_t:.2f}→{first_token_t:.2f}={ttft:.2f}</span><br>"
        blue_content = f"<span style='color:blue'>Decode({unit}): {blue_content_data}</span><br>"
        e2e_content = f"E2E({unit}): {start_t:.2f}→{end_t:.2f}={e2e:.2f}"
        
        hover_text[arr_idx] = red_content + blue_content + e2e_content

    curr_time = time.perf_counter()
    logger.info(f"Timeline data processed! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Request timeline diagram data processing complete! Rendering...")
    
    # ================== 请求图：分块渲染 ==================
    points_per_request = 3
    chunk_size = max(5000, min(n_valid, MAX_POINTS_PER_TRACE // points_per_request))
    n_chunks = (n_valid + chunk_size - 1) // chunk_size
    n_points = len(red_x)
    timeline_traces = []

    # 创建分块轨迹
    for i in range(n_chunks):
        start_idx = i * chunk_size * points_per_request
        end_idx = min((i+1) * chunk_size * points_per_request, n_points)
        chunk = slice(start_idx, end_idx)
        
        # 红色TTFT轨迹
        if any(~np.isnan(red_x[chunk])):
            timeline_traces.append(go.Scattergl(
                x=red_x[chunk],
                y=red_y[chunk],
                mode='lines',
                line=dict(color='red', width=1, shape="hv"),
                hoverinfo='text',
                hovertext=hover_text[chunk],
                showlegend=False,
                connectgaps=False
            ))
        
        # 蓝色Decode轨迹
        if any(~np.isnan(blue_x[chunk])):
            timeline_traces.append(go.Scattergl(
                x=blue_x[chunk],
                y=blue_y[chunk],
                mode='lines',
                line=dict(color='blue', width=1, shape="hv"),
                hoverinfo='none',
                showlegend=False,
                connectgaps=False
            ))
    
    # 清理大数组释放内存
    del red_x, red_y, blue_x, blue_y, hover_text

    curr_time = time.perf_counter()
    logger.info(f"Timeline rendered! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Processing data for request concurrency line chart...")

    # ================== 并发图：请求并发数量线段图处理 ==================
    # 生成事件数组（开始事件+1，结束事件-1）
    events = np.empty((2 * n_valid, 2), dtype=np.float32)
    events[:n_valid, 0] = adjusted_starts
    events[:n_valid, 1] = 1  # 开始事件
    events[n_valid:, 0] = adjusted_ends
    events[n_valid:, 1] = -1  # 结束事件

    # 稳定排序（时间相同则开始事件优先）
    sort_indices = np.lexsort((events[:, 1], events[:, 0]))
    events = events[sort_indices]

    # 分组计算每个时间点的净变化
    unique_times, indices = np.unique(
        events[:, 0], 
        return_index=False,
        return_counts=False,
        return_inverse=True
    )
    delta_per_time = np.bincount(indices, weights=events[:, 1])
    
    # 计算结束后的并发状态
    cumulative = np.cumsum(delta_per_time)
    
    # 精简数据结构：只存储结束状态 + 最终水平延伸点
    conc_times = unique_times.copy()
    conc_counts = cumulative.copy()
    
    # 添加最终水平延伸点（保持并发数不变）
    if len(conc_times) > 0:
        last_time = conc_times[-1] + 0.001 * (conc_times[-1] - conc_times[0]) 
        last_time = max(last_time, conc_times[-1] + 0.1)  # 确保有最小延伸
        conc_times = np.append(conc_times, last_time)
        conc_counts = np.append(conc_counts, conc_counts[-1])
    
    # 创建简洁悬浮文本
    conc_hover_text = [
        f"Time: {t:.2f}{unit}<br>Concurrency: {c:.0f}" 
        for t, c in zip(conc_times[:-1], conc_counts[:-1])
    ]
    conc_hover_text.append("")  # 最后延伸点无悬浮提示

    curr_time = time.perf_counter()
    logger.info(f"Concurrency data processed! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Request concurrency line chart data processing complete! Rendering...")

    # ================== 并发图：分块渲染 ==================
    n_points = len(conc_times)
    chunk_size = min(len(conc_times), MAX_POINTS_PER_TRACE)
    n_chunks = (len(conc_times) + chunk_size - 1) // chunk_size
    concurrency_traces = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        
        # 确保连续：每块起始点包含上一块结束点（第一块除外）
        if i > 0:
            start_idx = max(0, start_idx - 1)
        
        chunk_x = conc_times[start_idx:end_idx]
        chunk_y = conc_counts[start_idx:end_idx]
        chunk_hover = conc_hover_text[start_idx:end_idx]
        
        if len(chunk_x) == 0:
            continue
            
        # 创建并发轨迹块
        concurrency_traces.append(go.Scattergl(
            x=chunk_x,
            y=chunk_y,
            mode='lines',
            line=dict(color='#4CAF50', width=1, shape='hv'),
            fill='tozeroy',
            fillcolor='rgba(76,175,80,0.1)',
            hoverinfo="text",
            hovertext=chunk_hover,
            showlegend=False,
            connectgaps=True
        ))

    # 清理大数组释放内存
    del events, sort_indices, unique_times, indices, delta_per_time, cumulative
    del conc_times, conc_counts, conc_hover_text

    curr_time = time.perf_counter()
    logger.info(f"Request concurrency line chart rendered! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Merging both diagrams...")

    # ================== 合并图表 ==================
    combined_fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
    
    # 添加请求图轨迹
    for trace in timeline_traces:
        combined_fig.add_trace(trace, row=1, col=1)
    
    # 添加并发图轨迹
    for trace in concurrency_traces:
        combined_fig.add_trace(trace, row=2, col=1)

    # ================== 坐标轴配置 ==================
    axis_config = dict(
        showline=True,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(211,211,211,0.5)',
        linecolor='black',
    )
    
    # 计算X轴范围
    x_range = [0, np.max(adjusted_ends)]
    
    xaxis_config = dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#666',
        spikedash='dot',
        title=f"Relative Time ({unit})",
        range=x_range,
    )
    
    yaxis_config = dict(
        rangemode='nonnegative',
        tickmode='auto',
        nticks=10,
    )
    
    combined_fig.update_layout(
        height=1200,
        plot_bgcolor='white',
        xaxis1=dict(
            **axis_config,
            **xaxis_config,
            matches='x2',
        ),
        yaxis1=dict(
            **axis_config,
            **yaxis_config,
            title="Request Index", 
        ),
        xaxis2=dict(
            **axis_config,
            **xaxis_config,
        ),
        yaxis2=dict(
            **axis_config,
            **yaxis_config,
            title="Request Concurrency Count",
        ),
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            font_size=12,
            align='left'
        ),
        hovermode='closest',
    )

    curr_time = time.perf_counter()
    logger.info(f"The merged diagram has been set up! Time: {curr_time - prev_time:.4f} seconds")
    prev_time = curr_time
    logger.info("Rendering the merged diagram...")

    # ================== 输出HTML ==================
    combined_fig.write_html(
        output_file,
        include_plotlyjs='cdn',
        config=WEBGL_CONFIG,
        auto_open=False,
        full_html=False,
    )

    curr_time = time.perf_counter()
    logger.info(f"Merged diagram rendered! Time: {curr_time - prev_time:.4f} seconds")
    total_time = curr_time - start_time
    logger.info(f"Succeed! HTML file saved! Total execution time: {total_time:.4f} seconds")