import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ais_bench.benchmark.utils import get_logger
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time

# ================== 常量配置 ==================
WEBGL_CONFIG = {
    'scrollZoom': True,
    'plotGlPixelRatio': 1,
    'showLink': False,
    'displaylogo': False,
    'queueLength': 10,
}

AXIS_CONFIG = dict(
    showline=True,
    showgrid=True,
    showticklabels=True,
    gridwidth=0.5,
    gridcolor='rgba(211,211,211,0.5)',
    linecolor='black',
)

# ================== 分块渲染配置 ==================
MAX_POINTS_PER_TRACE = 10000  # 每个轨迹渲染的最大点数
TIMELINE_POINTS_PER_REQUEST = 3  # 每个请求在时间线图中占3个点（起始、末尾、断开）

# ================== 辅助函数 ==================
def validate_input_data(
    start_time_list: List[float],
    prefill_latency_list: List[float],
    end_time_list: List[float],
    decode_token_latencies_list: List[List[float]],
) -> bool:
    """验证输入数据是否合法"""
    logger = get_logger()
    n_requests = len(start_time_list)
    if n_requests == 0:
        logger.warning("No requests to plot!")
        return False

    if (n_requests != len(prefill_latency_list) or
        n_requests != len(end_time_list) or
        n_requests != len(decode_token_latencies_list)):
        logger.warning("Input list lengths mismatch! Details: ")
        logger.warning(f"start_list:{n_requests}, prefill_latency_list:{len(prefill_latency_list)}")
        logger.warning(f"end_list:{len(end_time_list)}, decode_token_latencies_list:{len(decode_token_latencies_list)}")
        return False

    return True

def is_non_streaming_scenario(
    prefill_latency_list: List[float],
    decode_token_latencies_list: List[List[float]]
) -> bool:
    """判断是否是非流式场景"""
    return all(p == 0.0 for p in prefill_latency_list)

def preprocess_data(
    start_time_list: List[float],
    prefill_latency_list: List[float],
    end_time_list: List[float],
    decode_token_latencies_list: List[List[float]],
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, bool]:
    """
    数据预处理
    返回: (first_token_times, adjusted_starts, adjusted_ends, is_non_streaming)
    """
    start = np.asarray(start_time_list, dtype=np.float64)
    prefill = np.asarray(prefill_latency_list, dtype=np.float64) / 1000  # prefill数据单位为ms，而其他数据均为s
    end = np.asarray(end_time_list, dtype=np.float64)

    # 检测是否是非流式场景
    is_non_streaming = is_non_streaming_scenario(prefill_latency_list, decode_token_latencies_list)

    # 计算首token时间
    first_token_times = (start + prefill) if not is_non_streaming else None

    # 对每条请求是否含有非首token时延判断请求索引对应的end_time是否需要更新，
    # 因为end_time_list因为打点位置会有误差，需用first_token_time_list的值修正
    # 仅在非流式场景修正结束时间
    if not is_non_streaming:
        no_decode_indices = [i for i, lst in enumerate(decode_token_latencies_list) if not lst.any()]
        if no_decode_indices:
            end[no_decode_indices] = first_token_times[no_decode_indices]
            get_logger().debug(f"Adjusted {len(no_decode_indices)} requests with no decode tokens")
            del no_decode_indices

    # 计算全局最小时间
    global_x_min = np.min(start) if len(start) > 0 else 0.0

    # 计算相对时间
    adjusted_starts = start - global_x_min
    adjusted_first_tokens = (first_token_times - global_x_min) if not is_non_streaming else None
    adjusted_ends = end - global_x_min

    return adjusted_first_tokens, adjusted_starts, adjusted_ends, is_non_streaming

def generate_timeline_traces(
    adjusted_starts: np.ndarray,
    adjusted_ends: np.ndarray,
    adjusted_first_tokens: np.ndarray,
    multiturn_group_id_list: list,
    unit: str
) -> List[go.Scattergl]:
    """生成请求时间线图的轨迹"""
    n_requests = len(adjusted_starts)
    if n_requests == 0:
        return []
    unique_ids = []  #without sorted group id
    first_index_lookup = {}  #key: without sorted group id; value: index
    index_map = [] # idex
    for idx, val in enumerate(multiturn_group_id_list):
        if val not in first_index_lookup:
            first_index_lookup[val] = len(unique_ids)
            unique_ids.append(val)
        index_map.append(first_index_lookup[val])

    # unique_ids, index_map = np.unique(multiturn_group_id_list, return_inverse=True)
    is_multiturn = True if unique_ids[0] else False
    if is_multiturn:
        get_logger().info("Visualization in multi-turn conversations")
    y_values = np.array(index_map) + 1
    # 预分配内存
    red_x = np.full(TIMELINE_POINTS_PER_REQUEST * n_requests, np.nan, dtype=np.float32)
    red_y = np.full_like(red_x, np.nan)
    blue_x = np.full_like(red_x, np.nan)
    blue_y = np.full_like(red_x, np.nan)
    hover_text = np.full(TIMELINE_POINTS_PER_REQUEST * n_requests, None, dtype=object)
    sorted_indices = np.argsort(adjusted_starts)

    for sorted_pos, orig_idx in enumerate(sorted_indices):
        # 获取当前请求的关键时间点
        start_t = adjusted_starts[orig_idx]
        first_token_t = adjusted_first_tokens[orig_idx]
        end_t = adjusted_ends[orig_idx]
        y = y_values[orig_idx]

        # 计算数组中的位置
        arr_idx = sorted_pos * 3

        # 红线段（TTFT）：从开始到第一个token
        red_x[arr_idx] = start_t
        red_x[arr_idx + 1] = first_token_t
        red_y[arr_idx:arr_idx + 2] = y if is_multiturn else sorted_pos + 1

        blue_content_data = "NaN"

        # 蓝线段（Decode）：从第一个token到结束
        if end_t > first_token_t:
            blue_x[arr_idx] = first_token_t
            blue_x[arr_idx + 1] = end_t
            blue_y[arr_idx:arr_idx + 2] = y if is_multiturn else sorted_pos + 1
            decode_time = end_t - first_token_t
            blue_content_data = f"{first_token_t:.2f}→{end_t:.2f}={decode_time:.2f}"

        # 悬停文本，触发点在红线段起点
        ttft = first_token_t - start_t
        e2e = end_t - start_t

        red_content = f"<span style='color:red'>TTFT({unit}): {start_t:.2f}→{first_token_t:.2f}={ttft:.2f}</span><br>"
        blue_content = f"<span style='color:blue'>Decode({unit}): {blue_content_data}</span><br>"
        e2e_content = f"E2E({unit}): {start_t:.2f}→{end_t:.2f}={e2e:.2f}"
        hover_text[arr_idx] = red_content + blue_content + e2e_content

    # 分块生成轨迹
    traces = []
    n_points = len(red_x)
    chunk_size = min(n_points, MAX_POINTS_PER_TRACE)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        chunk = slice(start_idx, end_idx)

        # 红线段
        if np.any(~np.isnan(red_x[chunk])):
            traces.append(go.Scattergl(
                x=red_x[chunk],
                y=red_y[chunk],
                mode='lines',
                line=dict(color='red', width=1, shape="hv"),
                hoverinfo='text',
                hovertext=hover_text[chunk],
                showlegend=False,
                connectgaps=False
            ))

        # 蓝线段
        if np.any(~np.isnan(blue_x[chunk])):
            traces.append(go.Scattergl(
                x=blue_x[chunk],
                y=blue_y[chunk],
                mode='lines',
                line=dict(color='blue', width=1, shape="hv"),
                hoverinfo='none',
                showlegend=False,
                connectgaps=False
            ))

    del red_x, red_y, blue_x, blue_y, hover_text
    return traces

def generate_concurrency_traces(
    adjusted_starts: np.ndarray,
    adjusted_ends: np.ndarray,
    unit: str
) -> List[go.Scattergl]:
    """生成并发图的轨迹"""
    # 过滤零长度请求
    valid_mask = adjusted_starts < adjusted_ends
    if not np.any(valid_mask):
        get_logger().warning("No valid requests for concurrency plot!")
        return []

    valid_starts = adjusted_starts[valid_mask]
    valid_ends = adjusted_ends[valid_mask]
    n_events = len(valid_starts) * 2

    # 生成事件数组
    events = np.empty((n_events, 2), dtype=np.float32)
    events[:len(valid_starts), 0] = valid_starts
    events[:len(valid_starts), 1] = 1  # 开始事件
    events[len(valid_starts):, 0] = valid_ends
    events[len(valid_starts):, 1] = -1  # 结束事件

    # 稳定排序（时间相同则开始事件优先）
    sort_indices = np.lexsort((events[:, 1], events[:, 0]))
    events = events[sort_indices]

    # 计算并发数
    unique_times, inverse_indices = np.unique(events[:, 0], return_inverse=True)
    delta_per_time = np.bincount(inverse_indices, weights=events[:, 1])
    cumulative = np.cumsum(delta_per_time)

    conc_times = unique_times
    conc_counts = cumulative

    # 创建悬停文本
    conc_hover_text = [
        f"Time: {t:.4f}{unit}<br>Concurrency: {c:.0f}"
        for t, c in zip(conc_times, conc_counts)
    ]

    # 分块渲染
    traces = []
    n_points = len(conc_times)
    chunk_size = min(n_points, MAX_POINTS_PER_TRACE)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)

        if i > 0:
            start_idx = max(0, start_idx - 1)  # 确保连续

        chunk = slice(start_idx, end_idx)

        traces.append(go.Scattergl(
            x=conc_times[chunk],
            y=conc_counts[chunk],
            mode='lines',
            line=dict(color='#4CAF50', width=1, shape='hv'),
            fill='tozeroy',
            fillcolor='rgba(76,175,80,0.1)',
            hoverinfo="text",
            hovertext=conc_hover_text[chunk],
            showlegend=False,
            connectgaps=True
        ))

    # 清理大数组释放内存
    del events, sort_indices, unique_times, inverse_indices, delta_per_time, cumulative
    del conc_times, conc_counts, conc_hover_text
    return traces

def create_plot_layout(
    max_time: float,
    unit: str,
    has_timeline: bool
) -> Dict[str, Any]:
    """创建图表布局配置"""
    xaxis_config = dict(
        **AXIS_CONFIG,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#666',
        spikedash='dot',
        title=f"Relative Time ({unit})",
        range=[0, max_time],
    )

    yaxis_config = dict(
        **AXIS_CONFIG,
        rangemode='nonnegative',
        tickmode='auto',
        nticks=10,
    )

    if has_timeline:
        # 双图模式
        return dict(
            height=1200,
            plot_bgcolor='white',
            xaxis1=dict(
                **xaxis_config,
                matches='x2',
            ),
            yaxis1=dict(
                **yaxis_config,
                title="Request Index",
            ),
            xaxis2=dict(
                **xaxis_config,
            ),
            yaxis2=dict(
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
    else:
        # 单图模式（只有并发图）
        return dict(
            height= 600,
            plot_bgcolor='white',
            xaxis=dict(**xaxis_config),
            yaxis=dict(**yaxis_config,
                         title = "Request Concurrency Count"
            ),
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.9)',
                font_size=12,
                align='left'
            ),
            hovermode='closest',
        )

# ================== 对文件外使用的主函数 ==================
def plot_sorted_request_timelines(
    start_time_list: List[float],
    prefill_latency_list: List[float],
    end_time_list: List[float],
    decode_token_latencies_list: List[List[float]],
    multiturn_group_id_list: List[str],
    output_file: str = "timeline.html",
    unit: str = "s"
) -> None:
    """绘制请求时间线和并发图表"""
    logger = get_logger()
    start_timestamp = time.perf_counter()

    # ===== 1. 数据验证和预处理 =====
    logger.info("Starting request timeline processing...")

    # 验证输入数据
    if not validate_input_data(start_time_list, prefill_latency_list, end_time_list, decode_token_latencies_list):
        return False

    # 数据预处理
    preprocess_start = time.perf_counter()
    adjusted_first_token_times, adjusted_starts, adjusted_ends, is_non_streaming = preprocess_data(
        start_time_list, prefill_latency_list, end_time_list, decode_token_latencies_list
    )

    if is_non_streaming:
        logger.warning("[Non-streaming scenario] The plot will only show the request concurrency chart!")

    n_requests = len(start_time_list)
    has_timeline = not is_non_streaming and adjusted_first_token_times is not None and n_requests > 0
    max_time = np.max(adjusted_ends) if n_requests > 0 else 1.0

    logger.info(f"Data preprocessing completed in {time.perf_counter() - preprocess_start:.4f}s")

    # ===== 2. 生成时间线图轨迹（仅流式场景下） =====
    timeline_traces = []
    if has_timeline:
        logger.info(f"Generating timeline traces for {n_requests} requests...")
        timeline_start = time.perf_counter()
        timeline_traces = generate_timeline_traces(
            adjusted_starts, adjusted_ends, adjusted_first_token_times, multiturn_group_id_list, unit
        )
        logger.info(f"Generated timeline trace chunks in {time.perf_counter() - timeline_start:.4f}s")

    # ===== 3. 生成并发图轨迹 =====
    logger.info("Generating concurrency traces...")
    concurrency_start = time.perf_counter()
    concurrency_traces = generate_concurrency_traces(adjusted_starts, adjusted_ends, unit)

    logger.info(f"Generated concurrency trace chunks in {time.perf_counter() - concurrency_start:.4f}s")

    # ===== 4. 创建图表 =====
    logger.info("Creating figure layout...")
    figure_start = time.perf_counter()

    # 创建布局配置
    layout = create_plot_layout(max_time, unit, has_timeline)

    # 创建图表对象
    if has_timeline:
        fig = make_subplots(
            rows=2,
            cols=1,
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        for trace in timeline_traces:
            fig.add_trace(trace, row=1, col=1)
        for trace in concurrency_traces:
            fig.add_trace(trace, row=2, col=1)
    else:
        fig = go.Figure()
        for trace in concurrency_traces:
            fig.add_trace(trace)

    # 应用布局配置
    fig.update_layout(layout)

    logger.info(f"Figure layout created in {time.perf_counter() - figure_start:.4f}s")

    # ===== 5. 输出HTML =====
    logger.info(f"Writing to {output_file}...")
    write_start = time.perf_counter()

    fig.write_html(
        output_file,
        include_plotlyjs='cdn',
        config=WEBGL_CONFIG,
        auto_open=False,
        full_html=True,
    )

    logger.info(f"HTML written in {time.perf_counter() - write_start:.4f}s")
    total_time = time.perf_counter() - start_timestamp
    logger.info(f"Completed! Total execution time: {total_time:.4f}s")
    return True