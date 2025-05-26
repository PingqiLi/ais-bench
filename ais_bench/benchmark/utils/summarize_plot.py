import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_sorted_request_timelines(request_times, output_file="timeline.html", unit = "ms"):
    # ================== WebGL 加速核心配置 ==================
    WEBGL_CONFIG = {
        'scrollZoom': True,
        'plotGlPixelRatio': 1,  # 渲染分辨率
        'showLink': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['toImage'],
    }
    
    rescale = {
        "s" : 1000,
        "ms" : 1,
    }

    # ================== 计算时间基准 ==================
    all_raw_times = [(float(t) / rescale[unit]) for req in request_times for t in req]
    global_x_min = min(all_raw_times)
    time_offset = global_x_min  # 获取最小时间作为基准偏移量

    # ================== 调整所有时间值为相对值 ==================
    adjusted_requests = []
    for times in request_times:
        # 将每个请求的时间转换为相对时间
        adjusted_times = [(float(t) / rescale[unit]) - time_offset for t in times]
        adjusted_requests.append(adjusted_times)

    # ================== 公共参数计算（使用调整后的时间） ==================
    all_times = [t for req in adjusted_requests for t in req]
    adjusted_x_min = 0  # 最小值归零
    adjusted_x_max = max(all_times)
    x_range = [adjusted_x_min, adjusted_x_max]

    # ================== 第一个图：请求时间线图 ==================
    sorted_requests = sorted(
        [(i, times) for i, times in enumerate(adjusted_requests)],
        key=lambda x: x[1][0],
        reverse=False
    )
	
    line_data = {
        'red': {'x': [], 'y': [], 'text': []},
        'blue': {'x': [], 'y': []},
    }

    for y_pos, (orig_idx, times) in enumerate(sorted_requests):
        if len(times) < 2:
            continue
        
        red_start, red_end = times[0], times[1]
        blue_start, blue_end = times[1], times[-1]
        
        # 红线段数据
        
        line_data['red']['x'].extend([red_start, red_end, None])
        line_data['red']['y'].extend([y_pos, y_pos, None])
        
        # 悬停文本
        hover_text = (
            f"<span style='color:red'>TTFT({unit}): {red_start:.2f} → {red_end:.2f} = {(red_end - red_start):.2f}</span><br>"
            f"<span style='color:blue'>Decode Time({unit}): {blue_start:.2f} → {blue_end:.2f} = {(blue_end - blue_start):.2f}</span><br>"
            f"E2EL({unit}): {red_start:.2f} → {blue_end:.2f} = {(blue_end - red_start):.2f}"
        )
        
        line_data['red']['text'].extend([hover_text, None, None])
        
        # 蓝线段数据
        if blue_end > blue_start:
            line_data['blue']['x'].extend([blue_start, blue_end, None])
            line_data['blue']['y'].extend([y_pos, y_pos, None])
        
        

    # 构建红色轨迹（带悬停信息）
    red_trace = go.Scattergl(
        x=line_data['red']['x'],
        y=line_data['red']['y'],
        mode='lines',
        line=dict(color='red', width=1, shape="hv"),  # 略微减小线宽
        hoverinfo='text',
        hovertext=line_data['red']['text'],
        showlegend=False,
        connectgaps=False
    )

    # 构建蓝色轨迹（无悬停）
    blue_trace = go.Scattergl(
        x=line_data['blue']['x'],
        y=line_data['blue']['y'],
        mode='lines',
        line=dict(color='blue', width=1, shape="hv"),
        hoverinfo='none',
        showlegend=False,
        connectgaps=False
    )

    # ================== 第二个图：请求并发数变化趋势折线图 ==================
    event_changes = []
    for times in adjusted_requests:
        if not times:
            continue
        event_changes.append((times[0], 1))
        event_changes.append((times[-1], -1))

    event_changes.sort(key=lambda x: (x[0], -x[1]))
    
    merged_events = []
    current_time, current_delta = None, 0
    for time, delta in event_changes:
        if time == current_time:
            current_delta += delta
        else:
            if current_time is not None:
                merged_events.append((current_time, current_delta))
            current_time, current_delta = time, delta
    if current_time is not None:
        merged_events.append((current_time, current_delta))

    # 生成阶梯图数据
    time_points, concurrent_counts = [], []
    current = 0
    for time, delta in merged_events:
        time_points.extend([time, time])
        concurrent_counts.extend([current, current + delta])
        current += delta
    
    # 使用轻量级的图形参数
    concurrency_trace = go.Scattergl(
        x=time_points,
        y=concurrent_counts,
        mode='lines',
        line=dict(color='#4CAF50', width=2, shape='hv'),
        fill='tozeroy',
        fillcolor='rgba(76,175,80,0.2)',  # 降低填充透明度
        hovertemplate="Timestamp: %{x:.2f}<br>Request Concurrency Count: %{y}<extra></extra>",
        showlegend=False,
    )

    # ================== 合并图表 ==================
    combined_fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
    combined_fig.add_trace(red_trace, row=1, col=1)
    combined_fig.add_trace(blue_trace, row=1, col=1, )
    combined_fig.add_trace(concurrency_trace, row=2, col=1)

    # 统一坐标轴配置
    axis_config = dict(
        showline=True,
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(211,211,211,0.5)',
        linecolor='black',
    )
    
    xaxis_config = dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#666',
        spikedash='dot',
        title=f"Relative Time ({unit})",
        range=x_range,
        rangeslider=dict(visible=False),  # 禁用范围滑块提升性能
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

    combined_fig.write_html(
        output_file,
        include_plotlyjs='cdn',
        config=WEBGL_CONFIG,
        auto_open=False,
        include_mathjax='cdn',
        full_html=False # 生成更简洁的HTML
    )