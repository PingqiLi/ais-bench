import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple, Any, Dict
from tabulate import tabulate

from .logging import get_logger

logger = get_logger(__name__)


def plot_rps_distribution(
        cumulative_delays: np.ndarray,  # Changed to ndarray for performance
        timing_anomaly_indices: np.ndarray, # show in p1/p2/p3 (red)
        burstiness_anomaly_indices: np.ndarray, # show in p1 (yellow)
        request_rate: float,
        burstiness: float,
        ramp_up_strategy: Optional[str],
        ramp_up_start_rps: Optional[float],
        ramp_up_end_rps: Optional[float],
        output_path: str
) -> None:
    """
    Main function: RPS distribution analysis with three charts
    """
    # 1. Prepare time-RPS data 
    time_points, rps_values = _prepare_time_rps_data(cumulative_delays)

    # 2. Separate points into three categories
    # Create masks for different anomaly types
    timing_mask = np.zeros(len(time_points), dtype=bool)
    if timing_anomaly_indices.size > 0:
        timing_mask[timing_anomaly_indices] = True
    burstiness_mask = np.zeros(len(time_points), dtype=bool)
    if burstiness_anomaly_indices.size > 0:
        burstiness_mask[burstiness_anomaly_indices] = True
    # Create all mask
    valid_mask = ~np.isinf(rps_values)
    invalid_mask = np.isinf(rps_values)
    total_invalid = np.sum(invalid_mask)
    # Apply masks
    normal_mask = valid_mask & ~timing_mask & ~burstiness_mask
    timing_anomaly_mask = valid_mask & timing_mask
    burstiness_anomaly_mask = valid_mask & burstiness_mask
    # Extract points for each category
    normal_time_points = time_points[normal_mask]
    normal_rps_values = rps_values[normal_mask]
    timing_anomaly_time_points = time_points[timing_anomaly_mask]
    timing_anomaly_rps_values = rps_values[timing_anomaly_mask]
    burstiness_anomaly_time_points = time_points[burstiness_anomaly_mask]
    burstiness_anomaly_rps_values = rps_values[burstiness_anomaly_mask]

    # 3. Calculate bins for classic RPS distribution 
    # For classic RPS distribution, burstiness anomalies are treated as normal
    combined_normal_rps = np.concatenate([normal_rps_values, burstiness_anomaly_rps_values])
    display_min, display_max, num_bins = _calculate_rps_bins(
        combined_normal_rps, ramp_up_start_rps, ramp_up_end_rps, request_rate
    )

    # 4. Calculate max y-value for normal values 
    max_normal_y = _calculate_max_normal_y(combined_normal_rps, display_min, display_max, num_bins)
    anomaly_y = max_normal_y * 10 if max_normal_y > 0 else 1

    # 5. Prepare request interval data 
    intervals = _prepare_interval_data(cumulative_delays)
    if timing_anomaly_indices.size > 0:
        normal_intervals, timing_anomaly_intervals = _separate_normal_anomaly_intervals(
            intervals, timing_anomaly_indices
        )
    else:
        # Handle case with no anomalies
        normal_intervals = intervals
        timing_anomaly_intervals = np.array([])

    # 6. Calculate bins for interval distribution 
    display_min_interval, display_max_interval, num_bins_interval = _calculate_interval_bins(
        normal_intervals
    )
    # 7. Calculate max y-value for normal intervals 
    max_normal_y_interval = _calculate_max_normal_y_interval(
        normal_intervals, display_min_interval, display_max_interval, num_bins_interval
    )
    anomaly_y_interval = max_normal_y_interval * 10 if max_normal_y_interval > 0 else 1
    # 8. Create combined title
    target_rate = ramp_up_end_rps if (
        ramp_up_strategy is not None and ramp_up_start_rps is not None and ramp_up_end_rps is not None
        ) else request_rate
    combined_title = _create_combined_title(target_rate, ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps)

    # 9. Create chart and add traces 
    fig = _create_chart_figure(
        normal_time_points, normal_rps_values,
        timing_anomaly_time_points, timing_anomaly_rps_values,
        burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
        combined_normal_rps,
        anomaly_y,
        normal_intervals,
        timing_anomaly_intervals,
        anomaly_y_interval,
        request_rate,
        burstiness,
        ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps,
        display_min, display_max, num_bins,
        display_min_interval, display_max_interval, num_bins_interval,
        time_points,
        combined_title
    )
    # 10. Save chart 
    _save_chart(fig, output_path)
    # 11. Log statistics
    _log_statistics(
        cumulative_delays,
        normal_rps_values,
        timing_anomaly_rps_values,
        burstiness_anomaly_rps_values,
        time_points,
        intervals,
        normal_intervals,
        timing_anomaly_intervals,
        target_rate,
        burstiness,
        total_invalid
    )


def _prepare_time_rps_data(cumulative_delays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time-RPS distribution data 
    Uses vectorized operations for better performance
    """
    intervals = np.diff(cumulative_delays, prepend=0.0)
    rps_values = np.divide(1.0, intervals, where=intervals > 1e-6, out=np.full_like(intervals, np.inf))
    time_points = np.round(cumulative_delays, 3)
    rps_values = np.round(rps_values, 3)
    return time_points, rps_values


def _separate_normal_anomaly(
    time_points: np.ndarray, 
    rps_values: np.ndarray, 
    timing_anomaly_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate normal and anomaly values 
    Uses boolean indexing for efficient separation
    """
    anomaly_mask = np.zeros(len(time_points), dtype=bool)
    if timing_anomaly_indices.size > 0:
        anomaly_mask[timing_anomaly_indices] = True

    valid_mask = ~np.isinf(rps_values)
    normal_mask = valid_mask & ~anomaly_mask
    anomaly_mask = valid_mask & anomaly_mask
    return (
        time_points[normal_mask],
        rps_values[normal_mask],
        time_points[anomaly_mask],
        rps_values[anomaly_mask]
    )


def _calculate_rps_bins(
    finite_normal_rps: np.ndarray,
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    request_rate: float
) -> Tuple[float, float, int]:
    """
    Calculate bins for classic RPS distribution 
    Uses efficient numpy operations for min/max calculations
    """
    if finite_normal_rps.size == 0:
        return 0.0, 1.0, 1

    data_min = np.min(finite_normal_rps)
    data_max = np.max(finite_normal_rps)

    ref_points = []
    if ramp_up_start_rps is not None:
        ref_points.append(ramp_up_start_rps)
    if ramp_up_end_rps is not None:
        ref_points.append(ramp_up_end_rps)
    if request_rate is not None:
        ref_points.append(request_rate)

    if ref_points:
        ref_min = min(ref_points)
        ref_max = max(ref_points)
        display_min = min(data_min, ref_min) * 0.9
        display_max = max(data_max, ref_max) * 1.1
    else:
        display_min = data_min * 0.9
        display_max = data_max * 1.1

    if ramp_up_end_rps is not None:
        display_max = min(display_max, ramp_up_end_rps * 1.5)

    if finite_normal_rps.size == 1 or np.isclose(data_min, data_max):
        display_min = max(0, data_min - 1)
        display_max = data_max + 1
        return float(display_min), float(display_max), 1

    std_dev = np.std(finite_normal_rps)
    if std_dev < 1e-6:
        data_range = data_max - data_min
        h = 3.5 * data_range / (finite_normal_rps.size ** (1/3))
    else:
        h = 3.5 * std_dev / (finite_normal_rps.size ** (1/3))

    bin_width = max(1e-6, h)
    num_bins = int((display_max - display_min) / bin_width)
    num_bins = max(10, min(100, num_bins))
    return float(display_min), float(display_max), num_bins


def _calculate_max_normal_y(
    finite_normal_rps: np.ndarray,
    display_min: float,
    display_max: float,
    num_bins: int
) -> float:
    """
    Calculate max y-value for normal values 
    Uses numpy histogram for efficient bin calculation
    """
    if finite_normal_rps.size == 0:
        return 1.0
    # Calculate histogram
    hist, _ = np.histogram(
        finite_normal_rps,
        bins=num_bins,
        range=(display_min, display_max)
    )
    return float(hist.max()) if hist.size > 0 else 1.0


def _calculate_interval_bins(
    normal_intervals: np.ndarray
) -> Tuple[float, float, int]:
    """
    Calculate bins for interval distribution 
    Efficiently calculates bin parameters using numpy
    """
    if normal_intervals.size == 0:
        return 0.0, 1.0, 1

    data_min = np.min(normal_intervals)
    data_max = np.max(normal_intervals)
    display_min_interval = max(0, data_min * 0.9)
    display_max_interval = data_max * 1.1

    if normal_intervals.size == 1 or np.isclose(data_min, data_max):
        display_min_interval = max(0, data_min - 0.001)
        display_max_interval = data_max + 0.001
        return float(display_min_interval), float(display_max_interval), 1

    std_dev = np.std(normal_intervals)

    if std_dev < 1e-6:
        data_range = data_max - data_min
        h = 3.5 * data_range / (normal_intervals.size ** (1/3))
    else:
        h = 3.5 * std_dev / (normal_intervals.size ** (1/3))

    bin_width = max(1e-6, h)
    num_bins_interval = int((display_max_interval - display_min_interval) / bin_width)
    num_bins_interval = max(10, min(100, num_bins_interval))
    return float(display_min_interval), float(display_max_interval), num_bins_interval


def _calculate_max_normal_y_interval(
    normal_intervals: np.ndarray,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int
) -> float:
    """
    Calculate max y-value for normal intervals 
    Efficiently calculates max bin height using numpy histogram
    """
    if normal_intervals.size == 0:
        return 1.0

    hist, _ = np.histogram(
        normal_intervals,
        bins=num_bins_interval,
        range=(display_min_interval, display_max_interval)
    )
    return float(hist.max()) if hist.size > 0 else 1.0


def _calculate_theoretical_ramp(
    total_requests: int,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    request_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Computing the theoretical climb path, correctly accounting for climb strategy."""

    request_indices = np.arange(total_requests)
    progress = request_indices / max(total_requests - 1, 1)

    if ramp_up_strategy == "linear" and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        theoretical_rates = ramp_up_start_rps + (ramp_up_end_rps - ramp_up_start_rps) * progress
    elif ramp_up_strategy == "exponential" and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        ratio = ramp_up_end_rps / ramp_up_start_rps
        theoretical_rates = ramp_up_start_rps * (ratio ** progress)
    else:
        theoretical_rates = np.full(total_requests, request_rate)

    theoretical_intervals = np.zeros(total_requests)
    non_zero_mask = theoretical_rates > 0
    theoretical_intervals[non_zero_mask] = 1.0 / theoretical_rates[non_zero_mask]

    cumulative_theoretical_times = np.cumsum(theoretical_intervals)
    return cumulative_theoretical_times, theoretical_rates


def _create_combined_title(
    target_rate: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float]
) -> str:
    """Create combined title for the chart"""
    title = f"Request Per Second(RPS) Distribution Analysis | Target Rate: {target_rate:.2f}"
    if ramp_up_strategy:
        title += f" | {ramp_up_strategy.capitalize()} Ramp-up: {ramp_up_start_rps or 0:.1f}→{ramp_up_end_rps or 0:.1f}"
    return title


def _prepare_interval_data(cumulative_delays: np.ndarray) -> np.ndarray:
    """Prepare interval data """
    if cumulative_delays.size == 0:
        return np.array([])
    # Calculate intervals using vectorized diff
    return np.diff(cumulative_delays, prepend=0.0)


def _separate_normal_anomaly_intervals(
    intervals: np.ndarray,
    timing_anomaly_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate normal and anomaly intervals """
    if intervals.size == 0:
        return np.array([]), np.array([])
    # Create anomaly mask
    anomaly_mask = np.zeros(intervals.size, dtype=bool)
    if timing_anomaly_indices.size > 0:
        # Convert indices to mask for efficient indexing
        anomaly_mask[timing_anomaly_indices] = True
    return intervals[~anomaly_mask], intervals[anomaly_mask]


def _density_based_sampling(time_points, values, max_samples=1000, num_bins=100):
    """
    Time-density-based sampling method
    Args:
        time_points: Array of time points
        values: Corresponding array of values
        max_samples: Maximum number of sample points
        num_bins: Number of bins for time axis partitioning
    Returns:
        Sampled time points and values
    """
    n = len(time_points)
    if n <= max_samples:
        return time_points, values
    min_time = np.min(time_points)
    max_time = np.max(time_points)

    bins = np.linspace(min_time, max_time, num_bins)
    bin_indices = np.searchsorted(bins, time_points, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)

    total_points = n
    sample_counts = np.round(bin_counts * max_samples / total_points).astype(int)
    total_samples = np.sum(sample_counts)

    if total_samples > max_samples:
        reduce_count = total_samples - max_samples
        sorted_indices = np.argsort(-sample_counts)
        for i in sorted_indices:
            if reduce_count <= 0:
                break
            if sample_counts[i] > 0:
                reduction = min(reduce_count, sample_counts[i])
                sample_counts[i] -= reduction
                reduce_count -= reduction

    indices = np.arange(n)
    bin_idx_arr = bin_indices
    sampled_indices = []

    for bin_idx in range(num_bins):
        if sample_counts[bin_idx] == 0:
            continue
        bin_mask = (bin_idx_arr == bin_idx)
        bin_indices_arr = indices[bin_mask]
        if len(bin_indices_arr) <= sample_counts[bin_idx]:
            sampled_indices.append(bin_indices_arr)
        else:
            selected = np.random.choice(
                bin_indices_arr, 
                size=sample_counts[bin_idx], 
                replace=False
            )
            sampled_indices.append(selected)

    if sampled_indices:
        sampled_indices = np.concatenate(sampled_indices)
    else:
        sampled_indices = np.array([], dtype=int)

    return time_points[sampled_indices], values[sampled_indices]


def _create_chart_figure(
    normal_time_points: np.ndarray,
    normal_rps_values: np.ndarray,
    timing_anomaly_time_points: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_time_points: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    finite_normal_rps: np.ndarray,
    anomaly_y: float,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    anomaly_y_interval: float,
    request_rate: float,
    burstiness: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    display_min: float,
    display_max: float,
    num_bins: int,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int,
    time_points: np.ndarray,
    combined_title: str
) -> go.Figure:
    """
    Create chart figure with traces 
    Uses efficient Plotly methods and avoids unnecessary data copies
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Time vs RPS - Distribution',
            'RPS vs Request Count - Distribution',
            f'Gamma Distribution (burstiness: {burstiness})',
            'Legend Explanation'),
        specs=[
            [{"type": "scatter", "rowspan": 1}, {"type": "xy", "rowspan": 1}],
            [{"type": "xy", "rowspan": 1}, {"type": "table", "rowspan": 1}]
        ],
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    legend_trace_name_prefix = [
        "Time - RPS: ",
        "RPS - Request Count: ",
        "Gamma Dist: ",
    ]
    traces_names_dict = dict()

    curr_chart_name = legend_trace_name_prefix[0]
    traces_names_dict[curr_chart_name]=_add_time_rps_traces(fig, 
                                            normal_time_points, normal_rps_values,
                                            timing_anomaly_time_points, timing_anomaly_rps_values,
                                            burstiness_anomaly_time_points, burstiness_anomaly_rps_values,
                                            request_rate, 
                                            ramp_up_strategy, ramp_up_start_rps, ramp_up_end_rps,
                                            time_points, curr_chart_name, row=1, col=1)
    
    curr_chart_name = legend_trace_name_prefix[1]
    traces_names_dict[curr_chart_name]=_add_classic_rps_traces(fig, finite_normal_rps, timing_anomaly_rps_values, 
                                            anomaly_y, request_rate, display_min, display_max, 
                                            num_bins, curr_chart_name, row=1, col=2)
    
    curr_chart_name = legend_trace_name_prefix[2]
    traces_names_dict[curr_chart_name]=_add_interval_traces(fig, normal_intervals, timing_anomaly_intervals, 
                                            anomaly_y_interval, display_min_interval, 
                                            display_max_interval, num_bins_interval, curr_chart_name, row=2, col=1)
    
    _add_legend_explanation_table(fig, traces_names_dict, row=2, col=2)

    fig.update_traces(
        legendgroup="time_rps",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix[0])
    )
    fig.update_traces(
        legendgroup="rps_dist",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix[1])
    )
    fig.update_traces(
        legendgroup="interval_dist",
        selector=lambda trace: trace.name is not None and trace.name.startswith(legend_trace_name_prefix[2])
    )
    fig.update_layout(
        title=combined_title,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        title_font=dict(size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10),
            groupclick="toggleitem",
            tracegroupgap=30
        ),
        margin=dict(l=50, r=50, t=150, b=200),
        hovermode="closest",
        plot_bgcolor="rgba(240,240,240,0.9)",
        paper_bgcolor="rgba(255,255,255,1)",
        autosize=True
    )

    return fig


def _add_time_rps_traces(
    fig: go.Figure,
    normal_time_points: np.ndarray,
    normal_rps_values: np.ndarray,
    timing_anomaly_time_points: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_time_points: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    request_rate: float,
    ramp_up_strategy: Optional[str],
    ramp_up_start_rps: Optional[float],
    ramp_up_end_rps: Optional[float],
    time_points: np.ndarray,
    legend_trace_name_prefix: Optional[str],
    row: int = 1,
    col: int = 1
) -> List[str]:
    """Add traces for time-RPS distribution with separate anomaly types"""
    scatter_class = go.Scattergl if len(normal_time_points) > 10000 else go.Scatter
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal RPS'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        scatter_class(
            x=normal_time_points,
            y=normal_rps_values,
            mode='lines+markers',
            marker=dict(size=3, color='rgba(31, 119, 180, 0.3)', opacity=0.3),
            line=dict(color='rgba(31, 119, 180, 0.3)'),
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_time_points.size > 0:
        sampled_time, sampled_rps = _density_based_sampling(
            timing_anomaly_time_points, 
            timing_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_time,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='red', opacity=1.0, symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    if burstiness_anomaly_time_points.size > 0:
        sampled_time, sampled_rps = _density_based_sampling(
            burstiness_anomaly_time_points, 
            burstiness_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Burstiness Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_time,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='yellow', opacity=1.0, symbol='square'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    window_size = 50
    if normal_rps_values.size > 0:
        finite_mask = np.isfinite(normal_rps_values)
        finite_rps = normal_rps_values[finite_mask]
        finite_time = normal_time_points[finite_mask]
        if finite_rps.size >= window_size:
            moving_avg = np.convolve(
                finite_rps, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            moving_avg_time = finite_time[window_size-1:]
        else:
            moving_avg = np.array([])
            moving_avg_time = np.array([])
    else:
        moving_avg = np.array([])
        moving_avg_time = np.array([])
    if moving_avg.size > 0:
        curr_trace_name = f'{legend_trace_name_prefix}{window_size}-point Moving Avg'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=moving_avg_time,
                y=moving_avg,
                mode='lines',
                line=dict(color='purple', width=2),
                name=curr_trace_name,
                showlegend=True,
            ),
            row=row, col=col
        )

    if ramp_up_strategy in ("linear", "exponential") and ramp_up_start_rps is not None and ramp_up_end_rps is not None:
        cumulative_theoretical_times, theoretical_rates = _calculate_theoretical_ramp(
            total_requests=len(normal_time_points) + len(timing_anomaly_time_points) + len(burstiness_anomaly_time_points),
            ramp_up_strategy=ramp_up_strategy,
            ramp_up_start_rps=ramp_up_start_rps,
            ramp_up_end_rps=ramp_up_end_rps,
            request_rate=request_rate
        )
        valid_mask = np.isfinite(theoretical_rates) & np.isfinite(cumulative_theoretical_times)
        if np.any(valid_mask):
            curr_trace_name = f'{legend_trace_name_prefix}Theoretical Ramp-up'
            traces_names.append(curr_trace_name)
            fig.add_trace(
                go.Scatter(
                    x=cumulative_theoretical_times[valid_mask],
                    y=theoretical_rates[valid_mask],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name=curr_trace_name,
                    showlegend=True,
                ),
                row=row, col=col
            )

    fig.update_xaxes(title_text="Time (seconds)", row=row, col=col)
    fig.update_yaxes(title_text="RPS", row=row, col=col)
    return traces_names


def _add_classic_rps_traces(
    fig: go.Figure,
    finite_normal_rps: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    anomaly_y: float,
    request_rate: float,
    display_min: float,
    display_max: float,
    num_bins: int,
    legend_trace_name_prefix: Optional[str],
    row: int = 1,
    col: int = 2
) -> List[str]:
    """Add traces for classic RPS distribution (only timing anomalies)"""
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal Request Count'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        go.Histogram(
            x=finite_normal_rps,
            xbins=dict(start=display_min, end=display_max, size=(display_max - display_min) / num_bins),
            marker_color='#2ca02c',
            opacity=0.6,
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_rps_values.size > 0:
        sampled_rps, _ = _density_based_sampling(
            timing_anomaly_rps_values, 
            timing_anomaly_rps_values,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_rps,
                y=sampled_rps,
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )

    fig.update_xaxes(title_text="RPS", row=row, col=col)
    fig.update_yaxes(title_text="Request Count", row=row, col=col)
    return traces_names


def _add_interval_traces(
    fig: go.Figure,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    anomaly_y_interval: float,
    display_min_interval: float,
    display_max_interval: float,
    num_bins_interval: int,
    legend_trace_name_prefix: Optional[str],
    row: int = 2,
    col: int = 1
) -> List[str]:
    """Add traces for interval distribution (only timing anomalies)"""
    traces_names = []
    curr_trace_name = f'{legend_trace_name_prefix}Normal Intervals'
    traces_names.append(curr_trace_name)

    fig.add_trace(
        go.Histogram(
            x=normal_intervals,
            xbins=dict(
                start=display_min_interval,
                end=display_max_interval,
                size=(display_max_interval - display_min_interval) / num_bins_interval
            ),
            marker_color='#9467bd',
            opacity=0.6,
            name=curr_trace_name,
            showlegend=True
        ),
        row=row, col=col
    )

    if timing_anomaly_intervals.size > 0:
        sampled_intervals, _ = _density_based_sampling(
            timing_anomaly_intervals, 
            timing_anomaly_intervals,
            max_samples=1000
        )
        curr_trace_name = f'{legend_trace_name_prefix}Time Interval Caused Anomaly'
        traces_names.append(curr_trace_name)
        fig.add_trace(
            go.Scatter(
                x=sampled_intervals,
                y=sampled_intervals,
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-up'),
                name=curr_trace_name,
                showlegend=True,
                visible='legendonly'
            ),
            row=row, col=col
        )
    # Set axis labels
    fig.update_xaxes(title_text="Interval Time (seconds)", row=row, col=col)
    fig.update_yaxes(title_text="Request Count", row=row, col=col)
    return traces_names


def _add_legend_explanation_table(
    fig: go.Figure,
    traces_names_dict: dict,
    row: int = 2,
    col: int = 2
) -> None:
    """Add expanded legend explanation table to the chart with grouped display"""
    prefix_list = list(traces_names_dict.keys())
    base_descriptions = {
        f"{prefix_list[0]}Normal RPS": ("请求率值(排除异常值)", "实际间隔时间 ≥ 1ms 且与期望间隔的偏差 ≤ 50%", "期望间隔 = 1 / 当前请求率", "蓝色连线+点状标记"),
        f"{prefix_list[0]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "系统无法可靠处理低于1ms的时间间隔", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[0]}Burstiness Caused Anomaly": ("突发性异常点", "Gamma分布生成的间隔时间显著偏离期望值", "|实际间隔 - 期望间隔| / 期望间隔 > 50%", "黄色方形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[0]}50-point Moving Avg": ("移动平均线", "MA = Σ(RPS_i to RPS_{i+49}) / 50", "对连续50个正常RPS值取平均值", "紫色实线"),
        f"{prefix_list[0]}Theoretical Ramp-up": ("理论爬升线", "线性: RPS = start + (end - start)*progress\n指数: RPS = start * (ratio^progress)", "根据配置的爬升策略计算期望RPS值", "绿色虚线"),
        f"{prefix_list[1]}Normal Request Count": ("请求个数(排除异常值)", "实际间隔时间 ≥ 1ms 且与期望间隔的偏差 ≤ 50%", "落于该横坐标区间内的请求频数", "绿色直方图"),
        f"{prefix_list[1]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "系统无法可靠处理低于1ms的时间间隔", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)"),
        f"{prefix_list[2]}Normal Intervals": ("请求间隔时间分布(排除异常值)", "自适应计算最优分桶范围", "统计正常间隔时间的分布频率", "紫色直方图"),
        f"{prefix_list[2]}Time Interval Caused Anomaly": ("请求间隔时间异常点", "仅包含时间间隔异常点", "实际间隔时间 < 1ms", "红色三角形标记<br>(密度采样最多1000个点)")
    }

    group_names = []
    trace_names = []
    meanings = []
    calculations = []
    criteria = []
    visualizations = []
    
    for group_prefix, trace_list in traces_names_dict.items():
        if not trace_list:
            continue
        group_names.append(group_prefix)
        trace_names.append("")
        meanings.append("") 
        calculations.append("")
        criteria.append("")
        visualizations.append("")
        for trace_name in trace_list:
            base_name = trace_name.replace(group_prefix, "").strip()
            description = base_descriptions.get(trace_name, ("", "", "", ""))
            group_names.append("")
            trace_names.append(base_name)
            meanings.append(description[0])
            calculations.append(description[1])
            criteria.append(description[2])
            visualizations.append(description[3])
    
    total_rows = 0
    for group_prefix, trace_list in traces_names_dict.items():
        if trace_list:
            total_rows += len(trace_list) + 1  # +1 for group header
    
    base_row_height = 25
    max_table_height = 400
    row_height = min(base_row_height, max_table_height / max(total_rows, 1))
    base_font_size = 14
    font_size = max(10, base_font_size - max(0, total_rows - 10) // 2)

    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>组名</b>', '<b>图例项</b>', '<b>含义</b>', '<b>计算原理</b>', '<b>判断方式</b>', '<b>可视化表现</b>'],
                font=dict(size=font_size * 1.2, color='white'),
                fill_color='#4a5568',
                align='left',
                height=row_height * 1.5,
            ),
            cells=dict(
                values=[group_names, trace_names, meanings, calculations, criteria, visualizations],
                font=dict(size=font_size),
                align='left',
                fill_color='rgba(247, 250, 252, 0.9)',
                height=row_height,
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            columnwidth=[100, 150, 120, 180, 180, 150]
        ),
        row=row, col=col
    )

    fig.update_xaxes(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False, 
        row=row, col=col,
        domain=[0, 1]
    )
    fig.update_yaxes(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False, 
        row=row, col=col,
        domain=[0, 1]
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=100, b=50),
        autosize=True
    )


def _save_chart(fig: go.Figure, output_path: str) -> None:
    """Save chart to HTML file """
    fig.write_html(
        output_path,
        config={
            'scrollZoom': True,
            'plotGlPixelRatio': 1,
            'showLink': False,
            'displaylogo': False,
            'responsive': True
        }
    )
    logger.info(f"RPS distribution charts saved to {output_path}")


def _log_statistics(
    cumulative_delays: np.ndarray,
    finite_normal_rps: np.ndarray,
    timing_anomaly_rps_values: np.ndarray,
    burstiness_anomaly_rps_values: np.ndarray,
    time_points: np.ndarray,
    intervals: np.ndarray,
    normal_intervals: np.ndarray,
    timing_anomaly_intervals: np.ndarray,
    target_rate: float,
    burstiness: float,
    total_invalid: int
) -> None:
    """Log essential statistical information in a compact tabular format"""
    total_normal = finite_normal_rps.size
    total_timing_anomaly = timing_anomaly_rps_values.size
    total_burstiness_anomaly = burstiness_anomaly_rps_values.size

    total_requests = cumulative_delays.size
    calculated_total = total_normal + total_timing_anomaly + total_burstiness_anomaly  + total_invalid
    if total_requests != calculated_total:
        logger.warning(f"Request count mismatch! Total: {total_requests}, "
                      f"Calculated: {calculated_total}. Adjusting counts.")
        scale_factor = total_requests / calculated_total
        total_normal = int(total_normal * scale_factor)
        total_timing_anomaly = int(total_timing_anomaly * scale_factor)
        total_burstiness_anomaly = int(total_burstiness_anomaly * scale_factor)
        total_invalid = total_requests - total_normal - total_timing_anomaly - total_burstiness_anomaly
    
    core_stats = [
        ("Total Requests", total_requests),
        ("Request Classification", 
         f"Normal: {total_normal} | "
         f"Timing Anomaly: {total_timing_anomaly} | "
         f"Burstiness Anomaly: {total_burstiness_anomaly} | "
         f"Infinite RPS Anomaly: {total_invalid}"),
        ("Target Rate", f"{target_rate:.2f} RPS"),
        ("Burstiness", f"{burstiness:.3f}")
    ]

    rps_stats = []
    if finite_normal_rps.size > 0:
        rps_stats.extend([
            ("Normal RPS", f"{finite_normal_rps.mean():.2f} ± {finite_normal_rps.std():.2f}"),
            ("Normal RPS Range", f"{finite_normal_rps.min():.2f}-{finite_normal_rps.max():.2f}")
        ])

    interval_stats = [
        ("Interval Stats", 
         f"Avg: {intervals.mean():.3f}s | "
         f"Min: {intervals.min():.3f}s | "
         f"Max: {intervals.max():.3f}s"),
        ("Interval Classification", 
         f"Normal (Normal + Burstiness Anomaly): {normal_intervals.size} | "
         f"Anomaly (Timing Anomaly + Infinite RPS Anomaly): {timing_anomaly_intervals.size}")
    ]

    stats = core_stats + rps_stats + interval_stats
    table = tabulate(
        stats,
        headers=["Metric", "Value"],
        tablefmt="simple",
        stralign="left",
        numalign="right"
    )

    title = "Request Per Second (RPS) Distribution Summary".center(len(table.split('\n')[0]))
    logger.info(f"\n{title}\n{table}\n")