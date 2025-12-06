import numpy as np
import plotly.graph_objects as go

def normalize_weights(weights):
    """Normalize attention weights between 0 and 1."""
    weights = np.array(weights, dtype=float)
    if weights.max() == 0:
        return np.zeros_like(weights)
    return weights / weights.max()


def make_head_heatmap(att_matrix, tokens, title="Head attention"):
    """
    Create a Plotly heatmap for a single head.
    
    att_matrix: (seq_len, seq_len)
    tokens: list of token strings
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=att_matrix,
            x=tokens,
            y=tokens,
            zmin=0,
            zmax=float(att_matrix.max()) if att_matrix.max() > 0 else 1,
            colorbar=dict(title="Attention")
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key tokens",
        yaxis_title="Query tokens",
        xaxis=dict(tickangle=45)
    )
    return fig


def make_row_bar(weights, tokens, query_token):
    """
    Bar chart for attention from one token to all tokens.
    """
    fig = go.Figure(
        data=go.Bar(
            x=tokens,
            y=weights
        )
    )
    fig.update_layout(
        title=f"Attention from '{query_token}' to all tokens",
        xaxis_title="Tokens",
        yaxis_title="Attention weight",
        xaxis=dict(tickangle=45)
    )
    return fig


def build_colored_sentence_html(tokens, weights, cls_sep=True):
    """
    Build HTML string with tokens colored by attention weights.
    weights: normalized between 0 and 1.
    """
    weights = normalize_weights(weights)
    html_tokens = []
    for tok, w in zip(tokens, weights):
        # Optionally hide [CLS] and [SEP] or keep them faint
        if cls_sep and tok in ["[CLS]", "[SEP]"]:
            alpha = 0.05
        else:
            alpha = 0.1 + 0.9 * float(w)  # 0.1 -> 1.0

        # Background color: red with variable alpha
        span = (
            f"<span style='"
            f"background-color: rgba(255, 0, 0, {alpha}); "
            f"padding: 3px 5px; margin: 2px; border-radius: 4px; "
            f"display: inline-block; font-family: monospace;"
            f"'>"
            f"{tok}"
            f"</span>"
        )
        html_tokens.append(span)

    return "<div style='line-height: 2.0;'>" + " ".join(html_tokens) + "</div>"


def summarize_head_pattern(att_matrix):
    """
    Very simple heuristic descriptions of head behaviour.
    Returns a short text description.
    """
    att = np.array(att_matrix)
    seq_len = att.shape[0]

    # Self-attention (diagonal)
    diag_mean = np.mean(np.diag(att))

    # Backward vs forward attention
    upper = np.triu(att, 1)
    lower = np.tril(att, -1)
    forward_mean = upper.mean() if upper.size > 0 else 0
    backward_mean = lower.mean() if lower.size > 0 else 0

    # Attention to [CLS] (assuming token 0)
    cls_mean = att[:, 0].mean() if seq_len > 0 else 0

    parts = []
    if diag_mean > 0.3:
        parts.append("strong self-attention")
    if backward_mean > forward_mean * 1.2:
        parts.append("looks more at previous tokens")
    elif forward_mean > backward_mean * 1.2:
        parts.append("looks more at next tokens")
    if cls_mean > 0.3:
        parts.append("pays a lot of attention to [CLS]")

    if not parts:
        return "balanced attention pattern"
    return ", ".join(parts)
