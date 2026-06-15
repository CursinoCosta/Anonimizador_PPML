import pandas as pd
import plotly.express as px


def criar_grafico_tradeoff(metrics, baseline_metrics=None, status=None):
    """
    Exibe o trade-off entre utilidade e privacidade.

    Quanto mais próximo do canto superior direito,
    melhor o equilíbrio.
    """

    privacidade = 1 - metrics["reidentification_risk"]

    _nomes_status = {
        "seguro": "seguro",
        "parcialmente_seguro": "parcialmente seguro",
        "nao_seguro": "não seguro",
    }
    _status_legivel = _nomes_status.get(status, status or "")
    label_atual = f"Atual: {_status_legivel}" if _status_legivel else "Atual"

    pontos = [
        {
            "Utilidade": metrics["utility_score"],
            "Privacidade": privacidade,
            "Status": label_atual,
        }
    ]

    dataset_modificado = baseline_metrics is not None and (
        baseline_metrics["utility_score"] != metrics["utility_score"]
        or baseline_metrics["reidentification_risk"] != metrics["reidentification_risk"]
    )

    if dataset_modificado:
        pontos.append(
            {
                "Utilidade": baseline_metrics["utility_score"],
                "Privacidade": 1 - baseline_metrics["reidentification_risk"],
                "Status": "Status inicial",
            }
        )

    df = pd.DataFrame(pontos)

    cor_status = {
        "seguro": "#22c55e",
        "parcialmente_seguro": "#f59e0b",
        "nao_seguro": "#ef4444",
    }.get(status, "#64748b")

    fig = px.scatter(
        df,
        x="Utilidade",
        y="Privacidade",
        text="Status",
        color="Status",
        size=[30] * len(pontos),
        color_discrete_map={
            label_atual: cor_status,
            "Status inicial": "#64748b",
        },
        title="Trade-off entre Utilidade e Privacidade",
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        xaxis_title="Utilidade",
        yaxis_title="Privacidade",
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
    )

    return fig