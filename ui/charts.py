import pandas as pd
import plotly.express as px


def criar_grafico_tradeoff(metrics):
    """
    Exibe o trade-off entre utilidade e privacidade.

    Quanto mais próximo do canto superior direito,
    melhor o equilíbrio.
    """

    privacidade = 1 - metrics["reidentification_risk"]

    df = pd.DataFrame(
        {
            "Utilidade": [metrics["utility_score"]],
            "Privacidade": [privacidade],
            "Status": ["Atual"],
        }
    )

    fig = px.scatter(
        df,
        x="Utilidade",
        y="Privacidade",
        text="Status",
        size=[30],
        title="Trade-off entre Utilidade e Privacidade",
    )

    fig.update_traces(textposition="top center")

    fig.update_layout(
        xaxis_title="Utilidade",
        yaxis_title="Privacidade",
    )

    return fig