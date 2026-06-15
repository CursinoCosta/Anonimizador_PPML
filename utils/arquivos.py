import io
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


@st.cache_data
def carregar_dataframe(arquivo_carregado):
    """Carrega o arquivo CSV num DataFrame do Pandas de forma eficiente."""
    if arquivo_carregado is not None:
        try:
            df = pd.read_csv(arquivo_carregado)
            return df
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            return None
    return None


def exportar_csv(df: pd.DataFrame) -> bytes:
    """Serializa o DataFrame transformado para bytes CSV com encoding UTF-8."""
    return df.to_csv(index=False).encode("utf-8")


def gerar_relatorio_pdf(
    linhas_resumo: list[str],
    historico: list[dict],
    df_conformidade: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
    )
    style_section = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=14,
        spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=4,
        leading=14,
    )

    elements = []

    elements.append(Paragraph("Relatório de Anonimização", style_title))
    elements.append(Spacer(1, 0.3 * cm))

    elements.append(Paragraph("Resumo da Anonimização", style_section))
    for linha in linhas_resumo:
        texto = linha.replace("**", "")
        elements.append(Paragraph(f"• {texto}", style_body))

    if historico:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph("Transformações aplicadas:", style_section))

        colunas = list(historico[0].keys())
        table_data = [colunas] + [[str(row.get(c, "")) for c in colunas] for row in historico]

        table = Table(table_data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6b7280")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("Relatório de Conformidade", style_section))

    conf_data = [df_conformidade.columns.tolist()] + df_conformidade.values.tolist()
    conf_table = Table(conf_data, repeatRows=1, hAlign="LEFT")

    row_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6b7280")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    status_col = df_conformidade.columns.tolist().index("Status")
    for i, row in enumerate(df_conformidade.itertuples(index=False), start=1):
        status = getattr(row, "Status")
        bg = colors.HexColor("#d4edda") if "Atende" in status else colors.HexColor("#f8d7da")
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))

    conf_table.setStyle(TableStyle(row_styles))
    elements.append(conf_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()