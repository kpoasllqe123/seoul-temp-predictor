import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연도별 평균기온 예측기")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    # 날짜를 datetime형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"].str.strip())
    df["연도"] = df["날짜"].dt.year
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    return df


df_raw = load_data()

# 연도별 관측일수 및 평균기온 계산
yearly_stats = (
    df_raw.groupby("연도")
    .agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean"),
    )
    .reset_index()
)

# 조건 필터링: 2025년 이하 & 관측일수 300일 이상
filtered_df = yearly_stats[
    (yearly_stats["연도"] <= 2025) & (yearly_stats["관측일수"] >= 300)
].copy()

# 데이터 기본 정보 산출
count_years = len(filtered_df)
start_year = int(filtered_df["연도"].min())
end_year = int(filtered_df["연도"].max())

# 2. 회귀 모델 계산 (1908년 기준 경과 연수)
filtered_df["경과연수"] = filtered_df["연도"] - 1908
X = filtered_df["경과연수"]
y = filtered_df["연평균기온"]

# 1차 선형 회귀 계수 산출 (기울기, 절편)
slope, intercept = np.polyfit(X, y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(filtered_df["연도"], filtered_df["연평균기온"])
correlation = corr_matrix[0, 1]

# 3. 화면 UI 레이아웃
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 데이터 분석 요약")
    st.metric(label="분석 대상 연도 개수", value=f"{count_years}개 해")
    st.write(f"- **시작 연도:** {start_year}년")
    st.write(f"- **끝 연도:** {end_year}년")
    st.write(f"- **연도와 기온의 상관계수:** `{correlation:.4f}`")

    st.markdown("---")
    st.subheader("🔮 기온 예측")
    selected_year = st.slider(
        "예측할 연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2026,
        step=1,
    )

    # 선택된 연도 예측값 계산
    pred_elapsed = selected_year - 1908
    predicted_temp = slope * pred_elapsed + intercept

    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temp:.2f} ℃",
    )

with col2:
    st.subheader("📈 서울 연평균기온 변화 추이 및 회귀선")

    # Plotly 시각화 준비
    fig = go.Figure()

    # 관측 데이터 산점도
    fig.add_trace(
        go.Scatter(
            x=filtered_df["연도"],
            y=filtered_df["연평균기온"],
            mode="markers",
            name="실제 관측 기온",
            marker=dict(size=7, color="#1f77b4"),
            hovertemplate="%{x}년: %{y:.2f}℃<extra></extra>",
        )
    )

    # 회귀 직선 (1900년~2100년 전체 구간)
    line_years = np.arange(1900, 2101)
    line_elapsed = line_years - 1908
    line_temps = slope * line_elapsed + intercept

    fig.add_trace(
        go.Scatter(
            x=line_years,
            y=line_temps,
            mode="lines",
            name="회귀 직선",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="%{x}년 추정: %{y:.2f}℃<extra></extra>",
        )
    )

    # 선택된 연도 강조 점 추가
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[predicted_temp],
            mode="markers",
            name=f"{selected_year}년 예측점",
            marker=dict(size=14, color="red", symbol="star"),
            hovertemplate=f"선택 연도: {selected_year}년<br>예상 기온: {predicted_temp:.2f}℃<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="평균기온 (℃)",
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)