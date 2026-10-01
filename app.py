"""Indian Stock Market Analytics Dashboard
Python | Pandas | Plotly | Dash | Scikit-learn

Run:  python app.py      (expects data/All_Stocks_Data_Cleaned.csv and assets/style.css)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# ---------------------------------------------------------------- config
DATA_PATH = Path(__file__).resolve().parent / "data" / "All_Stocks_Data_Cleaned.csv"
MAX_ML_ROWS = 60_000
GAIN, LOSS, ACCENT, BLUE = "#2fbf8f", "#ef5b5b", "#f2a541", "#5b8def"
GRAPH_CFG = {"displayModeBar": False}


# ---------------------------------------------------------------- data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Closing_Price", "Daily_Return"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (df.dropna(subset=["Symbol", "Date", "Closing_Price"])
            .sort_values(["Symbol", "Date"]).reset_index(drop=True))

    price = df.groupby("Symbol")["Closing_Price"]
    df["Year"] = df["Date"].dt.year
    df["MA_7D"] = price.transform(lambda s: s.rolling(7).mean())
    df["MA_30D"] = price.transform(lambda s: s.rolling(30).mean())
    df["Drawdown"] = (df["Closing_Price"] / price.cummax() - 1) * 100
    return df


def build_stock_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stock: return, risk and consistency metrics."""
    stats = (df.dropna(subset=["Daily_Return"]).groupby("Symbol")["Daily_Return"]
               .agg(Avg_Return="mean", Volatility="std", Trading_Days="count",
                    Positive_Days=lambda x: (x > 0).sum(),
                    Negative_Days=lambda x: (x < 0).sum()))
    stats["Avg_Return_Pct"] = stats["Avg_Return"] * 100
    stats["Volatility_Pct"] = stats["Volatility"] * 100
    stats["Max_Drawdown"] = df.groupby("Symbol")["Drawdown"].min()
    first = df.groupby("Symbol")["Closing_Price"].first()
    last = df.groupby("Symbol")["Closing_Price"].last()
    stats["Total_Return"] = (last / first - 1) * 100
    return stats.reset_index()


def build_yearly_returns(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (df.groupby(["Symbol", "Year"])["Closing_Price"]
                .agg(First="first", Last="last").reset_index())
    yearly["Yearly_Return"] = (yearly["Last"] / yearly["First"] - 1) * 100
    return yearly


# ---------------------------------------------------------------- machine learning
ML_FEATURES = ["Lag_Return_1D", "Lag_Return_5D", "Price_Momentum_7D",
               "Price_Momentum_30D", "Volatility_7D", "Volatility_30D"]


def train_model(df: pd.DataFrame) -> dict:
    """Random Forest predicting next-day direction, with a time-based split."""
    d = df[["Symbol", "Date", "Closing_Price", "Daily_Return"]].copy()
    ret = d.groupby("Symbol")["Daily_Return"]
    price = d.groupby("Symbol")["Closing_Price"]

    d["Lag_Return_1D"] = ret.shift(1)
    d["Lag_Return_5D"] = ret.shift(5)
    d["Price_Momentum_7D"] = d["Closing_Price"] / price.shift(7) - 1
    d["Price_Momentum_30D"] = d["Closing_Price"] / price.shift(30) - 1
    d["Volatility_7D"] = ret.transform(lambda s: s.rolling(7).std())
    d["Volatility_30D"] = ret.transform(lambda s: s.rolling(30).std())
    d["Next_Day_Return"] = ret.shift(-1)

    d = d.dropna(subset=ML_FEATURES + ["Next_Day_Return"])
    d = d.sort_values("Date").tail(MAX_ML_ROWS)
    d["Target"] = (d["Next_Day_Return"] > 0).astype(int)

    split = int(len(d) * 0.8)
    X_tr, X_te = d[ML_FEATURES].iloc[:split], d[ML_FEATURES].iloc[split:]
    y_tr, y_te = d["Target"].iloc[:split], d["Target"].iloc[split:]

    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42,
                                   n_jobs=-1, class_weight="balanced").fit(X_tr, y_tr)
    pred = model.predict(X_te)
    share_up = y_te.mean()
    return {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred, zero_division=0),
        "baseline": max(share_up, 1 - share_up),  # always guess the majority class
        "matrix": confusion_matrix(y_te, pred),
        "importance": pd.DataFrame({"Feature": ML_FEATURES,
                                    "Importance": model.feature_importances_})
                        .sort_values("Importance"),
    }


# ---------------------------------------------------------------- prepare everything once
df = load_data(DATA_PATH)
stats = build_stock_stats(df)
yearly = build_yearly_returns(df)
ml = train_model(df)

years = sorted(int(y) for y in df["Year"].unique())
symbols = sorted(df["Symbol"].unique())
stats_by_symbol = stats.set_index("Symbol")


# ---------------------------------------------------------------- figure helpers
def style(fig: go.Figure, title: str | None = None, unified: bool = False) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, system-ui, sans-serif", color="#e6eaf2"),
        margin=dict(l=45, r=20, t=60, b=40),
        hovermode="x unified" if unified else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#263149", zeroline=False)
    fig.update_yaxes(gridcolor="#263149", zeroline=False)
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#8b97ad"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig)


def hbar(data, x, y, color, title, x_label) -> go.Figure:
    fig = px.bar(data, x=x, y=y, orientation="h", labels={x: x_label, y: "Stock"})
    fig.update_traces(marker_color=color)
    return style(fig, title)


# ---------------------------------------------------------------- static figures (no inputs needed)
def risk_return_figure() -> go.Figure:
    fig = px.scatter(stats, x="Volatility_Pct", y="Avg_Return_Pct", hover_name="Symbol",
                     labels={"Volatility_Pct": "Volatility (% daily)",
                             "Avg_Return_Pct": "Average return (% daily)"})
    fig.update_traces(marker=dict(size=8, color=BLUE, opacity=0.75))
    return style(fig, "Risk vs return, all stocks")


def volatility_figure() -> go.Figure:
    top = stats.nlargest(10, "Volatility_Pct").sort_values("Volatility_Pct")
    return hbar(top, "Volatility_Pct", "Symbol", ACCENT,
                "10 most volatile stocks", "Volatility (% daily)")


def drawdown_figure() -> go.Figure:
    worst = stats.nsmallest(10, "Max_Drawdown").sort_values("Max_Drawdown", ascending=False)
    return hbar(worst, "Max_Drawdown", "Symbol", LOSS,
                "10 deepest maximum drawdowns", "Maximum drawdown (%)")


def importance_figure() -> go.Figure:
    return hbar(ml["importance"], "Importance", "Feature", BLUE,
                "Random Forest feature importance", "Importance")


def confusion_figure() -> go.Figure:
    labels = ["Down", "Up"]
    fig = px.imshow(ml["matrix"], text_auto=True, x=labels, y=labels,
                    color_continuous_scale=["#161e2e", BLUE],
                    labels={"x": "Predicted", "y": "Actual", "color": "Count"})
    fig.update_coloraxes(showscale=False)
    return style(fig, "Confusion matrix (test set)")


# ---------------------------------------------------------------- layout components
def kpi(label, value, note=""):
    return html.Div(className="kpi-card", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(note, className="kpi-note"),
    ])


def section(title, description):
    return html.Div(className="section-head", children=[
        html.H2(title, className="section-title"),
        html.P(description, className="section-description"),
    ])


def card(title, child, description=None, class_name=""):
    kids = [html.Div(title, className="card-title")]
    if description:
        kids.append(html.P(description, className="card-description"))
    kids.append(child)
    return html.Div(className=f"dashboard-card {class_name}".strip(), children=kids)


def graph(graph_id=None, figure=None):
    return dcc.Graph(id=graph_id, figure=figure, config=GRAPH_CFG) if graph_id \
        else dcc.Graph(figure=figure, config=GRAPH_CFG)


def dropdown(dropdown_id, options, value, searchable=False):
    return dcc.Dropdown(id=dropdown_id, options=options, value=value,
                        searchable=searchable, clearable=False, className="dropdown")


# ---------------------------------------------------------------- app + layout
app = Dash(__name__, title="Indian Stock Market Analytics")
server = app.server

app.layout = html.Div(className="dashboard-container", children=[
    html.Header(className="header", children=[
        html.Div([
            html.H1("Indian Stock Market Analytics", className="brand-title"),
            html.P("Prices, returns, risk and a next-day direction model "
                   "across the full stock dataset.", className="brand-subtitle"),
        ]),
        html.Div(f"{df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}",
                 className="header-badge"),
    ]),

    section("Market overview", "The dataset at a glance."),
    html.Div(className="kpi-grid", children=[
        kpi("Stocks", f"{df['Symbol'].nunique():,}", "Unique symbols"),
        kpi("Trading records", f"{len(df):,}", "Daily observations"),
        kpi("Average close", f"₹{df['Closing_Price'].mean():,.2f}", "Across all stocks"),
        kpi("Median close", f"₹{df['Closing_Price'].median():,.2f}", "Less skewed by outliers"),
        kpi("Average volatility", f"{stats['Volatility_Pct'].mean():.2f}%", "Std. dev. of daily return"),
    ]),

    section("Stock explorer", "Pick a stock to see its price, returns and key numbers."),
    card("Stock", dropdown("stock-selector", symbols, symbols[0], searchable=True)),
    html.Div(id="stock-kpis", className="kpi-grid"),
    html.Div(className="two-column", children=[
        card("Closing price", graph("stock-price-chart"),
             "Close with 7-day and 30-day moving averages."),
        card("Daily return", graph("stock-return-chart"),
             "Day-to-day percentage change."),
    ]),

    section("Performance", "Compare stocks within a single calendar year."),
    card("Year", dropdown("year-selector", [{"label": str(y), "value": y} for y in years], years[-1])),
    html.Div(className="two-column", children=[
        card("Top 10 performers", graph("top-chart")),
        card("Bottom 10 performers", graph("bottom-chart")),
    ]),
    html.Div(className="two-column", children=[
        card("Return distribution", graph("return-boxplot"),
             "Spread of daily returns for the selected year."),
        card("Up vs down days", graph("trading-day-pie"),
             "For the stock chosen in the explorer."),
    ]),

    section("Risk", "Volatility, drawdowns and the risk–return trade-off."),
    html.Div(className="two-column", children=[
        card("Risk vs return", graph(figure=risk_return_figure())),
        card("Highest volatility", graph(figure=volatility_figure())),
    ]),
    card("Maximum drawdown", graph(figure=drawdown_figure()),
         "Largest fall from a previous peak."),

    section("Correlation", "How the year's best performers move together."),
    card("Return correlation", graph("correlation-heatmap"),
         "Top 15 stocks of the selected year, by yearly return."),

    section("Machine learning", "Random Forest classifier for next-day direction, "
                                "tested on the most recent 20% of the data."),
    html.Div(className="kpi-grid", children=[
        kpi("Accuracy", f"{ml['accuracy']:.1%}", f"Majority-class baseline {ml['baseline']:.1%}"),
        kpi("Precision", f"{ml['precision']:.1%}", "Of predicted up days"),
        kpi("Recall", f"{ml['recall']:.1%}", "Of actual up days"),
        kpi("F1 score", f"{ml['f1']:.1%}", "Precision and recall combined"),
    ]),
    html.Div(className="two-column", children=[
        card("Feature importance", graph(figure=importance_figure()),
             "Which inputs the model relies on most."),
        card("Confusion matrix", graph(figure=confusion_figure()),
             "Actual vs predicted direction."),
    ]),
    card("Reading the results", html.Ul(className="notes", children=[
        html.Li("Inputs: 1-day and 5-day lagged returns, 7-day and 30-day momentum, "
                "7-day and 30-day volatility."),
        html.Li("Daily direction is close to a coin flip. Compare accuracy with the "
                "baseline before drawing conclusions."),
        html.Li("This is a learning project, not investment advice."),
    ])),

    html.Footer("Python · Pandas · Plotly · Dash · Scikit-learn", className="footer"),
])


# ---------------------------------------------------------------- callbacks
@app.callback(
    Output("stock-kpis", "children"),
    Output("stock-price-chart", "figure"),
    Output("stock-return-chart", "figure"),
    Output("trading-day-pie", "figure"),
    Input("stock-selector", "value"),
)
def update_stock_explorer(symbol):
    data = df[df["Symbol"] == symbol]
    s = stats_by_symbol.loc[symbol]

    kpis = [
        kpi("Latest close", f"₹{data['Closing_Price'].iloc[-1]:,.2f}", f"{data['Date'].iloc[-1]:%d %b %Y}"),
        kpi("Total return", f"{s['Total_Return']:+.1f}%", "First to last close"),
        kpi("Volatility", f"{s['Volatility_Pct']:.2f}%", "Daily std. dev."),
        kpi("Max drawdown", f"{s['Max_Drawdown']:.1f}%", "Worst fall from a peak"),
    ]

    price = go.Figure()
    for col, name, color, width in (("Closing_Price", "Close", "#e6eaf2", 1.6),
                                    ("MA_7D", "7-day MA", ACCENT, 1.2),
                                    ("MA_30D", "30-day MA", BLUE, 1.2)):
        price.add_trace(go.Scatter(x=data["Date"], y=data[col], name=name, mode="lines",
                                   line=dict(color=color, width=width),
                                   hovertemplate=f"{name}: ₹%{{y:,.2f}}<extra></extra>"))
    style(price, f"{symbol} price trend", unified=True).update_yaxes(title="Price (₹)")

    returns = px.line(data, x="Date", y="Daily_Return")
    returns.update_traces(line=dict(color=BLUE, width=1),
                          hovertemplate="%{y:.2%}<extra></extra>")
    style(returns, f"{symbol} daily returns", unified=True).update_yaxes(tickformat=".1%")

    pie = px.pie(names=["Up days", "Down days"],
                 values=[s["Positive_Days"], s["Negative_Days"]], hole=0.6,
                 color_discrete_sequence=[GAIN, LOSS])
    pie.update_traces(textinfo="label+percent")
    style(pie, f"{symbol} up vs down days").update_layout(showlegend=False)

    return kpis, price, returns, pie


@app.callback(
    Output("top-chart", "figure"),
    Output("bottom-chart", "figure"),
    Output("return-boxplot", "figure"),
    Output("correlation-heatmap", "figure"),
    Input("year-selector", "value"),
)
def update_yearly_views(year):
    y = yearly[yearly["Year"] == year]
    if y.empty:
        blank = empty_figure(f"No data for {year}")
        return blank, blank, blank, blank

    top = y.nlargest(10, "Yearly_Return").sort_values("Yearly_Return")
    bottom = y.nsmallest(10, "Yearly_Return").sort_values("Yearly_Return")
    top_fig = hbar(top, "Yearly_Return", "Symbol", GAIN, f"Top 10 in {year}", "Return (%)")
    bottom_fig = hbar(bottom, "Yearly_Return", "Symbol", LOSS, f"Bottom 10 in {year}", "Return (%)")

    year_data = df[df["Year"] == year].dropna(subset=["Daily_Return"])
    if len(year_data) > 50_000:  # keep the browser responsive
        year_data = year_data.sample(50_000, random_state=42)
    box = px.box(year_data, y="Daily_Return", points=False, labels={"Daily_Return": "Daily return"})
    box.update_traces(marker_color=BLUE)
    style(box, f"Daily return distribution, {year}").update_yaxes(tickformat=".1%")

    symbols_top = y.nlargest(15, "Yearly_Return")["Symbol"]
    corr = (df[(df["Year"] == year) & df["Symbol"].isin(symbols_top)]
            .pivot_table(index="Date", columns="Symbol", values="Daily_Return").corr())
    heat = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                     color_continuous_scale="RdBu_r", labels={"color": "Correlation"})
    style(heat, f"Return correlation of {year}'s top 15")

    return top_fig, bottom_fig, box, heat


if __name__ == "__main__":
    app.run(debug=True, port=8050)