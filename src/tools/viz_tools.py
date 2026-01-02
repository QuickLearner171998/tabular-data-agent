"""Visualization tools for creating professional, well-styled charts."""

from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

from src.data_loader import DataLoader

# Professional color palette
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2", 
    "accent": "#f093fb",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "danger": "#f87171",
}

COLOR_SEQUENCE = [
    "#667eea", "#764ba2", "#f093fb", "#4ade80", "#fbbf24", 
    "#f87171", "#38bdf8", "#a78bfa", "#fb7185", "#34d399"
]

# Chart theme settings
CHART_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, system-ui, sans-serif", "color": "#e2e8f0"},
    "title_font": {"size": 16, "color": "#f8fafc"},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
}


def apply_chart_style(fig: go.Figure, title: str = None) -> go.Figure:
    """Apply consistent professional styling to a chart."""
    fig.update_layout(
        template=CHART_THEME["template"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        title=dict(
            text=title,
            font=CHART_THEME["title_font"],
            x=0.5,
            xanchor="center",
        ) if title else None,
        margin=CHART_THEME["margin"],
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    
    # Style axes
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
        tickfont={"size": 11},
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.1)",
        linecolor="rgba(255,255,255,0.2)",
        tickfont={"size": 11},
    )
    
    return fig


def format_number(val: float) -> str:
    """Format large numbers for display."""
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


class VizTools:
    """Tools for creating professional visualizations."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
    
    def create_bar_chart(
        self,
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str | None = None,
        color_column: str | None = None,
        orientation: str = "v",
        sort_values: bool = True,
    ) -> dict[str, Any]:
        """Create a professional bar chart."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            # Sort by value for better visualization
            if sort_values and y_column in df.columns:
                df = df.sort_values(y_column, ascending=False)
            
            # Use horizontal for many categories or long labels
            if len(df) > 8 or df[x_column].astype(str).str.len().max() > 10:
                orientation = "h"
                x_col, y_col = y_column, x_column
            else:
                x_col, y_col = x_column, y_column
            
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=color_column,
                orientation=orientation,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            # Add value labels on bars
            fig.update_traces(
                texttemplate='%{value:,.0f}',
                textposition='outside' if orientation == 'v' else 'auto',
                textfont={"size": 10, "color": "#e2e8f0"},
            )
            
            chart_title = title or f"{y_column.replace('_', ' ').title()} by {x_column.replace('_', ' ').title()}"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "bar",
                "figure": fig.to_json(),
                "summary": f"Bar chart: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_line_chart(
        self,
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str | None = None,
        color_column: str | None = None,
    ) -> dict[str, Any]:
        """Create a professional line chart for trends."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            # Sort by x for proper line ordering
            df = df.sort_values(x_column)
            
            fig = px.line(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                markers=True,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            # Style the line
            fig.update_traces(
                line={"width": 3},
                marker={"size": 8},
            )
            
            # Add area fill for single line
            if not color_column:
                fig.update_traces(fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)')
            
            chart_title = title or f"{y_column.replace('_', ' ').title()} Over Time"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "line",
                "figure": fig.to_json(),
                "summary": f"Line chart: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_scatter_plot(
        self,
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str | None = None,
        color_column: str | None = None,
        size_column: str | None = None,
    ) -> dict[str, Any]:
        """Create a scatter plot with trendline."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                size=size_column,
                trendline="ols" if not color_column else None,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            fig.update_traces(marker={"size": 10, "opacity": 0.7})
            
            chart_title = title or f"{y_column.replace('_', ' ').title()} vs {x_column.replace('_', ' ').title()}"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "scatter",
                "figure": fig.to_json(),
                "summary": f"Scatter plot: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_histogram(
        self,
        dataset_name: str,
        column: str,
        title: str | None = None,
        nbins: int | None = None,
        color_column: str | None = None,
    ) -> dict[str, Any]:
        """Create a histogram."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            fig = px.histogram(
                df,
                x=column,
                nbins=nbins or 30,
                color=color_column,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            fig.update_traces(opacity=0.8)
            
            chart_title = title or f"Distribution of {column.replace('_', ' ').title()}"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "histogram",
                "figure": fig.to_json(),
                "summary": f"Histogram: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_pie_chart(
        self,
        dataset_name: str,
        names_column: str,
        values_column: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Create a donut chart (better than pie)."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            # Limit to top categories for readability
            if len(df) > 6:
                df = df.nlargest(5, values_column)
                others = df[values_column].sum() * 0.1  # Approximate "others"
            
            fig = px.pie(
                df,
                names=names_column,
                values=values_column,
                hole=0.4,  # Donut style
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            fig.update_traces(
                textposition='outside',
                textinfo='percent+label',
                textfont={"size": 11, "color": "#e2e8f0"},
            )
            
            chart_title = title or f"{values_column.replace('_', ' ').title()} by {names_column.replace('_', ' ').title()}"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "pie",
                "figure": fig.to_json(),
                "summary": f"Donut chart: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_heatmap(
        self,
        dataset_name: str,
        columns: list[str] | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Create a correlation heatmap."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            if columns:
                df = df[columns]
            
            numeric_df = df.select_dtypes(include=["number"])
            
            if numeric_df.empty:
                return {"success": False, "error": "No numeric columns for heatmap"}
            
            corr_matrix = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
            
            chart_title = title or "Correlation Heatmap"
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "heatmap",
                "figure": fig.to_json(),
                "summary": f"Heatmap: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_box_plot(
        self,
        dataset_name: str,
        y_column: str,
        x_column: str | None = None,
        title: str | None = None,
        color_column: str | None = None,
    ) -> dict[str, Any]:
        """Create a box plot."""
        try:
            df = self.data_loader.get_dataframe(dataset_name)
            
            fig = px.box(
                df,
                x=x_column,
                y=y_column,
                color=color_column or x_column,
                color_discrete_sequence=COLOR_SEQUENCE,
            )
            
            chart_title = title or f"Distribution of {y_column.replace('_', ' ').title()}"
            if x_column:
                chart_title += f" by {x_column.replace('_', ' ').title()}"
            
            apply_chart_style(fig, chart_title)
            
            return {
                "success": True,
                "chart_type": "box",
                "figure": fig.to_json(),
                "summary": f"Box plot: {chart_title}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def create_viz_tools(data_loader: DataLoader) -> list:
    """Create LangChain tools from VizTools."""
    viz_tools = VizTools(data_loader)
    
    @tool
    def create_bar_chart(
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str = None,
        color_column: str = None,
    ) -> dict:
        """
        Create a bar chart. Best for comparing categories or showing rankings.
        Automatically uses horizontal bars for many categories.
        
        Args:
            dataset_name: Name of the dataset
            x_column: Column for categories (x-axis)
            y_column: Column for values (y-axis)
            title: Optional chart title
            color_column: Optional column for color grouping
        """
        return viz_tools.create_bar_chart(dataset_name, x_column, y_column, title, color_column)
    
    @tool
    def create_line_chart(
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str = None,
        color_column: str = None,
    ) -> dict:
        """
        Create a line chart. Best for showing trends over time.
        Use ONLY when x-axis is a date/time column.
        
        Args:
            dataset_name: Name of the dataset
            x_column: Column for x-axis (should be date/time)
            y_column: Column for y-axis values
            title: Optional chart title
            color_column: Optional column for multiple lines
        """
        return viz_tools.create_line_chart(dataset_name, x_column, y_column, title, color_column)
    
    @tool
    def create_scatter_plot(
        dataset_name: str,
        x_column: str,
        y_column: str,
        title: str = None,
        color_column: str = None,
    ) -> dict:
        """
        Create a scatter plot with trendline. Best for showing correlation between two numeric variables.
        
        Args:
            dataset_name: Name of the dataset
            x_column: Column for x-axis (numeric)
            y_column: Column for y-axis (numeric)
            title: Optional chart title
            color_column: Optional column for color grouping
        """
        return viz_tools.create_scatter_plot(dataset_name, x_column, y_column, title, color_column)
    
    @tool
    def create_histogram(
        dataset_name: str,
        column: str,
        title: str = None,
        nbins: int = None,
    ) -> dict:
        """
        Create a histogram. Best for showing distribution of a single numeric variable.
        
        Args:
            dataset_name: Name of the dataset
            column: Numeric column to show distribution of
            title: Optional chart title
            nbins: Optional number of bins
        """
        return viz_tools.create_histogram(dataset_name, column, title, nbins)
    
    @tool
    def create_pie_chart(
        dataset_name: str,
        names_column: str,
        values_column: str,
        title: str = None,
    ) -> dict:
        """
        Create a donut chart. Best for showing parts of a whole (proportions).
        Use ONLY when you have 5-6 or fewer categories.
        
        Args:
            dataset_name: Name of the dataset
            names_column: Column for slice labels
            values_column: Column for slice values
            title: Optional chart title
        """
        return viz_tools.create_pie_chart(dataset_name, names_column, values_column, title)
    
    @tool
    def create_correlation_heatmap(
        dataset_name: str,
        title: str = None,
    ) -> dict:
        """
        Create a heatmap showing correlations between numeric columns.
        Best for identifying which variables are related.
        
        Args:
            dataset_name: Name of the dataset
            title: Optional chart title
        """
        return viz_tools.create_heatmap(dataset_name, title=title)
    
    return [
        create_bar_chart,
        create_line_chart,
        create_scatter_plot,
        create_histogram,
        create_pie_chart,
        create_correlation_heatmap,
    ]
