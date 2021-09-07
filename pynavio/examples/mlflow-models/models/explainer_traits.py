import json

import pandas as pd

from utils.styling import HEATMAP_COLOR_SCALE


class TabularExplainerTraits:
    BG_COLUMN = 'is_background'

    def has_background(self, data: pd.DataFrame) -> bool:
        if self.BG_COLUMN not in data.columns:
            return False

        if not data[self.BG_COLUMN].isin({True, False}).all():
            return False

        # must have at least one of each True and False
        if data[self.BG_COLUMN].nunique() != 2:
            return False

        return True

    def select_data(self, data: pd.DataFrame, background: bool) -> pd.DataFrame:
        return data.loc[data[self.BG_COLUMN] == background] \
                   .drop(self.BG_COLUMN, axis=1)

    def draw_plotly_explanation(self, data: pd.Series) -> dict:
        import plotly.express as px

        # shap or importance value for each variable
        df = data.rename('value').rename_axis('variable').reset_index()

        return json.loads(
            px.bar(
                df, x='variable',
                y='value').update_traces(marker_color='rgb(0,0,0)').to_json())


class TimeSeriesExplainerTraits(TabularExplainerTraits):

    def draw_plotly_explanation(self, data: pd.DataFrame) -> dict:
        import plotly.express as px

        # frame index is time. Plotly handles axes automatically
        fig = px.imshow(data.rename_axis('time').transpose(),
                        color_continuous_scale=HEATMAP_COLOR_SCALE,
                        color_continuous_midpoint=0.)

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')

        return json.loads(fig.to_json())
