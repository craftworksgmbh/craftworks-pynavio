import pandas as pd

from pynavio.traits import TabularExplainerTraits, TimeSeriesExplainerTraits

tabular_data = pd.Series({'var1': 10, 'var2': 20, 'var3': 30})
timeseries_data = pd.DataFrame({
    'time': [1, 2, 3],
    'var1': [10, 20, 30],
    'var2': [20, 30, 40]
})


def test_tabular_draw_plotly_explanation():
    tabular_explainer = TabularExplainerTraits()
    explanation = tabular_explainer.draw_plotly_explanation(tabular_data)

    assert isinstance(explanation, dict)


def test_timeseries_draw_plotly_explanation():
    timeseries_explainer = TimeSeriesExplainerTraits()
    explanation = timeseries_explainer.draw_plotly_explanation(timeseries_data)

    assert isinstance(explanation, dict)
