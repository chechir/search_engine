# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from search_engine import search as s


df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Asos red bag', 'reddish small bag', 'blue jeans'],
        })
search_text = 'asos red bag'


def test_SearchEngine():
    model = s.SearchEngine()
    model.fit(df)
    results_df = model.get_results('Asos RED bag')
    assert len(results_df) == 2
    assert np.all(results_df['id'].values == np.array([1, 2]))


def test_SearchEngine_preprocess():
    model = s.SearchEngine()
    model.fit(df, ngram_range=(1,1), perform_stem=False)
    result = model.preprocess(df)
    assert result[0] == 'asos red bag'
    

def test_SearchEngine_fit():
    model = s.SearchEngine()
    model.fit(df, ngram_range=(1,1))
    assert model.fitted_tfidf.shape[0] == 3
    assert model.fitted_tfidf.shape[1] == 7
    

def test_SearchEngine_get_score():
    model = s.SearchEngine()
    model.fit(df, ngram_range=(1,1))
    result = model.get_score('aso red jean')
    expected = [0.71910824, 0., 0.40824829]
    assert np.allclose(result, expected) 


def test_SearchEngine_steam_document():
    model = s.SearchEngine()
    doc = 'this is a file containing funny words'
    result = model.stem_document(doc)
    expected = 'this is a file contain funni word'
    assert result == expected