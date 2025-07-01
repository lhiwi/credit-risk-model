import src.data_processing

def test_data_processing_imports():
    """
    Simple sanity check: ensure the data_processing module can be imported.
    """
    assert hasattr(src.data_processing, "__file__")
