# Checks the version of the twfeiw

def test_import_twfeiw():
    import twfeiw

    assert twfeiw.__version__ == "0.1.0"