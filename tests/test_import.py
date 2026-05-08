# To run the test do this in your (.venv) environment:
# python -m pytest

# This test_import is usually used for import level standard checks.
# Afterwards I have to add other files to check other staff

# the most basic check (smoke test). Can python import the package?
# Does it have the correct version?
def test_import_twfeiw():
    import twfeiw

    assert twfeiw.__version__ == "0.1.0"