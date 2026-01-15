import pytest

from gptcot.contracts import parse_month_label


def test_parse_month_label_happy_path():
    assert parse_month_label("Feb-2015") == "2015-02-01"
    assert parse_month_label("Dec-1999") == "1999-12-01"


def test_parse_month_label_invalid():
    with pytest.raises(ValueError):
        parse_month_label("2015-Feb")
    with pytest.raises(ValueError):
        parse_month_label("Foo-2020")
