from ghxtox.multisource_conformer_analysis import _length_bucket


def test_length_buckets_cover_long_peptide_pilot_strata():
    expected = {
        15: "1-15",
        16: "16-20",
        25: "21-25",
        30: "26-30",
        31: "31-35",
        36: "36-40",
        41: "41-45",
        46: "46-50",
        51: "51+",
    }
    assert {length: _length_bucket(length) for length in expected} == expected
