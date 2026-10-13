"""Question source: the BigBio mirror of MedQA, which keeps the original US 4-option
train / dev / test split (10,178 / 1,272 / 1,273), Step labels and MetaMap phrases."""
QUESTIONS = {
    "name": "bigbio/med_qa",
    "config": "med_qa_en_4options_source",
    "splits": {"dev": "validation", "test": "test"},
}
