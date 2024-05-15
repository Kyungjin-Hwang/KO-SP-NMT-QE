# KO-SP-NMT-QE
### Towards Improved Korean-Spanish Machine Translation: Dataset Construction and Utilization Strateges

["The link to the paper is scheduled to be uploaded in July 2024."](URL)

This dataset is aimed at evaluating machine translation from a linguistic perspective, focusing on lexicology, morphology, syntax, pragmatics, and information structure.

### Data Usage

Each set of data is categorized by fields of linguistics. In the data, 'segmentid' distinguishes the categories as follows: Lexicology is represented by 'L', Morphology by 'M', Syntax by 'S', Pragmatics by 'P', and Information Structure by 'I'.

Subsequently, each linguistic category is divided into major classifications. The major classifications are as follows, with the indices in parentheses indicating the index of the respective major classification. In the cases of Lexicology and Information Structure, where there is no separate major classification, the index is denoted as '00'.

- **Morphology**: Internal structure of words, word formation
- **Syntax**: Constituents of sentences, types of sentences, word order of sentences
- **Pragmatics**: Pragmatics dataset, spoken and written text

Further, the data is divided into subcategories. The subcategories are detailed in the table below.

### Lexicology

| Subcategory | Code | 
|-------------|------|
| Usage       | CA   | 
| Domain      | DO   | 
| Form        | FM   | 

### Morphology

| Subcategory                        | Code | Subcategory | Code |
|------------------------------------|------|-------------|------|
| Internal Structure                 |      | Word Formation |      |
| Nouns                              | CA   | Derivatives | DE   |
| Pronouns and prepositional particles | PP  | Compounds   | CO   |
| Predicates                         | PR   |             |      |
| Complements                        | CM   |             |      |

### Syntax

| Subcategory | Code | Subcategory | Code | Subcategory | Code |
|-------------|------|-------------|------|-------------|------|
| Constituents|      | Structures  |      | Order       |      |
| Subject     | SU   | Types       | ST   | Movement and Scrambling  | MS   |
| Predicate   | PR   | Clause      | CL   | Islands     | IS   |
| Object      | OB   |             |      |             |      |

### Pragmatics and Information Structure

| Subcategory | Code | Subcategory | Code | Subcategory | Code |
|-------------|------|-------------|------|-------------|------|
| Deixis and anaphors | PS   | Spoken Text    | ST   | Subjects     | SU   |
| Principles of pragmatics | PC   | Written Text   | WT   | Non-subjects | NS   |

If you refer to the paper concerning this dataset, you will find that the evaluation items are divided by subcategory. This dataset aims to assess whether machine translation models or services can accurately translate based on these evaluation items. The 'segmentid' is assigned in the order mentioned in the paper, such as 01, 02, etc.


#### This dataset was constructed as part of a Ph.D. dissertation in the Department of Spanish Language and Literature at Korea University, scheduled for publication in June 2024.
