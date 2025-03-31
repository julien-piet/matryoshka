# Evaluation code

This code is used to compare Matryoshka parsers to various baselines. This code is mostly undocumented.

### Basic syntax evaluation

* First, save the results from parsing in a json or dill formatted structure. The expected format is the following:
```json
[
entry0,
entry1,
...
]
```
where entry #N represents the parsing of line #N (starting at N=0), with the following fields:
```json
entryk = 
(
    k,
    full_line_k,
    suffix_line_k,
    list_of_matched_template_ids,
    list_of_matched_templates,
    list_of_matches_per_template
)
```

In many cases, suffix_line_k = full_line_k, unless the file was pre-processed and only a set of suffixes had to be parsed (as in the loghub dataset)

For example, the entry associated with line "[Thu Jun 09 06:07:19 2005] [notice] mod_security/1.9dev2 configured" with suffix "mod_security/1.9dev2 configured" (at index 16 in its original file) is:

```
(
    16,
    "[Thu Jun 09 06:07:19 2005] [notice] mod_security/1.9dev2 configured",
    "mod_security/1.9dev2 configured",
    [14],
    ['<*>/<*> configured', 'mod_security/<*> configured'], 
    [['mod_security', '1.9dev2'], ['1.9dev2']]
)
```

Here note that two templates matched the line: this is a fictitious example so you can see the expected format. 

* Next, run `logparser-compare` with the following args:
```
logparser-eval-loghub --predictions SPACE_SEPARATED_LIST_OF_RESULTS --baseline BASELINE_RESULTS
```

For example, running the following should get 100% scores:

```
logparser-eval-loghub --predictions golden_parsers/parsed/cron.json --baseline golden_parsers/parsed/cron.json
```