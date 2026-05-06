# Failure Case Analysis

Analysis of the 264 misclassified examples from the 2,600-sample held-out test set
(118 false positives, 146 false negatives).

## Overview

|             | False Positives (FP)                      | False Negatives (FN)                           |
|-------------|-------------------------------------------|------------------------------------------------|
| Count       | 118                                       | 146                                            |
| Description | Predicted *claim*, actually *not a claim* | Predicted *not a claim*, actually *is a claim* |
| Cost        | Wasted fact-checking effort               | Missed claim — slips through undetected        |

The model's precision (0.900) and recall (0.880) suggest it errs
slightly more toward false negatives than false positives — it misses ~12% of real claims
while incorrectly flagging ~10% of non-claims.

## False Positive Patterns

Sentences the model incorrectly labeled as claims (precision errors).

| Pattern                   | % of FPs   | % of all non-claims |
|---------------------------|------------|---------------------|
| Contains numbers          | 15.3%      | 2.9%                |
| Contains named entities   | 44.9%      | 39.0%               |
| Is a question             | 2.5%       | 4.0%                |
| Short sentence (<8 words) | 10.2%      | 19.5%               |
| Avg word count            | 21.5 words | 17.0 words          |

**Most confidently wrong false positives:**

- *"In Puerto Rico this year, I met with six of the leading industrial nations' heads of state to meet the problem of inflation so we would be able to solve it before it got out of hand."* (confidence: 99.9%)
- *"Th is last year h as been a very ch allen ging balancing act ."* (confidence: 99.8%)
- *"Medicaid's in one agency; Medicare is in a different one."* (confidence: 99.8%)
- *"Six months after he said Osama bin Laden must be caught dead or alive, this president was asked, "Where is Osama bin Laden? ""* (confidence: 99.8%)
- *"The Republican party has produced McKinley and Harding, Coolidge, Dewey, and Landon."* (confidence: 99.8%)
- *"John Edwards is the author of the Patients' Bill of Rights."* (confidence: 99.8%)
- *"Governor Carter uh - brags about the unemployment during Democratic administrations and condemns the unemployment at the present time."* (confidence: 99.8%)
- *"He said that he'd cut in half the deficit."* (confidence: 99.8%)
- *"The new economic revitalization program that we have in mind, which will be implemented next year, would result in tax credits which would let business invest in new tools and new factories to create even more new jobs - about one million in the next two years."* (confidence: 99.8%)
- *"This is just 1 of 1000."* (confidence: 99.7%)

**Interpretation:** The strongest signal is numbers: false positives contain numbers at
5× the rate of typical non-claims (15.3% vs 2.9%). They are also longer on average
(21.5 vs 17.0 words). Looking at the examples, many false positives are genuinely
ambiguous — sentences like *"Medicaid's in one agency; Medicare is in a different one"*
or *"The Republican party has produced McKinley and Harding..."* read as verifiable claims.
This suggests some failures reflect real label ambiguity in the dataset rather than
model error, and that numeric content is the dominant false-positive trigger.

## False Negative Patterns

Sentences the model incorrectly labeled as non-claims (recall errors).

| Pattern                   | % of FNs   | % of all claims |
|---------------------------|------------|-----------------|
| Contains numbers          | 6.2%       | 42.5%           |
| Contains named entities   | 50.7%      | 62.4%           |
| Is a question             | 2.7%       | 0.7%            |
| Short sentence (<8 words) | 8.9%       | 6.1%            |
| Avg word count            | 21.3 words | 20.3 words      |

**Most confidently wrong false negatives:**

- *"The first position you took, when this matter first came up, was that we should draw the line and commit ourselves, as a matter of principle, to defend these islands."* (confidence: 99.1%)
- *"We know that the best path to a good job is a good education."* (confidence: 99.1%)
- *"That's what's meant by a government of, by, and for the people. What I'm suggesting is hard."* (confidence: 99.1%)
- *"And it works- -it works in Milwaukee."* (confidence: 99.1%)
- *"And China's a got a lot of influence over North Korea, some ways more than we do."* (confidence: 99.1%)
- *"I was against the war."* (confidence: 99.0%)
- *"Now, as someone who begins every day with an intelligence briefing, I know this is a dangerous time."* (confidence: 99.0%)
- *"He didn't take that position on Tibet."* (confidence: 99.0%)
- *"And if it gets overturned, then we'll end up with marriage being defined by courts, and I don't think that's in our nation's interests."* (confidence: 99.0%)
- *"And I have said freely, all over this country, that it was a mistake for me or anyone to ever try to put the Judeo-Christian heritage of this country, important as it is, and important as my religious faith is to me - it's a very deeply personal matter."* (confidence: 99.0%)

**Interpretation:** The dominant pattern is the absence of numbers: false negatives contain
numbers at less than one-sixth the rate of typical claims (6.2% vs 42.5%), and fewer named
entities too (50.7% vs 62.4%). The model has learned to associate numeric specificity with
claim-worthiness, so general or abstract claims — *"We know that the best path to a good job
is a good education"*, *"I was against the war"* — lack the surface signals it expects and
slip through. These are also the claims most worth catching, since they are often the kind
of sweeping political assertions that most need fact-checking.

## Implications

**For downstream fact-checking:** false negatives are the more costly error — a missed
claim never gets fact-checked. Given the model's slightly higher false negative rate,
a production system might lower the classification threshold (currently 0.5) to trade
some precision for higher recall, depending on the cost tolerance of the downstream workflow.

**For improving the model:**
1. *More training data* — the demo run used only 2,000 of 10,397 available samples. The
   full training run is expected to close most of the gap to the paper's benchmark (F1 0.911).
2. *Hard negative mining* — deliberately including factual-sounding non-claims (e.g. vivid
   descriptions with named entities) in training could sharpen the precision boundary.
3. *Out-of-domain robustness* — as Bell (2025) notes, BERT-based models generalize poorly
   to out-of-domain text. For inputs outside political speeches and fact-checks, the false
   positive rate rises sharply. An LLM-based fallback for low-confidence predictions could
   mitigate this.
