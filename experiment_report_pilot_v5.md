# Case Grammar Pilot Generation (Final Fair Comparison v5)

This report provides a truly fair comparison by cleaning the Mechanical version strings to match the length of Filled and LLM versions, and providing ALL to the evaluator with the same separate Context field.

## 1. Quantitative Analysis by Dataset

| Dataset | Metric | Filled (Baseline) | Mechanical Ablation | **LLM (Natural)** |
| :--- | :--- | :---: | :---: | :---: |
| multiwoz | Average PPL | 245.96 | 32.50 | 126.97 |
| | Naturalness | 4.69 | 3.14 | **4.33** |
| | Omission | 3.62 | 4.93 | **4.67** |
| | MinimalChange | 4.69 | 4.21 | **4.33** |
| qasrl | Average PPL | 202.41 | 216.72 | 294.43 |
| | Naturalness | 4.75 | 3.53 | **4.40** |
| | Omission | 3.11 | 4.53 | **4.40** |
| | MinimalChange | 4.93 | 4.33 | **4.80** |
| sgd | Average PPL | 3423.05 | 42.60 | 730.92 |
| | Naturalness | 4.30 | 3.16 | **4.67** |
| | Omission | 3.60 | 5.00 | **5.00** |
| | MinimalChange | 4.60 | 4.26 | **5.00** |

## 2. Qualitative Analysis (Samples)

### MULTIWOZ Samples

#### Sample 1 (Role: goal)
- **Context**: `[Domain: restaurant, train]Assistant: Yes, there is. What's your price range?`
- **Filled**: `Yes I'm looking for a train to Cambridge that same day.` (N/O/M: )
- **Mech Missing**: `Yes I'm looking for a train to that same day.` (N/O/M: Naturalness: 2, Omission: 5, MinimalChange: 4)
- **LLM Missing**: `Yes I'm looking for a train that same day.` (N/O/M: )
- **PPL (F/M/L)**: 137.2 / 51.7 / 73.3

#### Sample 2 (Role: time)
- **Context**: `[Domain: train]`
- **Filled**: `Im going to cambridge on thursday` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **Mech Missing**: `Im going to cambridge on` (N/O/M: Naturalness: 3, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `Im going to cambridge on` (N/O/M: Naturalness: 4, Omission: 4, MinimalChange: 5)
- **PPL (F/M/L)**: 1273.3 / 19.3 / 475.0

#### Sample 3 (Role: theme)
- **Context**: `[Domain: restaurant]`
- **Filled**: `Italian sounds good.  Can you give me an address and phone number?` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **Mech Missing**: `sounds good. Can you give me an address and phone number?` (N/O/M: Naturalness: 3, Omission: 5, MinimalChange: 4)
- **LLM Missing**: `Can you give me an address and phone number?` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **PPL (F/M/L)**: 54.4 / 27.1 / 36.6

#### Sample 4 (Role: manner)
- **Context**: `[Domain: hotel, train]
Assistant: There are three choices in that price range: Bridge Guest House in the south, Hamilton Lodge in the north, and Hobson's House in the west.`
- **Filled**: `Yeah, could you book me a 3 night stay at Hobson's House?` (N/O/M: )
- **Mech Missing**: `Yeah, could you book me a night stay at Hobson's House?` (N/O/M: Naturalness: 3, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `Yeah, could you book me a stay at Hobson's House?` (N/O/M: )
- **PPL (F/M/L)**: 158.7 / 19.0 / 167.3

#### Sample 5 (Role: source)
- **Context**: `[Domain: attraction, train]
Assistant: Sure, there are plenty of those in the centre.  I'd recommend Kambar, located at 1 Wheeler Street.  Can I get you more information?`
- **Filled**: `Great, thanks. Yes, actually. Can you find a train schedule for me? I'll be traveling Norwich to Cambridge on Thursday.` (N/O/M: )
- **Mech Missing**: `Great, thanks. Yes, actually. Can you find a train schedule for me? I'll be traveling to Cambridge on Thursday.` (N/O/M: Naturalness: 4, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `Great, thanks. Yes, actually. Can you find a train schedule for me? I'll be traveling  to Cambridge on Thursday.` (N/O/M: )
- **PPL (F/M/L)**: 54.5 / 15.7 / 38.0

---

### QASRL Samples

#### Sample 1 (Role: theme)
- **Context**: `None`
- **Filled**: `[Target Verb: make] Thats because the bowling ball is made of solid plastic , which contains a lot of tightly packed particles of matter .` (N/O/M: Naturalness: 5, Omission: 1, MinimalChange: 5)
- **Mech Missing**: `Thats because the bowling ball is made of, which contains a lot of tightly packed particles of matter.` (N/O/M: Naturalness: 2, Omission: 4, MinimalChange: 4)
- **LLM Missing**: `Thats because the bowling ball is made of  which contains a lot of tightly packed particles of matter .` (N/O/M: Naturalness: 3, Omission: 4, MinimalChange: 4)
- **PPL (F/M/L)**: 115.4 / 104.3 / 83.7

#### Sample 2 (Role: agent)
- **Context**: `None`
- **Filled**: `[Target Verb: state] Glaeser furthers his argument by stating that bigger cities do not pay more for equal productivity than in a smaller city , so it is reasonable to assume that workers become more productive if they move to a city twice the size as they initially worked in .` (N/O/M: Naturalness: 4, Omission: 3, MinimalChange: 4)
- **Mech Missing**: `furthers his argument by stating that bigger cities do not pay more for equal productivity than in a smaller city, so it is reasonable to assume that workers become more productive if they move to a city twice the size as they initially worked.` (N/O/M: Naturalness: 4, Omission: 5, MinimalChange: 4)
- **LLM Missing**: `furthers his argument by stating that bigger cities do not pay more for equal productivity than in a smaller city , so it is reasonable to assume that workers become more productive if they move to a city twice the size as they initially worked in .` (N/O/M: Naturalness: 5, Omission: 4, MinimalChange: 5)
- **PPL (F/M/L)**: 74.3 / 56.6 / 56.7

#### Sample 3 (Role: time)
- **Context**: `None`
- **Filled**: `[Target Verb: watch] SledgeD will be just as fun as the first , with another group of people sitting down to watch Assly and another funny trailer for a made-up movie .` (N/O/M: Naturalness: 4, Omission: 3, MinimalChange: 5)
- **Mech Missing**: `SledgeD will be just as fun as the first, with another group of people to watch Assly and another funny trailer for a made-up movie.` (N/O/M: Naturalness: 4, Omission: 4, MinimalChange: 5)
- **LLM Missing**: `SledgeD will be just as fun as the first , with another group of people watch Assly and another funny trailer for a made-up movie .` (N/O/M: Naturalness: 4, Omission: 4, MinimalChange: 5)
- **PPL (F/M/L)**: 231.7 / 210.0 / 300.9

#### Sample 4 (Role: time)
- **Context**: `None`
- **Filled**: `[Target Verb: loose] Snow : Yes , it will loose strength steadily as more and more of this swirling system moves from being over ocean to being over land .` (N/O/M: Naturalness: 2, Omission: 4, MinimalChange: 4)
- **Mech Missing**: `Snow: Yes, it will loose strength steadily` (N/O/M: Naturalness: 3, Omission: 4, MinimalChange: 4)
- **LLM Missing**: `Snow : Yes , it will loose strength steadily .` (N/O/M: Naturalness: 4, Omission: 4, MinimalChange: 5)
- **PPL (F/M/L)**: 282.6 / 1275.0 / 2336.3

#### Sample 5 (Role: time)
- **Context**: `None`
- **Filled**: `[Target Verb: kill] In 2009 , Major Nidal Hassan shot and killed 13 people and wounded 30 others at the base 's Readiness Processing Center .` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **Mech Missing**: `, Major Nidal Hassan shot and killed 13 people and wounded 30 others at the base 's Readiness Processing Center.` (N/O/M: Naturalness: 4, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `Major Nidal Hassan shot and killed 13 people and wounded 30 others at the base 's Readiness Processing Center .` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **PPL (F/M/L)**: 31.2 / 39.6 / 19.0

---

### SGD Samples

#### Sample 1 (Role: theme)
- **Context**: `[Domain: Services_2 / Intent: BookAppointment]
Assistant: What time works for you?`
- **Filled**: `No. Is it available to book it on 13th of March ?` (N/O/M: Naturalness: 3, Omission: 4, MinimalChange: 4)
- **Mech Missing**: `No. Is it available to book it on 13th of March?` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `No. Is it available to book it on 13th of March ?` (N/O/M: Naturalness: 5, Omission: 5, MinimalChange: 5)
- **PPL (F/M/L)**: 145.5 / 38.0 / 68.9

#### Sample 2 (Role: manner)
- **Context**: `[Domain: Events_2 / Intent: BuyEventTickets]
Assistant: Aloft Philadelphia Airport is 3 stars.`
- **Filled**: `Just 3.` (N/O/M: )
- **Mech Missing**: `Just.` (N/O/M: )
- **LLM Missing**: `Just stars.` (N/O/M: )
- **PPL (F/M/L)**: 510.6 / 118.7 / 3006.0

#### Sample 3 (Role: theme)
- **Context**: `[Domain: Movies_1 / Intent: GetTimesForMovie]
Assistant: sure thing, I found 4 options for you. Breakthrough, Dumbo or Missing Link. Any of these catch your attention?`
- **Filled**: `Pet Sematary is just what i'm in the mood for.Pull up show times for regular showings.` (N/O/M: ``` Naturalness: 4, Omission: 1, MinimalChange: 5  ```)
- **Mech Missing**: `is just what i'm in the mood.Pull up show times for regular showings.` (N/O/M: ``` Naturalness: 2, Omission: 5, MinimalChange: 4  ```)
- **LLM Missing**: `is just what i'm in the mood for.Pull up show times for regular showings.` (N/O/M: ``` Naturalness: 3, Omission: 5, MinimalChange: 5 ```)
- **PPL (F/M/L)**: 131.0 / 53.9 / 191.2

#### Sample 4 (Role: theme)
- **Context**: `[Domain: Events_2 / Intent: BuyEventTickets]`
- **Filled**: `Buy for me tickets for events in San Francisco on Monday next week.Giants Vs Diamondbacks I heard that is very good.` (N/O/M: Naturalness: 4, Omission: 3, MinimalChange: 4)
- **Mech Missing**: `Buy for me tickets for events in San Francisco on Monday next week. I heard that is very good.` (N/O/M: Naturalness: 4, Omission: 5, MinimalChange: 5)
- **LLM Missing**: `Buy for me tickets for events in San Francisco on Monday next week. I heard that is very good.` (N/O/M: )
- **PPL (F/M/L)**: 213.4 / 104.3 / 105.3

#### Sample 5 (Role: agent)
- **Context**: `[Domain: Hotels_2 / Intent: BookHouse]
Assistant: When would you need to check out?`
- **Filled**: `Yes the reservation is just for one and I need to check in March 1st.` (N/O/M: )
- **Mech Missing**: `Yes the reservation is just for one and I need to check in March st.` (N/O/M: )
- **LLM Missing**: `Yes the reservation is just for one and I need to check in March .` (N/O/M: )
- **PPL (F/M/L)**: 95.7 / 27.9 / 129.8

---

