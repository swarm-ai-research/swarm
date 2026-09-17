---
description: "Two AI-assisted cipher breaks re-checked here from the archives' own ciphertext — the 1941 Enigma message MVUEH and an unsolved 1918 ADFGVX message: results whose acceptance does not depend on the solver, and gates that are mechanical in form but only informative through language."
author: "SWARM Team"
date: "2026-09-17"
source: "https://mvueh-enigma-solved.carterl.chatgpt.site/"
keywords:
  - verification
  - gate cost
  - search scope
  - agent swarms
---

# The MVUEH Enigma break, read as a gate-design case

**Sources:** Carter Leffen, *MVUEH: the recovered message and the evidence
for its key* ([exhibit](https://mvueh-enigma-solved.carterl.chatgpt.site/),
completion report dated 16 September 2026). Frode Weierud's
[July 1941 German Army message list](https://cryptocellar.org/bgac/g-army-july-1941.html)
on CryptoCellar, which now marks the message "Broken". Both read 2026-09-17.

## The claim

MVUEH is message Nr. 172 of 10 July 1941. It has 82 body letters and the
indicator GTA/KCI. It went unbroken in the Sullivan–Weierud German Army
corpus. Weierud's note gives the reasons: it is short, it uses a key
different from that day's daily key for the SS-Totenkopf supply service, and
the left wheel turns over mid-message.

Leffen reports breaking it on 14–15 September 2026 with GPT-6 Astra and
parallel specialist agents. The key:

| Setting | Value |
|---|---|
| Rotors, left to right | II – V – III, reflector B |
| Rings | H M F |
| Plugboard | AC BE DG FH KN MO PR SU TV XZ |
| Indicator | GTA / KCI → body start RWD |

The machine output is `BTTE UM ANGABE DES MARSQWEGES X BEFINDE MIQ IN X
ROSENOW ROSENOW X SOFORT FUNKANTWORT X WASCHBBSCH`. In English: "Please
give the route of march. I am in Rosenow, Rosenow. Reply by radio at once."

Weierud's note calls it, as far as they know, "the first time a rather hard
Enigma problem has been broken with AI."

## Our check

We checked it on 2026-09-17 with a 30-line Enigma I simulator written for
this purpose. It uses the standard wiring for rotors I–V and reflector B,
plus double-stepping. None of the solver's code was used. The input was the
ciphertext as published on CryptoCellar, not the solver's preferred
reading.

```text
KCI at GTA -> RWD
IDEVSARMCCNQTATYEVFCDBZGGSMXWLPSYWZYTCBSWURRTBZCVGODVJUSLSOOMJQJZSXSEBZPEYMDNXJFTC
-> BRCEUZANGAKEDESMARSVWEGESXFEFINDEMIQINXROSTNOWROSENOWXSOFORTFUNKANTWORTXWASCHBBPCH
```

The output matches, letter for letter, the "published received" decryption
in the solver's transcription notes. The header gives the claimed start
position. On the archive's own text, the plaintext has 8 garbled letters out
of 82, and `MARS…WEGES`, `BEFINDE MI…`, `ROSENOW` and `SOFORTFUNKANTWORT`
are intact.

## Why this one checks cleanly

### 1. The acceptance input is held by someone other than the solver

The ciphertext was published in Weierud's archive before the break. The
key is five settings anyone can type into any simulator. So the check shares nothing
with the swarm: no code, no transcription, no scoring model. Compare the
[erdos 1038 phantom gate](erdos-1038-swarm-lessons.md), where the only
public gate was a job in the solver's own repo, and it was never run. Here
the gate is anyone's run on the archive's text.

### 2. A mechanical check that passes for every key tells you nothing

The completion report says it directly: "every Enigma key produces
reversible output." Exact re-encryption proves the arithmetic, not the key.
That check is G0 in form (pass/fail, no judgment) and carries zero bits,
because every candidate passes it. It is the
[false green](cantrip-runtime-lessons.md) in its purest form.

The checks that do discriminate are:

- **Unforced language.** The crib fixed 14 letters. The other 68 came out
  as connected German, and nothing in the search required that.
- **The indicator.** GTA/KCI is recorded separately and must decrypt to the
  body start. 923 of 14,829,646 physical keys pass it, so it is strong but
  not unique on its own.
- **The unchanged archive text.** See lesson 3.

The language check is technically a judgment gate. It is cheap because a
wrong key does not produce `SOFORTFUNKANTWORT`, and no reviewer needs to be
calibrated to see that. A gate's class should be set by what it
discriminates, not by whether it is automated.

### 3. Fitted inputs, and the result reported on the unfitted one

The outgoing facsimile is faint. Before searching, the solver recorded 12
uncertain positions with 2–3 allowed letters each, for 13,824 readings in
total. After the crib, a trigram score picked the best of the 576 readings
that remained. So the preferred ciphertext is an output of the method, and
its cleaner plaintext is partly fitted.

Two things defuse this. The alternatives were fixed before discovery, which
is the preregistration move. And the report leads with the decryption of the
unchanged published transcription, which has no letters chosen at all. That
is the result we re-ran. If a claim has a fitted input, report it on the
unfitted input first. A cleaner fitted version is a presentation choice, not
evidence.

### 4. The scope is declared, and so is what it excludes

"The search is complete within its stated assumptions": 43,016 batches, 60
rotor orders, 107 stepping classes, 38 viable crib placements, and 0
unresolved scopes. The report then lists what the search excludes: other
reflectors, other cable counts, other indicator procedures, readings outside
the recorded sets, and plaintext without the crib. It also says the download
audit re-ran the winning batch and the listed experiments, not the whole
campaign. That is the no-silent-caps rule from the
[erdos ecosystem lessons](erdos-1038-swarm-lessons.md):
what was ruled out and what was not tried are both deliverables.

### 5. Post-discovery checks are labelled post-discovery

Two robustness checks have a circularity risk, and the report names it both
times.

- **Rerun without source help.** A search on the unchanged text with the
  crib `SOFORTFUNKANTWORT`, given no source alternatives and no indicator,
  returns 4,056 physical keys. Exactly one passes GTA/KCI afterwards, and it
  is the recovered key. The crib was learned from the result, so the report
  calls this "a post-discovery robustness check."
- **Leak removal.** SIPVX, which supplied the original crib, was in the
  text the n-gram language tables were trained on. Removing it still leaves the recovered
  reading ranked first.

Neither check is evidence of discovery. Both are honest about being checks
of stability.

### 6. Acceptance was institutional

The exhibit says "solved." What made that public was the custodian of the
corpus editing the message list. That is the pattern from ecosystem lesson 5:
a kernel, or a simulator, checks the object, and a curated ledger confers the
status. The swarm's own report was careful not to claim more than the
calculation.

## Where this sits among the rig's swarm cases

| Case | Gate | Held by solver? | What the gate discriminates |
|---|---|---|---|
| [erdos 1038](erdos-1038-swarm-lessons.md) | Lean kernel job, never dispatched | yes | nothing, since it did not run |
| fm-agent-harness false green | example tests | yes | nothing, since broken code passed |
| MVUEH | re-decrypting the archive ciphertext | no | the key, through unforced German |
| ADFGVX 27 Nov 1918 | re-decrypting the scanned ciphertext | no | the key, through unforced German; one square cell by a ship's log |

MVUEH supports erdos ecosystem lesson 7: aggressive AI search pays where
checking is cheap relative to finding. It also sharpens the lesson. The
check was cheap because the verifier's input was external and the
discriminating signal was obvious to a non-expert. It was not cheap merely
because it was mechanical.

## A second case, checked the same way: an ADFGVX message from 1918

On 2026-09-17, the day this note was written, prinz reported that GPT-6 Astra
read one of the unsolved German WWI radio messages in the
[Klausis Krypto Kolumne list](https://www.prinzai.com/p/gpt-6-astra-solves-a-wwi-german-radio),
transmitted 27 November 1918 and enciphered with ADFGVX. Most of that corpus
was already solved, by hand in 1918 and by
[Lasry and Niebel (Cryptologia 2017)](https://www.tandfonline.com/doi/abs/10.1080/01611194.2016.1169461)
with hill-climbing since; roughly a dozen were left.

We re-ran this one too. The ciphertext and the substitution square exist only
as page scans of Childs in the post, so we read them off the images,
rebuilt the column order for the key `TRUPPENVERSCHIEBUNG` from the scanned
numbering, undid the transposition and looked up the digraphs:

```text
EINENGLISCHERKREUZEREINLIEGXSEWASTOPOLXS?STENXEINGESCHWADERDERXALLIIERTENFOLGT26STENX
```

"An English cruiser docked at Sevastopol on the ?4th. An allied squadron of
the Allies follows on the 26th." The 170 characters split into eighteen
columns of 9 and one of 8, exactly as the key length requires.

Two features make it a useful companion to MVUEH rather than a repetition.

**The discriminating check is again language, and again cheap.** No indicator
and no re-encryption identity is involved. A wrong key gives noise, this key
gives German, and no calibrated reviewer is needed to tell the two apart.

**An ambiguous cell was resolved by an outside record, not by scoring.** The
square's scan carries handwritten corrections, and one cell reads as either
4 or 8. Our first pass guessed 8 and produced `S8STEN`. The post reads 4 and
cites HMS Canterbury's log putting the ship in Sevastopol on the 24th, with an
allied squadron on the 26th. The log is external to the break, so it fixes
that cell and the message's date arithmetic at once. Compare § 3: the
degree of freedom was closed by a record the solver did not produce.

**Where this case is weaker than MVUEH.** Novelty is unconfirmed: the post
says only "to my knowledge" the message was never decoded, and no custodian
has marked it solved, so the § 6 institutional step has not happened. The key
`TRUPPENVERSCHIEBUNG` is documented as in use from 9 December 1918, twelve
days after this message's date; the post flags the discrepancy and does not
resolve it. Either the recorded key period or the date is wrong. The German
is not in question either way, which is the point: the decryption is checkable
even while its provenance is not.

## Candidate rig change (not applied)

`/bv-dispatch` step 7 classifies a gate as G0 when it is "pass/fail without
human judgment." MVUEH shows two holes in that test. Re-encryption is
pass/fail and G0, and it passes for every candidate. The language check
needs judgment and is still near-decisive. A proposed amendment to step 7:

- Before a gate counts as G0, name a wrong candidate that it would reject.
  If there isn't one, the gate is vacuous and the bead is treated as G2.
- For each gate, record whether its input is held by the solver. External
  input, such as an archive, a registry, or a fixed dataset the claimant
  didn't produce, allows a one-step promotion.

The change is left for review, not applied. It touches dispatch budgets.

## Limitations

- We re-ran the decryption only. We did not re-run the 43,016-batch
  search, the SAT cross-check, or the competitor-key audit. Uniqueness among
  the 923 indicator-compatible keys is taken from the solver's report.
- "Connected German" is our own reading of the output, not a German
  specialist's.
- For the ADFGVX message we transcribed the ciphertext and the square from
  page scans by eye. A transcription slip would garble the plaintext locally,
  which is its own check, but the reading of the 4/8 cell rests on the
  Canterbury log as reported in the post, not on the scan.
- The historical claims are the solver's and Weierud's: the SS-Totenkopf
  supply-service key, the retransmission link to SIPVX, "Waschbusch" as a
  signature, and which Rosenow. The report lists the last three as open.
- The model and agent setup (GPT-6 Astra plus parallel specialists) is
  self-reported. The public package holds search records, not agent
  transcripts, so this note cannot say how the work split between the human
  and the agents.
- The dates in the solver's archive are local records, not externally
  authenticated timestamps, as the report itself notes.
