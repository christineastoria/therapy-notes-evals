from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

# ---------------------------------------------------------------------------
# Option A: use an existing LangSmith dataset (skip upload entirely)
#
# If your golden examples already live in LangSmith, point run_experiments.py
# at that dataset name and delete everything below. You can also pull the
# examples locally for inspection:
#
#   client = Client()
#   dataset = client.read_dataset(dataset_name="Your Existing Dataset Name")
#   examples = list(client.list_examples(dataset_id=dataset.id))
#   for ex in examples:
#       print(ex.inputs, ex.outputs, ex.metadata)
# ---------------------------------------------------------------------------

DATASET_NAME = "Therapy Notes - Golden Examples"

GOLDEN_EXAMPLES = [
    # --- Template Type 1: SOAP ---
    {
        "inputs": {
            "transcript": (
                "Therapist: How have you been feeling since our last session?\n"
                "Patient: Pretty anxious, honestly. Work has been overwhelming. "
                "I've been waking up at 3am almost every night thinking about deadlines.\n"
                "Therapist: How's that affecting your day-to-day?\n"
                "Patient: I'm exhausted and I've been snapping at my partner a lot.\n"
                "Therapist: Have you tried the 4-7-8 breathing we talked about?\n"
                "Patient: Once. It helped a little but I forgot to keep doing it.\n"
                "Therapist: Let's make a small goal — try it three nights this week before bed."
            ),
            "template_type": 1,
        },
        "outputs": {
            "notes": (
                "Subjective: Patient reports significant work-related anxiety and sleep disruption, "
                "waking nightly at approximately 3am due to rumination about deadlines. "
                "Notes daytime fatigue and increased irritability affecting relationship with partner.\n\n"
                "Objective: Patient appeared tired but engaged. Mood described as anxious. "
                "Patient demonstrated insight into impact of stress on daily functioning.\n\n"
                "Assessment: Patient presenting with symptoms consistent with anxiety and occupational stress, "
                "resulting in sleep disruption and interpersonal strain. Previously introduced coping skill "
                "(4-7-8 breathing) has been trialed once with partial adherence.\n\n"
                "Plan: Patient agreed to practice 4-7-8 breathing exercise on three evenings before next session. "
                "Continue to explore sleep hygiene and stress management strategies."
            )
        },
        "metadata": {"template_type": 1},
    },
    {
        "inputs": {
            "transcript": (
                "Therapist: What's been on your mind this week?\n"
                "Patient: I've been feeling really down. I don't enjoy things I used to love, like painting. "
                "I haven't picked up a brush in two months.\n"
                "Therapist: How's your energy and appetite?\n"
                "Patient: Very low energy. I sleep a lot but still feel exhausted. Barely eating.\n"
                "Therapist: Have you had any thoughts of hurting yourself?\n"
                "Patient: No, nothing like that. I just feel gray. Like nothing matters much.\n"
                "Therapist: Let's try activating some small pleasures — even 10 minutes of sketching this week."
            ),
            "template_type": 1,
        },
        "outputs": {
            "notes": (
                "Subjective: Patient reports persistent low mood, anhedonia (loss of interest in painting "
                "for two months), low energy, hypersomnia, and decreased appetite. Denies suicidal ideation.\n\n"
                "Objective: Patient appeared flat in affect, spoke slowly. Mood described as 'gray.' "
                "No evidence of psychomotor agitation or acute distress. Safety screening negative.\n\n"
                "Assessment: Patient presenting with symptoms consistent with a depressive episode including "
                "anhedonia, neurovegetative changes (sleep, appetite, energy), and flat affect. "
                "No safety concerns at this time.\n\n"
                "Plan: Behavioral activation introduced — patient to attempt 10-minute sketching session "
                "before next appointment. Monitor depressive symptoms and safety. "
                "Assess response to behavioral activation at next session."
            )
        },
        "metadata": {"template_type": 1},
    },
    # --- Template Type 2: DAP ---
    {
        "inputs": {
            "transcript": (
                "Therapist: How did the job interview go?\n"
                "Patient: Better than I expected. I was nervous beforehand but once I got in the room "
                "I felt okay. I used the breathing techniques.\n"
                "Therapist: How did you feel afterward?\n"
                "Patient: Proud, actually. Which is new for me. Usually I just think about what I did wrong.\n"
                "Therapist: That's a significant shift. What made the difference?\n"
                "Patient: I've been catching negative thoughts faster, like you said — notice them and name them.\n"
                "Therapist: You're applying the cognitive restructuring skills well. Let's keep building on that."
            ),
            "template_type": 2,
        },
        "outputs": {
            "notes": (
                "Data: Patient reported successful completion of a job interview. Noted pre-interview anxiety "
                "that resolved upon entering the interview. Applied previously learned breathing techniques. "
                "Patient reported feeling proud post-event rather than engaging in habitual negative "
                "self-evaluation. Described active use of cognitive restructuring (noticing and naming "
                "negative thoughts).\n\n"
                "Assessment: Patient demonstrating meaningful progress in anxiety management and cognitive "
                "restructuring. Shift from automatic negative post-event processing to self-efficacy represents "
                "a clinically significant change. Treatment goals related to cognitive distortions show "
                "measurable progress.\n\n"
                "Plan: Continue to reinforce cognitive restructuring skills. Explore additional situations "
                "where skill can be applied. Assess interview outcome and emotional response at next session."
            )
        },
        "metadata": {"template_type": 2},
    },
    {
        "inputs": {
            "transcript": (
                "Therapist: Last week you mentioned wanting to set a boundary with your mother. How did that go?\n"
                "Patient: I told her I couldn't talk every day anymore, maybe just a few times a week. "
                "She got upset and said I was abandoning her.\n"
                "Therapist: How did you respond?\n"
                "Patient: I held my ground. It was hard. I felt guilty for days but I didn't call her back "
                "the next day like I normally would.\n"
                "Therapist: That took real courage. How are you feeling about it now?\n"
                "Patient: Still guilty, but also a little lighter. Like I proved to myself I could do it."
            ),
            "template_type": 2,
        },
        "outputs": {
            "notes": (
                "Data: Patient enacted a planned boundary with mother, communicating preference to speak "
                "several times per week rather than daily. Mother responded negatively, expressing feelings "
                "of abandonment. Patient maintained the boundary and did not revert to the previous pattern "
                "of calling the following day. Patient reported guilt lasting several days alongside a "
                "new sense of relief and self-efficacy.\n\n"
                "Assessment: Patient successfully completed a boundary-setting behavior, representing progress "
                "on goals related to enmeshment and self-assertion. Persistence in the face of relational "
                "pushback is a positive indicator. Ambivalent emotional response (guilt alongside relief) "
                "is developmentally appropriate at this stage of treatment.\n\n"
                "Plan: Continue processing guilt and its function. Reinforce boundary-setting self-efficacy. "
                "Explore next steps in renegotiating the relationship dynamic with mother."
            )
        },
        "metadata": {"template_type": 2},
    },
    # --- Template Type 3: Narrative ---
    {
        "inputs": {
            "transcript": (
                "Therapist: You mentioned a panic attack this week. Can you walk me through it?\n"
                "Patient: I was on the subway and my heart started racing. I thought I was dying. "
                "I got off two stops early and walked the rest of the way.\n"
                "Therapist: How long did it last?\n"
                "Patient: Maybe ten minutes. I called my sister and that helped.\n"
                "Therapist: Good — reaching out was smart. What was happening just before it started?\n"
                "Patient: I was thinking about a presentation I have to give next week. I think that triggered it.\n"
                "Therapist: That's a helpful connection. Let's work on some grounding techniques for the subway."
            ),
            "template_type": 3,
        },
        "outputs": {
            "notes": (
                "Patient presented this session reporting a panic attack experienced on the subway earlier "
                "in the week, characterized by racing heart and fear of dying, lasting approximately ten minutes. "
                "Patient exited the subway early and walked to their destination, and called their sister for "
                "support — an adaptive coping response. Patient and therapist identified anticipatory anxiety "
                "about an upcoming work presentation as the likely precipitant, and patient demonstrated good "
                "insight in making this connection. Session focused on psychoeducation around panic and "
                "exploring grounding techniques for use in triggering environments such as public transit. "
                "Patient engagement was high throughout."
            )
        },
        "metadata": {"template_type": 3},
    },
    {
        "inputs": {
            "transcript": (
                "Therapist: How are things with your grief group?\n"
                "Patient: I actually spoke up for the first time last week. It was scary but people were kind.\n"
                "Therapist: That's a big step. What made you decide to share?\n"
                "Patient: Someone else shared something similar to what happened with my dad and I just felt "
                "like I wasn't alone for a minute.\n"
                "Therapist: That moment of connection sounds powerful.\n"
                "Patient: It was. I've been so isolated since he died. I didn't realize how much I needed "
                "to hear that someone else understood.\n"
                "Therapist: Grief can be very isolating. It sounds like the group is becoming a safe space."
            ),
            "template_type": 3,
        },
        "outputs": {
            "notes": (
                "Patient reported a meaningful milestone this session: speaking up for the first time in "
                "their grief support group following the death of their father. The decision to share was "
                "prompted by hearing another group member describe a similar experience, which created a "
                "felt sense of connection and temporarily reduced isolation. Patient reflected on the profound "
                "isolation experienced since their father's death and noted that the group is beginning to "
                "feel like a safe space. The therapist validated the significance of this social reconnection "
                "as an important step in the grief process. Patient's affect was notably warmer than in "
                "prior sessions, with increased emotional expressiveness and engagement throughout."
            )
        },
        "metadata": {"template_type": 3},
    },
]


def main():
    client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        client.delete_dataset(dataset_id=existing[0].id)
        print(f"Replaced existing dataset: {DATASET_NAME}")

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Golden examples for therapy notes generation — used for hallucination, relevance, and template conformity evaluation.",
    )

    client.create_examples(
        inputs=[e["inputs"] for e in GOLDEN_EXAMPLES],
        outputs=[e["outputs"] for e in GOLDEN_EXAMPLES],
        metadata=[e["metadata"] for e in GOLDEN_EXAMPLES],
        dataset_id=dataset.id,
    )

    print(f"Uploaded {len(GOLDEN_EXAMPLES)} examples to '{DATASET_NAME}'")
    for t in [1, 2, 3]:
        count = sum(1 for e in GOLDEN_EXAMPLES if e["metadata"]["template_type"] == t)
        print(f"  Template {t}: {count} examples")


if __name__ == "__main__":
    main()
